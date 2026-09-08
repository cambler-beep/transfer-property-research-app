import streamlit as st
import pandas as pd
import re
import time
import difflib
from google import genai
from google.genai import types
from google.genai.errors import ClientError

# -----------------------------------------
# 1. SETUP & CONFIGURATION
# -----------------------------------------
st.set_page_config(page_title="Property Transition Researcher", page_icon="🏢")
st.title("🏢 Property Transition AI Researcher")

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

if not GEMINI_API_KEY:
    st.error("Please add your GEMINI_API_KEY in the Streamlit Secrets settings.")
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)

SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1SJQ7YWUVcSSBKCKMSQFlMInxBTOeiLoJal6g2EHwhUU/export?format=csv"
    "&gid=1440084512"
)

# Try these models in order. If your account's available models change,
# update this list (check "available models" in Google AI Studio).
MODELS_TO_TRY = ["gemini-2.5-flash", "gemini-2.0-flash"]


# -----------------------------------------
# 2. TEXT / MATCHING HELPERS
# -----------------------------------------
def normalize_text(text):
    """Strip non-breaking spaces (\\xa0), tabs, and extra whitespace that
    sneak in when copy-pasting directly out of Google Sheets."""
    if text is None:
        return ""
    return re.sub(r"[\xa0\s]+", " ", str(text)).strip()


def clean_search_term(raw_name):
    """Strip internal deal suffixes ('/ SM Transfer', '(RKW)', etc.) to get
    the core property name for matching and for the research prompt."""
    normalized = normalize_text(raw_name)
    clean = normalized.split("/")[0].split("(")[0]
    clean = re.sub(r"(?i)\b(transfer|sop|retention|deal|sm|ai|tt|ttfl|aibi)\b", "", clean)
    return normalize_text(clean)


@st.cache_data(ttl=300, show_spinner=False)
def load_sheet():
    """Pull tab 1 ('Properties Pending Transfer') as a DataFrame, with
    normalized column names. Cached 5 min so repeated lookups don't
    re-hit Google Sheets every click."""
    df = pd.read_csv(SHEET_URL, skiprows=1, dtype=str).fillna("")
    df.columns = [normalize_text(c) for c in df.columns]
    return df


def get_col(row, *possible_headers):
    """Case-insensitive, whitespace-tolerant column lookup. This is what
    fixes the 'address shows in the sheet but app says Not Specified' bug —
    it no longer depends on the header matching byte-for-byte."""
    if row is None:
        return ""
    wanted = {normalize_text(h).lower() for h in possible_headers}
    for key in row.index:
        if normalize_text(key).lower() in wanted:
            val = normalize_text(row[key])
            if val and val.lower() not in ("nan", "none", "#n/a"):
                return val
    return ""


def find_property_row(df, search_term):
    """Multi-tier match so slightly-off pastes still find the right row.
    Returns (row, match_quality) where match_quality is 'exact', 'fuzzy',
    or None if nothing usable was found."""
    target_clean = clean_search_term(search_term).lower()
    raw_clean = normalize_text(search_term).lower()

    def opp(row):
        return normalize_text(get_col(row, "Opportunity Name")).lower()

    def prop(row):
        return normalize_text(get_col(row, "Property Name")).lower()

    # Tier 1: exact / substring match on opportunity or property name
    for _, row in df.iterrows():
        o, p = opp(row), prop(row)
        if not (o or p):
            continue
        if target_clean and (target_clean in o or target_clean in p or o in target_clean or p in target_clean):
            return row, "exact"
        if raw_clean and (raw_clean in o or raw_clean in p):
            return row, "exact"

    # Tier 2: every significant token in the search term appears somewhere
    tokens = [t for t in target_clean.split() if len(t) > 2]
    if tokens:
        for _, row in df.iterrows():
            combined = f"{opp(row)} {prop(row)}"
            if combined.strip() and all(t in combined for t in tokens):
                return row, "exact"

    # Tier 3: fuzzy match against the Opportunity Name list
    opp_list = [opp(row) for _, row in df.iterrows()]
    matches = difflib.get_close_matches(target_clean, opp_list, n=1, cutoff=0.45)
    if matches:
        idx = opp_list.index(matches[0])
        return df.iloc[idx], "fuzzy"

    return None, None


# -----------------------------------------
# 3. RESEARCH (Gemini + native Google Search grounding)
# -----------------------------------------
def generate_research_note(prop_name, full_address, prev_owner, prev_sop, known_owner, known_manager):
    clean_name = clean_search_term(prop_name)

    known_lines = []
    if known_owner:
        known_lines.append(f"- HubSpot currently lists the owner as: {known_owner} (verify/update this)")
    if known_manager:
        known_lines.append(f"- HubSpot currently lists the manager as: {known_manager} (verify/update this)")
    known_block = "\n".join(known_lines) if known_lines else "- No current owner/manager on file in HubSpot yet."

    prompt = f"""
Act as a Senior Commercial Real Estate (CRE) & Senior Housing Research Analyst.

YOUR CRITICAL TASK:
Use your built-in Google Search tool to browse the live internet and find the
most up-to-date property ownership and management details for the asset
below. Check the property's own website footer/about page, press releases,
and commercial real estate news for the ownership and management transition.

PROPERTY TO RESEARCH:
- Name: {clean_name}
- Location / Address: {full_address}
- Previous Owner on file (CRM): {prev_owner or "Unknown"}
- Previous Manager/SOP on file (CRM): {prev_sop or "Unknown"}
{known_block}

TARGET INSTRUCTIONS:
1. CURRENT OWNER: Identify the buyer, purchasing entity (LLC), holding company, REIT, or parent entity.
2. CURRENT MANAGER / OPERATOR: Identify the active property manager or operating company. Check the footer of the property's official website.
3. PREVIOUS OWNER & MANAGER: Confirm or correct the seller/developer and former manager/SOP operator listed above.
4. HEADQUARTERS STATES: New Owner HQ State and Current Manager HQ State (City, State).
5. COMPANY DOMAIN: Official domain name of the buyer or property manager/operator.
6. REBRAND STATUS: Any name changes or rebranding.
7. OVERVIEW: Physical specs, building style, unit/bed count.
8. VALUE-ADD / RENOVATIONS: Only list specific capital improvement plans if found.
9. TRANSACTION CONTEXT: Purchase price, sale date, buyer, seller.

If you cannot verify a field after searching, write "Not found" rather than
guessing. Do not fabricate names, dates, or dollar amounts.

Return strictly in the following vertical layout without raw markdown symbols:

📋 Property Transition Research Note
Property: {clean_name} ({full_address})

Research Summary:
[2-3 sentence overview]

Ownership & Management:
• Current Owner: [Owner Name]
• Previous Owner: [Previous Owner]
• New Owner HQ State: [City, State]
• Current Manager: [Current Property Manager]
• Current Manager HQ State: [City, State]
• Previous Manager: [Previous Manager]

HubSpot Info:
• Company Domain: [Official domain name]
• Account Executive: [Look up in HubSpot manually]

Property Details & Context:
• Rebrand Status: [Primary and secondary community branding]
• Overview: [Property physical details, unit/bed count]
• Value-Add / Renovations: [Specific renovation plans]
• Transaction Context: [Purchase price, sale date, seller, buyer]

Sources & Evidence:
• List the URLs you found during your Google Search.
"""

    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    config = types.GenerateContentConfig(tools=[grounding_tool])

    error_logs = []
    for model_id in MODELS_TO_TRY:
        # Up to 2 retries on rate limits with backoff, then move to next model.
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_id, contents=prompt, config=config
                )
                if response and response.text:
                    return response.text, None
                error_logs.append(f"{model_id}: empty response")
                break
            except ClientError as e:
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    if attempt == 0:
                        time.sleep(15)
                        continue
                    error_logs.append(f"{model_id}: quota exhausted (429)")
                    break
                error_logs.append(f"{model_id}: {msg}")
                break
            except Exception as e:
                error_logs.append(f"{model_id}: {str(e)}")
                break

    quota_hit = any("429" in e or "quota" in e.lower() for e in error_logs)
    return None, ("QUOTA" if quota_hit else "ERROR", " | ".join(error_logs))


# -----------------------------------------
# 4. STREAMLIT UI
# -----------------------------------------
st.write(
    "Paste an Opportunity Name from the **Properties Pending Transfer** tab "
    "(e.g. `Lumiere Chandler / Transfer`) to pull the sheet data and run "
    "live research."
)

opportunity_input = st.text_input("Opportunity Name")

if st.button("Generate Research Note"):
    if not opportunity_input:
        st.warning("Please enter an Opportunity Name first.")
        st.stop()

    try:
        df = load_sheet()
    except Exception as e:
        st.error(f"Couldn't load the Google Sheet: {e}")
        st.stop()

    row, quality = find_property_row(df, opportunity_input)

    if row is None:
        st.error(
            "Couldn't find that Opportunity Name in the sheet. "
            "Double check the spelling, or try pasting just the property name."
        )
        st.stop()

    prop_name = get_col(row, "Property Name") or opportunity_input
    street = get_col(row, "Street", "Street Address")
    city = get_col(row, "City")
    state = get_col(row, "State/Province", "State")
    zip_code = get_col(row, "Zip/Postal Code", "Zip Code", "Zip")
    prev_owner = get_col(row, "Previous License Account", "Previous Owner")
    prev_sop = get_col(row, "Previous SOP", "Previous Manager")
    known_owner = get_col(row, "HS Property Owner")
    known_manager = get_col(row, "HS Property Manager")

    addr_parts = [p for p in [street, city, state, zip_code] if p]
    full_address = ", ".join(addr_parts) if addr_parts else "Address Not Specified"

    if quality == "fuzzy":
        st.info(f"Closest match found: **{get_col(row, 'Opportunity Name')}** — {full_address}")
    else:
        st.caption(f"Matched: {get_col(row, 'Opportunity Name')} — {full_address}")

    with st.spinner("🔍 Researching via live Google Search..."):
        final_note, err = generate_research_note(
            prop_name, full_address, prev_owner, prev_sop, known_owner, known_manager
        )

    if final_note:
        st.success("Research complete! Click the copy icon in the top-right of the box below.")
        st.code(final_note, language="text")
    else:
        kind, detail = err
        if kind == "QUOTA":
            st.error(
                "⚠️ Gemini API quota exhausted (429). This is a Google billing/rate-limit "
                "issue, not a code bug — either wait ~60 seconds (per-minute limit) or "
                "until tomorrow (daily limit), or add billing to your Google AI Studio "
                "account to remove the free-tier cap.\n\nDetails: " + detail
            )
        else:
            st.error(f"⚠️ Gemini API error:\n\n{detail}")

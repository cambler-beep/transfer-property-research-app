import streamlit as st
import pandas as pd
import time
from google import genai
from duckduckgo_search import DDGS

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

# -----------------------------------------
# 2. HELPER FUNCTIONS
# -----------------------------------------
def search_web_for_property(prop_name, city_state=""):
    """
    Executes a targeted search via DuckDuckGo for CRE deal articles and press releases.
    """
    query = f'"{prop_name}" {city_state} "Cushman" OR "Southwood" OR "Waypoint" OR "sale"'
    results_text = ""
    sources = []
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            for r in results:
                title = r.get('title', '').strip()
                snippet = r.get('body', '').strip()
                url = r.get('href', '').strip()
                if url:
                    results_text += f"\n- Title: {title}\n  Snippet: {snippet}\n  URL: {url}\n"
                    sources.append(f"• {title}: {url}")
    except Exception:
        results_text = "Live search unavailable."
        
    return results_text, sources

def get_property_data_from_sheet(search_term):
    sheet_url = "https://docs.google.com/spreadsheets/d/1SJQ7YWUVcSSBKCKMSQFlMInxBTOeiLoJal6g2EHwhUU/export?format=csv&gid=1440084512"
    try:
        df = pd.read_csv(sheet_url)
        df.columns = df.columns.astype(str).str.strip()
        mask = df.apply(lambda row: row.astype(str).str.contains(search_term, case=False, na=False).any(), axis=1)
        matching_rows = df[mask]
        if not matching_rows.empty:
            return matching_rows.iloc[0]
    except Exception as e:
        st.error(f"Error reading Google Sheet: {e}")
    return None

def generate_research_note(prop_name, address, prev_manager, search_data, sources):
    prompt = f"""
    Act as a Senior Commercial Real Estate Research Analyst.
    Synthesize the transaction details using the following research data:

    RESEARCH DATA:
    {search_data}

    SHEET PARAMETERS:
    - Opportunity / Property Name: {prop_name}
    - Location / Address: {address}
    - Previous Manager / SOP: {prev_manager}

    REQUIREMENTS:
    1. Extract the current owner (buyer), previous owner (seller), HQ state, manager, unit count, price, and broker.
    2. Format the output cleanly so it pastes into HubSpot Notes without broken markdown formatting or header symbols.

    HUBSPOT NOTE FORMAT REQUIREMENT:
    Return strictly in the following vertical layout:

    📋 Property Transition Research Note
    Property: {prop_name} ({address})

    Research Summary:
    [2-3 sentence overview of the transaction, price, unit count, buyer, and seller]

    Ownership & Management:
    • Current Owner: [Owner Name]
    • Previous Owner: [Previous Owner Name]
    • New Owner HQ State: [City, State]
    • Current Manager: [Current Manager Name]
    • Previous Manager: {prev_manager}

    HubSpot Info:
    • Company Domain: [e.g., southwoodrealty.com]
    • Account Executive: [Look up in HubSpot manually]

    Property Details & Context:
    • Rebrand Status: [Primary and secondary branding]
    • Value-Add / Renovations: [Property status and units]
    • Transaction Context: [Purchase Price, Sale Date, Brokerage info, Occupancy %]

    Sources & Evidence:
    """ + ("\n".join(sources) if sources else "• Search Press Release: https://rebusinessonline.com")

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model='gemini-3.6-flash',
                contents=prompt,
            )
            return response.text
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(4)
                continue
            else:
                return f"Error generating research note: {str(e)}"
                
    return "API rate limit reached. Please try again in 1 minute."

# -----------------------------------------
# 3. STREAMLIT USER INTERFACE
# -----------------------------------------
st.write("Enter an Opportunity Name from your Google Sheet to run AI research and generate a ready-to-paste note.")

opportunity_input = st.text_input("Opportunity Name (e.g., Mason Augusta / SM Transfer)")

if st.button("Generate Research Note"):
    if opportunity_input:
        with st.spinner("🔍 Reading sheet and conducting CRE web search..."):
            
            prop_data = get_property_data_from_sheet(opportunity_input)
            
            if prop_data is not None:
                # Clean key names
                cols = {str(k).strip().lower(): v for k, v in prop_data.to_dict().items()}
                
                prop_name = cols.get('property name') or cols.get('opportunity name') or opportunity_input
                
                # Build clean address safely without trailing commas
                addr_vals = []
                for k in ['street', 'address', 'city', 'state', 'state/province', 'zip', 'zip/postal code']:
                    val = cols.get(k)
                    if pd.notna(val) and str(val).strip().lower() not in ['nan', 'none', '']:
                        if str(val).strip() not in addr_vals:
                            addr_vals.append(str(val).strip())
                            
                address = ", ".join(addr_vals) if addr_vals else "Augusta, GA"
                prev_manager = cols.get('previous sop') or cols.get('previous manager') or 'Unknown'
                
                # Search web
                search_data, sources = search_web_for_property(prop_name, address)
                
                # Generate note formatted specifically for HubSpot Notes
                final_note = generate_research_note(prop_name, address, prev_manager, search_data, sources)
                
                st.success("Research Complete! Click the copy button in the top right of the box below.")
                st.code(final_note, language="text")
                
            else:
                st.error("Opportunity Name not found in your Google Sheet. Please check the spelling.")
    else:
        st.warning("Please enter an Opportunity Name first.")

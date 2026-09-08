import streamlit as st
import pandas as pd
import time
import re
from google import genai

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
def clean_search_term(raw_name):
    """Strips internal deal tags (/ SM Transfer, / Transfer, SOP) to extract pure name."""
    if not raw_name:
        return ""
    clean = str(raw_name).split('/')[0].split('(')[0]
    clean = re.sub(r'(?i)\b(transfer|sop|retention|deal|sm|ai)\b', '', clean)
    return clean.strip()

def search_web_for_property(prop_clean_name, street, city, state):
    """Executes DuckDuckGo searches targeted for Multifamily & Senior Living."""
    from duckduckgo_search import DDGS
    base_name = clean_search_term(prop_clean_name)
    
    queries = []
    if street and city:
        queries.append(f'"{street}" "{city}" sale OR acquired OR owner OR "$"')
    if base_name and city:
        queries.append(f'"{base_name}" "{city}" sale OR owner OR manager OR rebranded')
    if base_name:
        queries.append(f'"{base_name}" "acquired by" OR "Senior Living" OR "managed by"')
    
    results_text = ""
    sources = []
    seen_urls = set()
    
    try:
        with DDGS() as ddgs:
            for q in queries[:3]:
                results = list(ddgs.text(q, max_results=5))
                for r in results:
                    url = r.get('href', '').strip()
                    title = r.get('title', '').strip()
                    snippet = r.get('body', '').strip()
                    
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        results_text += f"\n- Title: {title}\n  Snippet: {snippet}\n  URL: {url}\n"
                        sources.append(f"• {title}: {url}")
    except Exception:
        results_text = "Live search data processing."
        
    return results_text, sources

def get_property_data_from_sheet(search_term):
    """Reads Google Sheet CSV using the original logic, with a fix for NaN float errors."""
    sheet_url = "https://docs.google.com/spreadsheets/d/1SJQ7YWUVcSSBKCKMSQFlMInxBTOeiLoJal6g2EHwhUU/export?format=csv&gid=1440084512"
    try:
        df = pd.read_csv(sheet_url)
        df.columns = df.columns.astype(str).str.strip()
        
        target_clean = clean_search_term(search_term).lower()
        
        for _, row in df.iterrows():
            opp_name = str(row.get('Opportunity Name', '')).lower()
            prop_name = str(row.get('Property Name', '')).lower()
            
            # THE FIX: .astype(str) prevents the sequence item expected str, float found error
            row_str = " ".join(row.astype(str).values).lower()
            
            if target_clean in opp_name or target_clean in prop_name or target_clean in row_str:
                return row
                
    except Exception as e:
        st.error(f"Error reading Google Sheet: {e}")
    return None

def generate_research_note(prop_name, full_address, prev_owner, prev_sop, search_data, sources):
    """Generates structured CRE research note for Multifamily & Senior Living assets."""
    clean_name = clean_search_term(prop_name)
    
    prompt = f"""
    Act as a Senior Commercial Real Estate (CRE) & Senior Housing Research Analyst.
    Synthesize property transaction, ownership, management/operator, and rebranding details strictly using the provided search research data.

    SEARCH RESEARCH DATA:
    {search_data}

    GOOGLE SHEET PARAMETERS:
    - Opportunity / Property Name: {clean_name}
    - Location / Address: {full_address}
    - Previous Owner / Account: {prev_owner}
    - Previous Manager / SOP: {prev_sop}

    TARGET INSTRUCTIONS:
    1. CURRENT OWNER: Identify the buyer, purchasing entity (LLC), holding company, REIT, or parent entity (e.g. Canyon Multifamily Impact Fund).
    2. CURRENT MANAGER / OPERATOR: Identify active property manager or operating company (e.g. Cannon Management).
    3. PREVIOUS OWNER & MANAGER: Identify seller/developer (e.g. Bridge Investment Group) and former property manager/SOP operator.
    4. HEADQUARTERS STATES: Identify New Owner HQ State and Current Manager HQ State (City, State).
    5. COMPANY DOMAIN: Identify official domain name of the buyer or property manager/operator.
    6. REBRAND STATUS: Identify any name changes or rebranding (e.g. Formerly Coronado Palms / Palmilla Villas; rebranded to Oro Apartments).
    7. OVERVIEW: Always include an Overview bullet detailing physical specs, building style, unit/bed count (e.g. 168-unit community), care levels (if Senior Living), and key amenities.
    8. VALUE-ADD / RENOVATIONS: Only list specific capital improvement plans if explicitly found in research. Otherwise, strictly state "N/A".
    9. TRANSACTION CONTEXT: Summarize purchase price, sale date, buyer, seller, and brokerage details.

    HUBSPOT NOTE FORMAT REQUIREMENT:
    Return strictly in the following vertical layout without raw markdown symbols like ### or **:

    📋 Property Transition Research Note
    Property: {clean_name} ({full_address})

    Research Summary:
    [2-3 sentence overview of the acquisition/transition, buyer, seller, transaction price, unit/bed count, care levels if applicable, and rebranding details]

    Ownership & Management:
    • Current Owner: [Owner Name / Holding Entity / Purchasing LLC]
    • Previous Owner: {prev_owner if prev_owner != 'Unknown' else '[Previous Owner / Seller Name]'}
    • New Owner HQ State: [City, State of HQ]
    • Current Manager: [Current Property Manager / Operating Company]
    • Current Manager HQ State: [City, State of HQ]
    • Previous Manager: {prev_sop if prev_sop != 'Unknown' else '[Previous Manager Name]'}

    HubSpot Info:
    • Company Domain: [Official domain name]
    • Account Executive: [Look up in HubSpot manually]

    Property Details & Context:
    • Rebrand Status: [Primary and secondary community branding]
    • Overview: [Property physical details, unit/bed count, care levels if senior living, building style, amenities]
    • Value-Add / Renovations: [Specific renovation plans if explicitly found in research; otherwise state N/A]
    • Transaction Context: [Purchase price, sale date, seller, buyer, brokerage details, or operational transition]

    Sources & Evidence:
    """ + ("\n".join(sources[:4]) if sources else "• Search Public Records: https://www.google.com")

    models_to_try = ['gemini-3.6-flash', 'gemini-3.1-flash-lite']

    for model_id in models_to_try:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                )
                if response and response.text:
                    return response.text
            except Exception as e:
                err_msg = str(e)
                if "503" in err_msg or "UNAVAILABLE" in err_msg or "429" in err_msg:
                    time.sleep(2 * (attempt + 1))
                    continue
                else:
                    break

    return "Google AI servers are currently experiencing high demand. Please wait 10 seconds and click Generate again."

# -----------------------------------------
# 3. STREAMLIT USER INTERFACE
# -----------------------------------------
st.write("Enter an Opportunity Name from your Google Sheet to run AI research and generate a ready-to-paste note.")

opportunity_input = st.text_input("Opportunity Name (e.g., Property Name / Transfer)")

if st.button("Generate Research Note"):
    if opportunity_input:
        with st.spinner("🔍 Reading sheet and conducting CRE web search..."):
            
            row = get_property_data_from_sheet(opportunity_input)
            
            if row is not None:
                def get_col_val(header_name):
                    val = row.get(header_name)
                    if pd.notna(val) and str(val).strip().lower() not in ['nan', 'none', '']:
                        return str(val).strip()
                    return ''

                opp_name = get_col_val('Opportunity Name') or opportunity_input
                prop_name = get_col_val('Property Name') or opp_name
                street = get_col_val('Street')
                city = get_col_val('City')
                state = get_col_val('State/Province')
                zip_code = get_col_val('Zip/Postal Code')
                
                prev_owner = get_col_val('Previous License Account') or 'Unknown'
                prev_sop = get_col_val('Previous SOP') or 'Unknown'
                
                # Dynamic address builder
                addr_parts = [p for p in [street, city, state, zip_code] if p]
                if addr_parts:
                    full_address = ", ".join(addr_parts)
                elif city or state:
                    full_address = ", ".join([p for p in [city, state] if p])
                else:
                    full_address = "Address Not Specified"
                
                # Execute Python web search
                search_data, sources = search_web_for_property(prop_name, street, city, state)
                
                # Generate research note
                final_note = generate_research_note(prop_name, full_address, prev_owner, prev_sop, search_data, sources)
                
                st.success("Research Complete! Click the copy button in the top right of the box below.")
                st.code(final_note, language="text")
                
            else:
                st.error("Opportunity Name not found in your Google Sheet. Please check the spelling.")
    else:
        st.warning("Please enter an Opportunity Name first.")

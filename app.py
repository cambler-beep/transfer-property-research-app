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
def search_web_for_property(prop_name, address):
    """
    Executes targeted web search using DuckDuckGo to pull real CRE press release text,
    transaction metrics, and listing URLs prior to running Gemini.
    """
    search_query = f'"{prop_name}" "Cushman" OR "Southwood" OR "Waypoint" OR "sale"'
    results_text = ""
    sources = []
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(search_query, max_results=5))
            for r in results:
                title = r.get('title', '')
                snippet = r.get('body', '')
                url = r.get('href', '')
                results_text += f"\n- Title: {title}\n  Snippet: {snippet}\n  URL: {url}\n"
                sources.append(f"* [{title}]: {url}")
    except Exception as e:
        results_text = "Search unavailable."
        
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
    Extract and structure the transaction details using the following live web search research data:

    WEB SEARCH DATA:
    {search_data}

    PROPERTY PARAMETERS FROM SHEET:
    - Opportunity / Property Name: {prop_name}
    - Location / Address: {address}
    - Previous Manager / SOP: {prev_manager}

    TARGET CRE DATA REQUIREMENTS:
    1. CURRENT OWNER & MANAGER: Identify buyer (e.g., Southwood Realty Co.) and current property management.
    2. PREVIOUS OWNER & DEVELOPER: Identify seller (e.g., Waypoint Residential) and former management/license account (e.g., Greystar).
    3. NEW OWNER HQ STATE: Corporate HQ state of buyer (e.g., Gastonia, NC or North Carolina).
    4. COMPANY DOMAIN: Domain name of current owner (e.g., southwoodrealty.com).
    5. TRANSACTION METRICS: Purchase Price (e.g., $87M), sale date (mid-2026/July 2026), unit count (e.g., 462 units), occupancy rate at sale (e.g., 95%), and listing brokers (e.g., Cushman & Wakefield).
    6. BRANDING / REBRAND: Note primary branding (Mason Augusta) and secondary branding (The Mason).

    OUTPUT FORMAT REQUIREMENT:
    Return strictly in the following vertical layout:

    ### 📋 Property Transition Research Note
    **Property:** {prop_name} ({address})

    **Research Summary**
    [Insert 2-3 sentence overview of the acquisition, including transaction price, unit count, and buyer/seller]

    **Ownership & Management**
    * **Current Owner:** [Owner Name]
    * **Previous Owner:** [Previous Owner / Developer Name]
    * **New Owner HQ State:** [City, State of HQ]
    * **Current Manager:** [Current Manager Name]
    * **Previous Manager:** {prev_manager}

    **HubSpot Info**
    * **Company Domain:** [e.g., southwoodrealty.com]
    * **Account Executive:** [Look up in HubSpot manually]

    **Property Details & Context**
    * **Rebrand Status:** [Primary & secondary branding details]
    * **Value-Add / Renovations:** [Property status, e.g., stabilized 462-unit multi-phase development]
    * **Transaction Context:** [Purchase Price, Sale Date, Brokerage info, Occupancy %]

    **Sources/Evidence:**
    """ + "\n".join(sources if sources else ["* [Press Release Search]: https://rebusinessonline.com"])

    # Call the active gemini-3.6-flash model
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model='gemini-3.6-flash',
                contents=prompt,
            )
            return response.text
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(5)
                continue
            else:
                return f"Error generating research note: {str(e)}"
                
    return "API rate limit reached. Please wait 1 minute and try again."

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
                cols = {str(k).strip().lower(): v for k, v in prop_data.to_dict().items()}
                
                prop_name = cols.get('property name') or cols.get('opportunity name') or opportunity_input
                street = cols.get('street', '')
                city = cols.get('city', '')
                state = cols.get('state/province') or cols.get('state', '')
                zip_code = cols.get('zip/postal code') or cols.get('zip', '')
                
                address_parts = [str(p) for p in [street, city, state, zip_code] if pd.notna(p) and str(p).strip() != 'nan']
                address = ", ".join(address_parts) if address_parts else "Address Not Provided"
                
                prev_manager = cols.get('previous sop') or cols.get('previous manager') or 'Unknown'
                
                # Fetch web search data directly
                search_data, sources = search_web_for_property(prop_name, address)
                
                # Run Gemini Research
                final_note = generate_research_note(prop_name, address, prev_manager, search_data, sources)
                
                st.success("Research Complete! Click the copy button in the top right of the box below.")
                st.code(final_note, language="markdown")
                
            else:
                st.error("Opportunity Name not found in your Google Sheet. Please check the spelling.")
    else:
        st.warning("Please enter an Opportunity Name first.")

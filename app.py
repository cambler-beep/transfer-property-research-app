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
def search_web_for_property(prop_name, street, city, state):
    """
    Executes a targeted search using the exact property name, street address, 
    and city/state extracted from your Google Sheet.
    """
    base_name = prop_name.split('/')[0].replace('Transfer', '').strip()
    
    # Priority search query construction
    if street and city:
        query = f'"{base_name}" "{street}" "{city}" owner manager acquisition sale'
    elif city and state:
        query = f'"{base_name}" "{city}" "{state}" owner manager acquisition sale'
    else:
        query = f'"{base_name}" owner manager acquisition sale'
        
    results_text = ""
    sources = []
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=6))
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
        # Strip trailing/leading spaces from column headers
        df.columns = df.columns.astype(str).str.strip()
        
        # Search for exact term across all columns
        mask = df.apply(lambda row: row.astype(str).str.contains(search_term, case=False, na=False).any(), axis=1)
        matching_rows = df[mask]
        if not matching_rows.empty:
            return matching_rows.iloc[0]
    except Exception as e:
        st.error(f"Error reading Google Sheet: {e}")
    return None

def generate_research_note(prop_name, full_address, prev_owner, prev_sop, search_data, sources):
    prompt = f"""
    Act as a Senior Commercial Real Estate Research Analyst.
    Extract the real transition and acquisition details strictly using the provided live web research data. Do not invent details or reuse data from unrelated properties.

    LIVE WEB SEARCH RESULTS:
    {search_data}

    PROPERTY PARAMETERS FROM GOOGLE SHEET:
    - Opportunity / Property Name: {prop_name}
    - Location / Address: {full_address}
    - Previous Owner / Account: {prev_owner}
    - Previous Manager / SOP: {prev_sop}

    TARGET INSTRUCTIONS:
    1. Identify the NEW Owner / Buyer and NEW Property Manager from search snippets (e.g., Camden Property Trust).
    2. Identify the PREVIOUS Owner / Seller or Developer (e.g., Cortland).
    3. Identify New Owner HQ State (e.g., Houston, TX for Camden).
    4. Identify Rebrand or Community Name changes (e.g., Cortland Santos Flats -> Camden Brandon).
    5. Extract transaction details, purchase price, sale date, and unit count if present in search data.
    6. If a piece of data is truly unknown in search results, state "Unknown" or "Not specified in press release".

    HUBSPOT NOTE FORMAT REQUIREMENT:
    Return strictly in the following vertical layout:

    📋 Property Transition Research Note
    Property: {prop_name} ({full_address})

    Research Summary:
    [2-3 sentence overview of the acquisition, new owner/manager, transaction details, and rebranding]

    Ownership & Management:
    • Current Owner: [Owner Name]
    • Previous Owner: {prev_owner if prev_owner != 'Unknown' else '[Previous Owner Name]'}
    • New Owner HQ State: [City, State]
    • Current Manager: [Current Manager Name]
    • Previous Manager: {prev_sop if prev_sop != 'Unknown' else '[Previous Manager Name]'}

    HubSpot Info:
    • Company Domain: [Domain of current owner, e.g. camdenliving.com]
    • Account Executive: [Look up in HubSpot manually]

    Property Details & Context:
    • Rebrand Status: [Primary and secondary branding details]
    • Value-Add / Renovations: [Property details & unit count]
    • Transaction Context: [Transaction details, sale date, occupancy]

    Sources & Evidence:
    """ + ("\n".join(sources[:4]) if sources else "• Search Press Release: https://www.google.com")

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

opportunity_input = st.text_input("Opportunity Name (e.g., Cortland Santos Flats / Transfer)")

if st.button("Generate Research Note"):
    if opportunity_input:
        with st.spinner("🔍 Reading sheet and conducting CRE web search..."):
            
            row = get_property_data_from_sheet(opportunity_input)
            
            if row is not None:
                # Helper to fetch string safely by exact header name
                def get_col_val(header_name):
                    val = row.get(header_name)
                    if pd.notna(val) and str(val).strip().lower() not in ['nan', 'none', '']:
                        return str(val).strip()
                    return ''

                # Exact column header mapping from Google Sheet
                opp_name = get_col_val('Opportunity Name') or opportunity_input
                prop_name = get_col_val('Property Name') or opp_name
                street = get_col_val('Street')
                city = get_col_val('City')
                state = get_col_val('State/Province')
                zip_code = get_col_val('Zip/Postal Code')
                
                prev_owner = get_col_val('Previous License Account') or 'Unknown'
                prev_sop = get_col_val('Previous SOP') or 'Unknown'
                
                # Build Address String
                addr_parts = [p for p in [street, city, state, zip_code] if p]
                full_address = ", ".join(addr_parts) if addr_parts else "Location Not Specified"
                
                # Execute web search using exact Property Name, Street, City, and State
                search_data, sources = search_web_for_property(prop_name, street, city, state)
                
                # Generate research note
                final_note = generate_research_note(prop_name, full_address, prev_owner, prev_sop, search_data, sources)
                
                st.success("Research Complete! Click the copy button in the top right of the box below.")
                st.code(final_note, language="text")
                
            else:
                st.error("Opportunity Name not found in your Google Sheet. Please check the spelling.")
    else:
        st.warning("Please enter an Opportunity Name first.")

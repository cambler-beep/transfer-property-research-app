import streamlit as st
import pandas as pd
import time
import re
from google import genai
from google.genai import types

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
    """Clean internal deal tags like '/ Transfer', 'Transfer', 'SOP' from search terms."""
    if not raw_name:
        return ""
    clean = raw_name.split('/')[0].split('(')[0]
    clean = re.sub(r'(?i)\b(transfer|sop|retention|deal)\b', '', clean)
    return clean.strip()

def get_property_data_from_sheet(search_term):
    """
    Reads Google Sheet CSV and performs fuzzy matching against Opportunity Name 
    and Property Name columns to ensure exact Street, City, and State are extracted.
    """
    sheet_url = "https://docs.google.com/spreadsheets/d/1SJQ7YWUVcSSBKCKMSQFlMInxBTOeiLoJal6g2EHwhUU/export?format=csv&gid=1440084512"
    try:
        df = pd.read_csv(sheet_url)
        df.columns = df.columns.astype(str).str.strip()
        
        target_clean = clean_search_term(search_term).lower()
        
        # Search row by row with substring and cleaned string comparison
        for _, row in df.iterrows():
            opp_name = str(row.get('Opportunity Name', '')).lower()
            prop_name = str(row.get('Property Name', '')).lower()
            
            if target_clean in opp_name or target_clean in prop_name or target_clean in str(row.values).lower():
                return row
                
    except Exception as e:
        st.error(f"Error reading Google Sheet: {e}")
    return None

def generate_research_note(prop_name, full_address, prev_owner, prev_sop):
    """
    Prompts Gemini 3.6 Flash using Google Search Grounding to query public records,
    CRE deal filings, and REIT management transitions.
    """
    clean_name = clean_search_term(prop_name)
    
    prompt = f"""
    Act as a Commercial Real Estate (CRE) Research Analyst.
    Perform a targeted live web search for actual property records, corporate ownership filings (LLCs/REITs), and property management records for:
    - Property / Community Name: {clean_name}
    - Location / Address: {full_address}
    - Known/Previous Account: {prev_owner}
    - Known/Previous Manager: {prev_sop}

    TARGET CRE SEARCH INSTRUCTIONS:
    1. Search public property records, assessor data, and corporate ownership filings (e.g. BREIT, Blackstone, Greystar, AIR Communities, Cortland, Camden).
    2. Identify the CURRENT OWNER / HOLDING ENTITY (e.g. Breit Mf Lumiere Chandler LLC / Blackstone BREIT).
    3. Identify the CURRENT PROPERTY MANAGER (e.g. AIR Communities / Apartment Income REIT Corp.).
    4. Identify PREVIOUS OWNER / DEVELOPER and PREVIOUS MANAGER if applicable.
    5. Identify New Owner Corporate Headquarters State.
    6. Identify Company Domain (e.g. aircommunities.com or breit.com).
    7. Summarize transaction context, unit count, and rebranding details.

    HUBSPOT NOTE FORMAT REQUIREMENT:
    Return strictly in the following vertical layout without raw markdown symbols like ### or **:

    📋 Property Transition Research Note
    Property: {clean_name} ({full_address})

    Research Summary:
    [2-3 sentence overview of the ownership entity, management company, unit count, and transaction context]

    Ownership & Management:
    • Current Owner: [Owner Name / Holding LLC, e.g. Breit Mf Lumiere Chandler LLC / Blackstone BREIT]
    • Previous Owner: {prev_owner if prev_owner != 'Unknown' else '[Previous Owner / Developer Name]'}
    • New Owner HQ State: [City, State of HQ]
    • Current Manager: [Current Property Manager, e.g. AIR Communities]
    • Previous Manager: {prev_sop if prev_sop != 'Unknown' else '[Previous Manager Name]'}

    HubSpot Info:
    • Company Domain: [Official domain, e.g. aircommunities.com]
    • Account Executive: [Look up in HubSpot manually]

    Property Details & Context:
    • Rebrand Status: [Primary and secondary community branding]
    • Value-Add / Renovations: [Property condition, amenities, unit count]
    • Transaction Context: [Acquisition details, price, sale date, or management transition]

    Sources & Evidence:
    • [Source Title/Press Release]: [URL]
    • [Source Title/Press Release]: [URL]
    """
    
    # Configure Google Search Grounding
    config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())]
    )

    try:
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt,
            config=config,
        )
        if response and response.text:
            return response.text
    except Exception as e:
        st.warning(f"Grounded search notice: {e}")

    # Fallback attempt if grounding tool needs standard call
    try:
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt,
        )
        return response.text
    except Exception as final_e:
        return f"Error generating research note: {str(final_e)}"

# -----------------------------------------
# 3. STREAMLIT USER INTERFACE
# -----------------------------------------
st.write("Enter an Opportunity Name from your Google Sheet to run AI research and generate a ready-to-paste note.")

opportunity_input = st.text_input("Opportunity Name (e.g., Lumiere Chandler / Transfer)")

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
                
                # Build Address String cleanly
                addr_parts = [p for p in [street, city, state, zip_code] if p]
                full_address = ", ".join(addr_parts) if addr_parts else "Chandler, AZ"
                
                # Generate research note
                final_note = generate_research_note(prop_name, full_address, prev_owner, prev_sop)
                
                st.success("Research Complete! Click the copy button in the top right of the box below.")
                st.code(final_note, language="text")
                
            else:
                st.error("Opportunity Name not found in your Google Sheet. Please check the spelling.")
    else:
        st.warning("Please enter an Opportunity Name first.")

import streamlit as st
import pandas as pd
import time
from google import genai
from google.genai import types

# -----------------------------------------
# 1. SETUP & CONFIGURATION
# -----------------------------------------
st.set_page_config(page_title="Property Transition Researcher", page_icon="🏢")
st.title("🏢 Property Transition AI Researcher")

# Retrieve Gemini API Key safely from Streamlit Secrets
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

if not GEMINI_API_KEY:
    st.error("Please add your GEMINI_API_KEY in the Streamlit Secrets settings.")
    st.stop()

# Initialize Google GenAI client
client = genai.Client(api_key=GEMINI_API_KEY)

# -----------------------------------------
# 2. HELPER FUNCTIONS
# -----------------------------------------
def get_property_data_from_sheet(search_term):
    """
    Reads Google Sheet CSV export link, cleans column names, 
    and searches across all text columns for the entered property/opportunity name.
    """
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

def generate_research_note(prop_name, address, prev_manager):
    """
    Prompts Gemini 3.6 Flash with Search Grounding to research real-time CRE news/data
    and return a vertically formatted note with HQ state and source links.
    """
    prompt = f"""
    Act as a Senior Commercial Real Estate Research Analyst. 
    Perform a live, targeted web search for actual CRE transaction articles, press releases (Cushman & Wakefield, JLL, CBRE, REBusinessOnline, Connect CRE), and property listings for:
    - Opportunity / Property Name: {prop_name}
    - Location/Address: {address}
    - Previous Manager / SOP: {prev_manager}

    TARGET RESEARCH REQUIREMENTS:
    1. CURRENT OWNER & MANAGER: Find the real buyer (e.g., Southwood Realty Co.) and current property management.
    2. PREVIOUS OWNER & DEVELOPER: Find the seller/developer (e.g., Waypoint Residential) and previous management/license account (e.g., Greystar).
    3. NEW OWNER HQ STATE: Corporate HQ location of buyer (City, State).
    4. COMPANY DOMAIN: Domain name of current owner (e.g., southwoodrealty.com).
    5. TRANSACTION METRICS: Purchase Price (e.g., $87M), sale date, unit count (e.g., 462 units), occupancy rate at sale (e.g., 95%), and listing brokers (e.g., Cushman & Wakefield).
    6. BRANDING / REBRAND: Note primary branding (Mason Augusta) and secondary branding (The Mason).
    7. SOURCES / EVIDENCE: Include exact web link URLs found in your search.

    OUTPUT FORMAT REQUIREMENT:
    Return strictly in the following vertical layout with exact line breaks, headers, and bullet points:

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
    * [Article/Press Release Title]: [URL]
    * [Article/Press Release Title]: [URL]
    """
    
    # Configure Google Search Grounding Tool
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
        st.warning(f"Grounded search note: {e}")

    # Fallback call if tool-grounding call returns empty or times out
    try:
        fallback_response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt,
        )
        return fallback_response.text if fallback_response.text else "Unable to generate research response."
    except Exception as fallback_e:
        return f"Error executing research generation: {str(fallback_e)}"

# -----------------------------------------
# 3. STREAMLIT USER INTERFACE
# -----------------------------------------
st.write("Enter an Opportunity Name from your Google Sheet to run AI research and generate a ready-to-paste note.")

opportunity_input = st.text_input("Opportunity Name (e.g., Mason Augusta / SM Transfer)")

if st.button("Generate Research Note"):
    if opportunity_input:
        with st.spinner("🔍 Reading sheet and conducting grounded web research..."):
            
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
                
                # Run Gemini Research
                final_note = generate_research_note(prop_name, address, prev_manager)
                
                st.success("Research Complete! Click the copy button in the top right of the box below.")
                st.code(final_note, language="markdown")
                
            else:
                st.error("Opportunity Name not found in your Google Sheet. Please check the spelling.")
    else:
        st.warning("Please enter an Opportunity Name first.")

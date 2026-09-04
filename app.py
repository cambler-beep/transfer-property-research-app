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
    Prompts Gemini with Google Search Grounding to research real-time news/data
    and return a vertically formatted note with HQ state and source links.
    """
    prompt = f"""
    Act as a Commercial Real Estate Research Analyst. 
    Perform a live web search to verify actual commercial real estate transaction press releases, news, and records for:
    - Property Name: {prop_name}
    - Location/Address: {address}
    - Previous/Known Manager: {prev_manager}

    Search specifically for recent sales, acquisitions, seller names, buyer names, brokerage press releases (e.g., Cushman & Wakefield, JLL, CBRE), purchase prices, unit counts, and corporate headquarters location.

    Find and verify:
    1. The actual current/new owner and property manager (e.g. Southwood Realty Co.).
    2. The Headquartered State (HQ State) of the new owner.
    3. The domain name of the new company.
    4. Previous owner/developer (e.g. Waypoint Residential) and previous management.
    5. Rebrand details (e.g. Mason Augusta / The Mason).
    6. Transaction metrics: Purchase Price, Occupancy at sale, Unit count, Brokers involved.
    7. Article or press release URLs confirming these facts.

    CRITICAL INSTRUCTION:
    Rely strictly on live search results for commercial real estate transactions. Do not fabricate or confuse site history.

    OUTPUT FORMAT REQUIREMENT:
    Return your response strictly in the following vertical layout with exact line breaks, headers, and bullet points:

    ### 📋 Property Transition Research Note
    **Property:** {prop_name} ({address})

    **Research Summary** 
    [Insert 2-3 sentence overview of the acquisition, including transaction price, unit count, and buyer/seller]

    **Ownership & Management**
    * **Current Owner:** [Owner Name]
    * **Previous Owner:** [Previous Owner/Developer Name]
    * **New Owner HQ State:** [City, State of HQ]
    * **Current Manager:** [Current Manager Name]
    * **Previous Manager:** {prev_manager}

    **HubSpot Info**
    * **Company Domain:** [e.g., southwoodrealty.com]
    * **Account Executive:** [Look up in HubSpot manually]

    **Property Details & Context**
    * **Rebrand Status:** [Details on primary/secondary branding]
    * **Value-Add / Renovations:** [Details on property status/renovations]
    * **Transaction Context:** [Details on purchase price, brokerage, occupancy, multi-phase expansion]

    **Sources/Evidence:**
    * [Article/Press Release Title]: [URL]
    * [Article/Press Release Title]: [URL]
    """
    
    models_to_try = ['gemini-3.5-flash', 'gemini-3.1-flash-lite', 'gemini-2.5-flash']
    
    # Configure Google Search Grounding
    config = types.GenerateContentConfig(
        tools=[{"google_search": {}}]
    )

    for model_id in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=config,
            )
            return response.text
        except Exception as e:
            if "429" in str(e) or "ResourceExhausted" in str(e):
                time.sleep(3)
                continue
            else:
                continue

    # Fallback
    response = client.models.generate_content(
        model='gemini-3.5-flash',
        contents=prompt,
        config=config,
    )
    return response.text

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
                
                # Run Gemini Research with Search Grounding
                final_note = generate_research_note(prop_name, address, prev_manager)
                
                st.success("Research Complete! Click the copy button in the top right of the box below.")
                st.code(final_note, language="markdown")
                
            else:
                st.error("Opportunity Name not found in your Google Sheet. Please check the spelling.")
    else:
        st.warning("Please enter an Opportunity Name first.")

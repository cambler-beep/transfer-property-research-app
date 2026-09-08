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
    if not raw_name:
        return ""
    clean = str(raw_name).replace('\xa0', ' ').split('/')[0].split('(')[0]
    clean = re.sub(r'(?i)\b(transfer|sop|retention|deal|sm|ai)\b', '', clean)
    return clean.strip()

def get_property_data_from_sheet(search_term):
    sheet_url = "https://docs.google.com/spreadsheets/d/1SJQ7YWUVcSSBKCKMSQFlMInxBTOeiLoJal6g2EHwhUU/export?format=csv&gid=1440084512"
    try:
        df = pd.read_csv(sheet_url, skiprows=1, dtype=str).fillna("")
        target_clean = clean_search_term(search_term).lower()
        
        for _, row in df.iterrows():
            row_str = " ".join(row.values).lower()
            if target_clean in row_str:
                return row
    except Exception as e:
        st.error(f"Error reading Google Sheet: {e}")
    return None

def get_flexible_col(row, possible_headers):
    if row is None:
        return ''
    for key in row.keys():
        clean_key = re.sub(r'[\xa0\s]+', ' ', str(key)).strip().lower()
        for h in possible_headers:
            if clean_key == h.lower():
                val = row[key]
                if val and str(val).strip().lower() not in ['nan', 'none', '', '#n/a']:
                    return str(val).strip()
    return ''

def generate_research_note(prop_name, full_address, prev_owner, prev_sop):
    clean_name = clean_search_term(prop_name)
    
    prompt = f"""
    Act as a Senior Commercial Real Estate (CRE) & Senior Housing Research Analyst.
    
    YOUR CRITICAL TASK:
    Use your built-in Google Search tool to browse the live internet and find the most up-to-date property management, ownership, and transaction details for the following asset. Search for the property website, press releases, and commercial real estate news.
    
    PROPERTY TO RESEARCH:
    - Name: {clean_name}
    - Location / Address: {full_address}

    TARGET INSTRUCTIONS:
    1. CURRENT OWNER: Identify the buyer, purchasing entity (LLC), holding company, REIT, or parent entity.
    2. CURRENT MANAGER / OPERATOR: Identify active property manager or operating company (e.g. Pegasus Residential). Check the footer of the property's official website.
    3. PREVIOUS OWNER & MANAGER: Identify seller/developer and former property manager/SOP operator.
    4. HEADQUARTERS STATES: Identify New Owner HQ State and Current Manager HQ State (City, State).
    5. COMPANY DOMAIN: Identify official domain name of the buyer or property manager/operator.
    6. REBRAND STATUS: Identify any name changes or rebranding.
    7. OVERVIEW: Include physical specs, building style, unit/bed count.
    8. VALUE-ADD / RENOVATIONS: Only list specific capital improvement plans if found.
    9. TRANSACTION CONTEXT: Summarize purchase price, sale date, buyer, seller.

    Return strictly in the following vertical layout without raw markdown symbols:

    📋 Property Transition Research Note
    Property: {clean_name} ({full_address})

    Research Summary:
    [2-3 sentence overview]

    Ownership & Management:
    • Current Owner: [Owner Name]
    • Previous Owner: {prev_owner if prev_owner != 'Unknown' and not prev_owner.startswith('006') else '[Previous Owner]'}
    • New Owner HQ State: [City, State]
    • Current Manager: [Current Property Manager]
    • Current Manager HQ State: [City, State]
    • Previous Manager: {prev_sop if prev_sop != 'Unknown' and not prev_sop.startswith('006') else '[Previous Manager]'}

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

    models_to_try = ['gemini-2.0-flash', 'gemini-1.5-flash']
    error_logs = []

    for model_id in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[{"google_search": {}}]
                )
            )
            if response and response.text:
                return response.text
        except Exception as e:
            error_logs.append(f"Model {model_id} failed: {str(e)}")

    # If both models fail, return the EXACT error messages so we can debug
    return "CRITICAL_ERROR: " + " | ".join(error_logs)

# -----------------------------------------
# 3. STREAMLIT USER INTERFACE
# -----------------------------------------
st.write("Enter an Opportunity Name from your Google Sheet to run AI research and generate a ready-to-paste note.")

opportunity_input = st.text_input("Opportunity Name (e.g., Property Name / Transfer)")

if st.button("Generate Research Note"):
    if opportunity_input:
        with st.spinner("🔍 Reading sheet and conducting live Google Search via AI..."):
            
            row = get_property_data_from_sheet(opportunity_input)
            
            if row is not None:
                opp_name = get_flexible_col(row, ['Opportunity Name']) or opportunity_input
                prop_name = get_flexible_col(row, ['Property Name']) or opp_name
                street = get_flexible_col(row, ['Street', 'Street Address'])
                city = get_flexible_col(row, ['City'])
                state = get_flexible_col(row, ['State/Province', 'State'])
                zip_code = get_flexible_col(row, ['Zip/Postal Code', 'Zip Code', 'Zip'])
                
                prev_owner = get_flexible_col(row, ['Previous License Account', 'Previous Owner']) or 'Unknown'
                prev_sop = get_flexible_col(row, ['Previous SOP', 'Previous Manager']) or 'Unknown'
                
                addr_parts = [p for p in [street, city, state, zip_code] if p]
                if addr_parts:
                    full_address = ", ".join(addr_parts)
                elif city or state:
                    full_address = ", ".join([p for p in [city, state] if p])
                else:
                    full_address = "Address Not Specified"
                
                final_note = generate_research_note(prop_name, full_address, prev_owner, prev_sop)
                
                if "CRITICAL_ERROR:" in final_note:
                    # Print the exact, raw error to the screen
                    exact_error = final_note.replace("CRITICAL_ERROR: ", "")
                    st.error(f"⚠️ Gemini API Error Details:\n\n{exact_error}")
                else:
                    st.success("Research Complete! Click the copy button in the top right of the box below.")
                    st.code(final_note, language="text")
                
            else:
                st.error("Opportunity Name not found in your Google Sheet. Please check the spelling.")
    else:
        st.warning("Please enter an Opportunity Name first.")

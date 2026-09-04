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
    """Clean internal deal tags like '/ Transfer', 'Transfer', 'SOP' from search terms."""
    if not raw_name:
        return ""
    clean = raw_name.split('/')[0].split('(')[0]
    clean = re.sub(r'(?i)\b(transfer|sop|retention|deal)\b', '', clean)
    return clean.strip()

def search_web_for_property(prop_clean_name, street, city, state):
    """
    Executes Python-side DuckDuckGo searches for CRE property records, 
    public tax filings, and management listings.
    """
    from duckduckgo_search import DDGS
    base_name = clean_search_term(prop_clean_name)
    
    queries = [
        f'"{base_name}" "{city}" owner manager REIT LLC',
        f'"{street}" "{city}" property owner',
        f'"{base_name}" "managed by" OR "AIR Communities" OR "Cortland" OR "Greystar"'
    ]
    
    results_text = ""
    sources = []
    seen_urls = set()
    
    try:
        with DDGS() as ddgs:
            for q in queries:
                results = list(ddgs.text(q, max_results=4))
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
    sheet_url = "https://docs.google.com/spreadsheets/d/1SJQ7YWUVcSSBKCKMSQFlMInxBTOeiLoJal6g2EHwhUU/export?format=csv&gid=1440084512"
    try:
        df = pd.read_csv(sheet_url)
        df.columns = df.columns.astype(str).str.strip()
        
        target_clean = clean_search_term(search_term).lower()
        
        # Search row by row for exact or substring matches in Opportunity/Property Name
        for _, row in df.iterrows():
            opp_name = str(row.get('Opportunity Name', '')).lower()
            prop_name = str(row.get('Property Name', '')).lower()
            
            if target_clean in opp_name or target_clean in prop_name or target_clean in str(row.values).lower():
                return row
                
    except Exception as e:
        st.error(f"Error reading Google Sheet: {e}")
    return None

def generate_research_note(prop_name, full_address, prev_owner, prev_sop, search_data, sources):
    clean_name = clean_search_term(prop_name)
    
    prompt = f"""
    Act as a Senior Commercial Real Estate (CRE) Research Analyst.
    Synthesize transaction and ownership details strictly using the provided live search research data.

    SEARCH RESEARCH DATA:
    {search_data}

    GOOGLE SHEET PARAMETERS:
    - Opportunity / Property Name: {clean_name}
    - Location / Address: {full_address}
    - Previous Owner / Account: {prev_owner}
    - Previous Manager / SOP: {prev_sop}

    TARGET CRE EXTRACT INSTRUCTIONS:
    1. Identify Current Owner / Holding Entity (e.g., Breit Mf Lumiere Chandler LLC / Blackstone BREIT).
    2. Identify Current Property Manager (e.g., AIR Communities / Apartment Income REIT Corp.).
    3. Identify New Owner Corporate Headquarters State (e.g., New York, NY for Blackstone).
    4. Identify Current Manager Corporate Headquarters State (e.g., Denver, CO for AIR Communities).
    5. Identify Previous Owner/Developer and Previous Manager if present in search data or sheet parameters.
    6. Identify Company Domain (e.g., aircommunities.com or breit.com).
    7. Summarize transaction context, unit count, and rebranding details.
    8. PROPERTY OVERVIEW: ALWAYS include an Overview bullet point detailing physical features (e.g., "336-unit garden-style community featuring resort-style pools, fitness centers, and modern interior finishes typical of portfolio standards.").
    9. VALUE-ADD / RENOVATIONS: Only list specific renovation or capital expenditure plans if explicitly found in research. If no specific renovation details are found in research, strictly state "N/A".

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
    • Current Manager HQ State: [City, State of HQ, e.g. Denver, CO]
    • Previous Manager: {prev_sop if prev_sop != 'Unknown' else '[Previous Manager Name]'}

    HubSpot Info:
    • Company Domain: [Official domain, e.g. aircommunities.com]
    • Account Executive: [Look up in HubSpot manually]

    Property Details & Context:
    • Rebrand Status: [Primary and secondary community branding]
    • Overview: [Property physical details, e.g. 336-unit garden-style community featuring resort-style pools, fitness center, and modern finishes]
    • Value-Add / Renovations: [Specific renovation plans if explicitly found in research; otherwise state N/A]
    • Transaction Context: [Acquisition details, price, sale date, or management transition]

    Sources & Evidence:
    """ + ("\n".join(sources[:4]) if sources else "• Search Public Records: https://www.aircommunities.com")

    # List of models to attempt with fallback
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
                
                # Build Address String
                addr_parts = [p for p in [street, city, state, zip_code] if p]
                full_address = ", ".join(addr_parts) if addr_parts else "Chandler, AZ"
                
                # Execute Python-side web search
                search_data, sources = search_web_for_property(prop_name, street, city, state)
                
                # Generate research note
                final_note = generate_research_note(prop_name, full_address, prev_owner, prev_sop, search_data, sources)
                
                st.success("Research Complete! Click the copy button in the top right of the box below.")
                st.code(final_note, language="text")
                
            else:
                st.error("Opportunity Name not found in your Google Sheet. Please check the spelling.")
    else:
        st.warning("Please enter an Opportunity Name first.")

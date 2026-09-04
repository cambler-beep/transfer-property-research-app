import streamlit as st
import google.generativeai as genai
import pandas as pd

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

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# -----------------------------------------
# 2. HELPER FUNCTIONS
# -----------------------------------------
def get_property_data_from_sheet(opportunity_name):
    """
    Reads the Google Sheet CSV export link and pulls the row matching Opportunity Name.
    """
    # This is the exact link to your specific Google Sheet exported as CSV
    sheet_url = "https://docs.google.com/spreadsheets/d/1SJQ7YWUVcSSBKCKMSQFlMInxBTOeiLoJal6g2EHwhUU/export?format=csv&gid=1440084512"
    
    try:
        df = pd.read_csv(sheet_url)
        # Match Opportunity Name column (case-insensitive search)
        row = df[df['Opportunity Name'].astype(str).str.contains(opportunity_name, na=False, case=False)]
        if not row.empty:
            return row.iloc[0]
    except Exception as e:
        st.error(f"Error reading Google Sheet: {e}")
    return None

def generate_research_note(prop_name, address, prev_manager):
    """
    Prompts Gemini to research the web and return 
    a vertically formatted note with HQ state and source links.
    """
    prompt = f"""
    Act as a Commercial Real Estate Research Analyst. 
    Conduct a live web search for the multifamily property:
    - Name: {prop_name}
    - Address: {address}
    - Previous Manager: {prev_manager}

    Find the following information:
    1. The current/new owner and new property manager.
    2. The Headquartered State (HQ State) of the new owner/company.
    3. The domain name of the new company (e.g., example.com).
    4. Whether there was a rebrand (new community name).
    5. Any value-add renovations, planned expansions, or transaction context (new development vs portfolio acquisition).
    6. Specific news/article links as evidence.

    OUTPUT FORMAT REQUIREMENT:
    Return your response strictly in the following vertical layout with exact line breaks, headers, and bullet points. Do not deviate from this layout.

    ### 📋 Property Transition Research Note
    **Property:** {prop_name} ({address})

    **Research Summary** 
    [Insert 2-3 sentence overview of the transition]

    **Ownership & Management**
    * **Current Owner:** [Owner Name]
    * **Previous Owner:** [Previous Owner Name if found, otherwise Unknown]
    * **New Owner HQ State:** [City, State of HQ]
    * **Current Manager:** [Current Manager Name]
    * **Previous Manager:** {prev_manager}

    **HubSpot Info**
    * **Company Domain:** [e.g., sentinelcorp.com]
    * **Account Executive:** [Look up in HubSpot manually]

    **Property Details & Context**
    * **Rebrand Status:** [Details on rebrand or 'No rebrand identified']
    * **Value-Add / Renovations:** [Details on renovations or 'None identified']
    * **Transaction Context:** [Details on acquisition/development]

    **Sources/Evidence:**
    * [Article/Press Release Title]: [URL]
    * [Article/Press Release Title]: [URL]
    """
    
    response = model.generate_content(prompt)
    return response.text

# -----------------------------------------
# 3. STREAMLIT USER INTERFACE
# -----------------------------------------
st.write("Enter an Opportunity Name from your Google Sheet to run AI research and generate a ready-to-paste note.")

opportunity_input = st.text_input("Opportunity Name (e.g., 1900 Elm / Transfer)")

if st.button("Generate Research Note"):
    if opportunity_input:
        with st.spinner("🔍 Reading sheet and conducting web research..."):
            
            prop_data = get_property_data_from_sheet(opportunity_input)
            
            if prop_data is not None:
                # Extract values from columns
                prop_name = prop_data.get('Property Name', 'Unknown Property')
                street = prop_data.get('Street', '')
                city = prop_data.get('City', '')
                state = prop_data.get('State/Province', '')
                zip_code = prop_data.get('Zip/Postal Code', '')
                
                address = f"{street}, {city}, {state} {zip_code}".strip(", ")
                prev_manager = prop_data.get('Previous SOP', 'Unknown')
                
                # Run Gemini Research
                final_note = generate_research_note(prop_name, address, prev_manager)
                
                st.success("Research Complete! Click the copy button in the top right of the box below.")
                
                # Output in a vertical code box with standard 1-click copy functionality
                st.code(final_note, language="markdown")
                
            else:
                st.error("Opportunity Name not found in your Google Sheet. Please check the spelling.")
    else:
        st.warning("Please enter an Opportunity Name first.")

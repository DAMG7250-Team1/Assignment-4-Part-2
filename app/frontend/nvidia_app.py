import streamlit as st
import sys
import os

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Add project root to Python path
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import pages directly
from pages.home import home_page
from pages.upload import upload_page
from pages.query import query_page

# Set page config
st.set_page_config(
    page_title="NVIDIA Report Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main entry point for the Streamlit application."""
    # Add page navigation to sidebar
    with st.sidebar:
        st.title("Navigation")
        page = st.radio(
            "Select Page",
            ["Home", "Upload", "Query", "Analyze by Year"],
            help="Choose which page to view",
            key="page_selector"
        )
    
    # Display selected page
    if page == "Home":
        home_page()
    elif page == "Upload":
        upload_page()
    elif page == "Analyze by Year":
        analyze_by_year()
    else:  # Query
        query_page()

if __name__ == "__main__":
    main()
import streamlit as st
from app.frontend.pages.home import home_page
from app.frontend.pages.upload import upload_page
from app.frontend.pages.query import query_page

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
            ["Home", "Upload", "Query"],
            help="Choose which page to view",
            key="page_selector"
        )
    
    # Display selected page
    if page == "Home":
        home_page()  # Empty home page
    elif page == "Upload":
        upload_page()
    else:  # Query
        query_page()

if __name__ == "__main__":
    main() 
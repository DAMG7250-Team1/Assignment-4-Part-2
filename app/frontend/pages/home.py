import streamlit as st

def home_page():
    """Home page of the NVIDIA Report Analysis application."""
    st.title("NVIDIA Report Analysis")
    
    st.markdown("""
    ## About
    This tool helps you analyze NVIDIA's financial reports using AI technology.
    
    ### Available Features:
    
    #### 📊 Report Processing
    - Upload NVIDIA financial reports
    - Download reports directly from NVIDIA's website
    - Process reports using advanced AI
    
    #### 🔍 Analysis Tools
    - Extract key financial metrics
    - Compare quarterly results
    - Track performance trends
    
    #### 💡 Smart Querying
    - Ask questions about NVIDIA's performance
    - Get insights from multiple reports
    - Compare data across quarters
    """)
    
    # Add visual separation
    st.markdown("---")
    
    # Add example queries
    st.subheader("Example Questions:")
    st.markdown("""
    - What was NVIDIA's revenue in the latest quarter?
    - How has the gaming segment performed?
    - What are the key growth drivers mentioned?
    - Compare data center revenue across quarters
    """)

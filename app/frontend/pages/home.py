import streamlit as st
import os
import requests
import json

# FastAPI backend URL
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")

def check_api_health():
    """Check if the FastAPI backend is healthy."""
    try:
        response = requests.get(f"{FASTAPI_URL}/api/health", timeout=2)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except Exception as e:
        return False, {"error": str(e)}

def home_page():
    """Home page for the NVIDIA RAG Pipeline."""
    st.title("NVIDIA RAG Pipeline")
    
    # Create a stylish layout with columns
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        ## About This Project
        
        The NVIDIA RAG Pipeline is a comprehensive system for analyzing NVIDIA's financial reports using 
        **Retrieval-Augmented Generation (RAG)**. This pipeline allows you to:

        * **Download** quarterly financial reports directly from NVIDIA's investor relations website
        * **Process and analyze** report content using advanced NLP techniques
        * **Query** the reports using natural language to extract valuable insights
        
        ### What is RAG?
        
        Retrieval-Augmented Generation (RAG) combines the power of large language models with 
        retrieval mechanisms that access external data. This allows the system to:
        
        1. Ground responses in factual information from NVIDIA's reports
        2. Provide accurate, up-to-date answers based on official financial data
        3. Cite sources and evidence for its responses
        """)
        
        # System architecture diagram
        st.subheader("System Architecture")
        st.markdown("""
        ```
        ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
        │  Data Sources │     │  Processing   │     │  Retrieval    │
        │  - PDF Reports│──►  │  - Extraction │──►  │  - Embedding  │
        │  - NVIDIA IR  │     │  - Parsing    │     │  - Indexing   │
        └───────────────┘     └───────────────┘     └───────┬───────┘
                                                            │
                                                            ▼
        ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
        │   Frontend    │     │   API Layer   │     │  Generation   │
        │  - Streamlit  │◄──► │  - FastAPI    │◄──► │  - LLM        │
        │  - UI/UX      │     │  - Endpoints  │     │  - Synthesis  │
        └───────────────┘     └───────────────┘     └───────────────┘
        ```
        """)
    
    with col2:
        # System status card
        st.markdown("## System Status")
        api_status, api_details = check_api_health()
        
        if api_status:
            st.success("✅ FastAPI Backend: Online")
            
            # Try to get available endpoints
            if api_details and "available_endpoints" in api_details:
                st.write("Available Endpoints:")
                for endpoint in api_details["available_endpoints"]:
                    st.write(f"- {endpoint}")
        else:
            st.error("❌ FastAPI Backend: Offline")
            if api_details and "error" in api_details:
                st.write(f"Error: {api_details['error']}")
            st.info(f"Expected URL: {FASTAPI_URL}")
        
        # Quick navigation
        st.markdown("## Quick Actions")
        
        if st.button("📥 Download NVIDIA Reports", use_container_width=True):
            import importlib
            download_page_module = importlib.import_module("app.frontend.pages.download")
            download_page_module.download_page()
            return
        
        if st.button("📤 Upload Custom Reports", use_container_width=True):
            import importlib
            upload_page_module = importlib.import_module("app.frontend.pages.upload")
            upload_page_module.upload_page()
            return
            
        if st.button("📊 Query Financial Data", use_container_width=True, type="primary"):
            import importlib
            query_page_module = importlib.import_module("app.frontend.pages.query")
            query_page_module.query_page()
            return
    
    # Horizontal divider
    st.divider()
    
    # Technical details section
    st.markdown("## Technical Details")
    
    # Three columns for the three main components of the pipeline
    tech_col1, tech_col2, tech_col3 = st.columns(3)
    
    with tech_col1:
        st.markdown("### Data Ingestion")
        st.markdown("""
        * Web scraping for NVIDIA financial reports
        * PDF processing with multiple parser options:
            * Basic Parser (PyPDF2)
            * Mistral Parser (LLM-based)
            * Docling Parser (Advanced NLP)
        * S3 storage for documents and processed data
        """)
    
    with tech_col2:
        st.markdown("### RAG Pipeline")
        st.markdown("""
        * Multiple RAG implementations:
            * Naive RAG
            * Pinecone RAG (vector database)
            * Chroma RAG (local vector store)
        * Customizable chunking strategies:
            * Fixed-size chunks
            * Sliding window
            * Semantic chunking
        * Context extraction and relevance ranking
        """)
    
    with tech_col3:
        st.markdown("### API & Frontend")
        st.markdown("""
        * FastAPI backend with:
            * Health monitoring
            * Query processing
            * Report downloads
            * Asynchronous tasks
        * Streamlit frontend with:
            * Interactive UI
            * Real-time progress tracking
            * Document visualization
            * Query interface
        """)
    
    # Example usage
    st.markdown("## Example Queries")
    st.markdown("""
    Try asking questions like:
    
    * "What was NVIDIA's revenue in the most recent quarter?"
    * "How has NVIDIA's gross margin changed over the last year?"
    * "What are the key growth drivers mentioned in the latest earnings report?"
    * "What are NVIDIA's projections for data center revenue?"
    * "How much did NVIDIA spend on R&D in 2023 compared to 2022?"
    """)
    
    # Footer
    st.divider()
    st.write("NVIDIA RAG Pipeline - Powered by Streamlit and FastAPI")

if __name__ == "__main__":
    home_page()

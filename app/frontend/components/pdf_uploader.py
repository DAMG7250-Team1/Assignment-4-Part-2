import streamlit as st
from typing import List
from streamlit.runtime.uploaded_file_manager import UploadedFile

def pdf_uploader() -> List[UploadedFile]:
    """
    Streamlit component for uploading PDF files.
    
    Returns:
        List[UploadedFile]: List of uploaded PDF files
    """
    uploaded_files = st.file_uploader(
        "Upload PDF Files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Select one or more PDF files to upload and process"
    )
    
    if uploaded_files:
        st.info(f"Uploaded {len(uploaded_files)} files")
        
        # Display file details
        for file in uploaded_files:
            st.text(f"File: {file.name} ({file.size / 1024:.1f} KB)")
            
    return uploaded_files or []

import streamlit as st
from typing import List
from streamlit.runtime.uploaded_file_manager import UploadedFile

def pdf_uploader(label: str = "Upload PDF Files") -> List[UploadedFile]:
    """
    Streamlit component for uploading PDF files.
    
    Args:
        label (str): The label to display for the file uploader
        
    Returns:
        List[UploadedFile]: List of uploaded PDF files
    """
    uploaded_files = st.file_uploader(
        label,
        type=["pdf"],
        accept_multiple_files=True,
        help="Select one or more PDF files to upload and process"
    )
            
    return uploaded_files or []

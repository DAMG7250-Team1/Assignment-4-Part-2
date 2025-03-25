import streamlit as st
from enum import Enum

class ParserType(Enum):
    """Available PDF parser types."""
    BASIC = "basic"
    MISTRAL = "mistral"
    DOCLING = "docling"

def parser_selector(prefix: str = "") -> str:
    """
    Streamlit component for selecting a PDF parser.
    
    Args:
        prefix (str): Prefix for the key to make it unique across pages
    
    Returns:
        str: Selected parser type
    """
    parser_type = st.selectbox(
        "Select Parser Type",
        options=[parser.value for parser in ParserType],
        help="Choose which parser to use for processing the PDF reports",
        key=f"{prefix}_parser_type_selector"
    )
    
    # Display parser descriptions
    if parser_type == ParserType.BASIC.value:
        st.info("Basic Parser: Simple text extraction using PyPDF2. Fast but no OCR or layout preservation.")
    elif parser_type == ParserType.MISTRAL.value:
        st.info("Mistral Parser: Advanced OCR with layout preservation, table detection, and image extraction.")
    elif parser_type == ParserType.DOCLING.value:
        st.info("Docling Parser: Specialized in document structure analysis and table extraction.")
        
    return parser_type

import streamlit as st
from enum import Enum

class RAGMethod(Enum):
    """Available RAG methods."""
    NAIVE = "naive"
    PINECONE = "pinecone"
    CHROMA = "chroma"

def rag_selector(prefix: str = "") -> str:
    """
    Streamlit component for selecting a RAG method.
    
    Args:
        prefix (str): Prefix for the key to make it unique across pages
    
    Returns:
        str: Selected RAG method
    """
    rag_method = st.selectbox(
        "Select RAG Method",
        options=[method.value for method in RAGMethod],
        help="Choose which RAG method to use for processing queries",
        key=f"{prefix}_rag_method_selector"
    )
    
    # Display method descriptions
    if rag_method == RAGMethod.NAIVE.value:
        st.info("Naive RAG: Simple similarity-based retrieval without advanced context handling.")
    elif rag_method == RAGMethod.PINECONE.value:
        st.info("Pinecone RAG: Vector similarity search with Pinecone for efficient retrieval.")
    elif rag_method == RAGMethod.CHROMA.value:
        st.info("Chroma RAG: Local vector store with ChromaDB for quick testing and development.")
        
    return rag_method

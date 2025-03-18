import streamlit as st
from app.frontend.components.rag_selector import rag_selector
from src.rag.methods.naive_rag import NaiveRAG
from src.rag.methods.pinecone_rag import PineconeRAG
from src.rag.methods.chroma_rag import ChromaRAG

def get_rag_method(method_type: str):
    """Get the appropriate RAG method based on type."""
    if method_type == "naive":
        return NaiveRAG()
    elif method_type == "pinecone":
        return PineconeRAG()
    elif method_type == "chroma":
        return ChromaRAG()
    else:
        raise ValueError(f"Unknown RAG method: {method_type}")

def query_page():
    """Query page for analyzing processed reports."""
    st.title("Query NVIDIA Reports")
    
    # Select RAG method
    selected_rag = rag_selector("query_page")
    
    # Query input
    query = st.text_area(
        "Enter your query",
        placeholder="Example: What were NVIDIA's key financial metrics in the last quarter?",
        help="Enter your question about NVIDIA's reports"
    )
    
    if query and st.button("Submit Query"):
        with st.spinner("Processing query..."):
            try:
                # Get RAG method instance
                rag_method = get_rag_method(selected_rag)
                
                # Process query and get response
                response = rag_method.process_query(query)
                
                # Display response
                st.markdown("### Answer")
                st.write(response.answer)
                
                # Display sources
                st.markdown("### Sources")
                for source in response.sources:
                    with st.expander(f"Source: {source.document}"):
                        st.markdown(source.context)
                        st.caption(f"Relevance Score: {source.score:.2f}")
                
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
    
    # Display query history in sidebar
    with st.sidebar:
        st.markdown("### Recent Queries")
        # TODO: Implement query history storage and display

if __name__ == "__main__":
    query_page()

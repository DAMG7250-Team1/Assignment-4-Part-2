import streamlit as st
from enum import Enum

class ChunkingStrategy(Enum):
    """Available chunking strategies."""
    FIXED_SIZE = "fixed_size"
    SLIDING_WINDOW = "sliding_window"
    SEMANTIC = "semantic"

def chunking_selector(prefix: str = "") -> str:
    """
    Streamlit component for selecting a chunking strategy.
    
    Args:
        prefix (str): Prefix for the key to make it unique across pages
    
    Returns:
        str: Selected chunking strategy
    """
    st.subheader("Document Chunking Strategy")
    
    # Add a recommendation banner
    st.markdown("""
    ### Recommendations:
    - **Fixed Size** (Default): Best balance for most cases
    - **Sliding Window**: For critical analytics when every detail matters
    - **Semantic**: For complex documents with strong topical structure
    """)
    
    chunking_strategy = st.selectbox(
        "Select Chunking Strategy",
        options=[strategy.value for strategy in ChunkingStrategy],
        index=0,  # Set Fixed Size as default
        help="Choose which chunking strategy to use for document processing",
        key=f"{prefix}_chunking_strategy_selector"
    )
    
    # Display detailed strategy descriptions with performance insights
    if chunking_strategy == ChunkingStrategy.FIXED_SIZE.value:
        st.info("""
        **Fixed Size**: Splits text into chunks of a fixed size (e.g., 1000 characters).
        
        **Performance Profile**:
        - **Speed**: Good balance (moderate ingestion, fast queries)
        - **Accuracy**: High for most queries
        - **Storage**: Efficient (moderate number of chunks)
        - **Best For**: General-purpose use cases with balanced performance
        """)
        
    elif chunking_strategy == ChunkingStrategy.SLIDING_WINDOW.value:
        st.warning("""
        **Sliding Window**: Creates overlapping chunks of text for better context preservation.
        
        **Performance Profile**:
        - **Speed**: Slow ingestion, fastest query response
        - **Accuracy**: Highest (captures all context through overlap)
        - **Storage**: High (creates 3-4x more chunks than other methods)
        - **Best For**: When retrieval accuracy is critical and resources are abundant
        
        ⚠️ Note: This strategy creates significantly more chunks and requires more storage space.
        """)
        
    elif chunking_strategy == ChunkingStrategy.SEMANTIC.value:
        st.success("""
        **Semantic**: Creates chunks based on semantic boundaries like paragraphs and sections.
        
        **Performance Profile**:
        - **Speed**: Moderate (requires additional processing)
        - **Accuracy**: High with coherent, meaningful chunks
        - **Storage**: Moderate (more chunks than fixed-size but fewer than sliding window)
        - **Best For**: Documents with complex structure or natural topic boundaries
        """)
        
    # Options for the selected strategy
    with st.expander("Advanced Options", expanded=False):
        if chunking_strategy == ChunkingStrategy.FIXED_SIZE.value:
            chunk_size = st.slider("Chunk Size", min_value=100, max_value=2000, value=1000, step=100,
                                key=f"{prefix}_fixed_size_chunk_size")
            chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=200, step=50,
                                key=f"{prefix}_fixed_size_chunk_overlap")
            st.session_state[f"{prefix}_chunking_options"] = {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap
            }
            
        elif chunking_strategy == ChunkingStrategy.SLIDING_WINDOW.value:
            window_size = st.slider("Window Size", min_value=100, max_value=2000, value=1000, step=100,
                                key=f"{prefix}_sliding_window_window_size")
            slide_amount = st.slider("Slide Amount", min_value=50, max_value=500, value=200, step=50,
                                key=f"{prefix}_sliding_window_slide_amount")
            st.session_state[f"{prefix}_chunking_options"] = {
                "window_size": window_size,
                "slide_amount": slide_amount
            }
            
        elif chunking_strategy == ChunkingStrategy.SEMANTIC.value:
            max_chunk_size = st.slider("Maximum Chunk Size", min_value=500, max_value=2000, value=1500, step=100,
                                key=f"{prefix}_semantic_max_size")
            min_chunk_size = st.slider("Minimum Chunk Size", min_value=50, max_value=1000, value=500, step=50,
                                key=f"{prefix}_semantic_min_size")
            st.session_state[f"{prefix}_chunking_options"] = {
                "max_chunk_size": max_chunk_size,
                "min_chunk_size": min_chunk_size
            }
    
    # Add a comparison table based on our test results
    with st.expander("Performance Comparison", expanded=False):
        st.markdown("""
        | Strategy | Chunks Created | Query Time | Storage Impact | Ingestion Time |
        |----------|----------------|------------|----------------|----------------|
        | Fixed-Size | 242 | 0.88 sec | Baseline | ~38 sec |
        | Sliding Window | 969 | 0.68 sec | ~4x | ~3 min 16 sec |
        | Semantic | 329 | 0.73 sec | ~1.4x | ~52 sec |
        
        *Based on tests with NVIDIA quarterly report PDFs*
        """)
        
    return chunking_strategy

def get_chunking_options(prefix: str = "") -> dict:
    """
    Get the chunking options from the session state.
    
    Args:
        prefix (str): Prefix for the key to make it unique across pages
        
    Returns:
        dict: Chunking options
    """
    return st.session_state.get(f"{prefix}_chunking_options", {})
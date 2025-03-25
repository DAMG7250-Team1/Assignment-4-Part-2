import streamlit as st
from app.frontend.components.rag_selector import rag_selector
from app.frontend.components.parser_selector import parser_selector, ParserType
from src.rag.methods.naive_rag import NaiveRAG
from src.rag.methods.pinecone_rag import PineconeRAG
from src.rag.methods.chroma_rag import ChromaRAG
import os
import logging
import re
from datetime import datetime
import io
import requests
import json

# Add import for parsers
from src.parsers.basic_parser import BasicPDFParser
from src.parsers.docling_parser import DoclingPDFParser
from src.parsers.mistral_parser import MistralPDFParser

# Add import for S3FileManager
try:
    from src.data.storage.s3_handler import S3FileManager
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False

# FastAPI backend URL
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")

logger = logging.getLogger(__name__)

def check_api_health():
    """Check if the FastAPI backend is healthy."""
    try:
        response = requests.get(f"{FASTAPI_URL}/api/health", timeout=2)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"API health check failed: {str(e)}")
        return False

def get_available_years(s3_manager):
    """Get available years for NVIDIA reports in S3."""
    try:
        all_files = []
        
        # Check multiple directories
        for prefix in ["processed/", "reports/", "downloaded_reports/"]:
            files = s3_manager.list_files(prefix=prefix)
            all_files.extend(files)
        
        # Extract years from filenames
        years = set()
        for file_path in all_files:
            # Try to extract year
            match = re.search(r'(\d{4})', file_path)
            if match:
                years.add(match.group(1))
        
        # Sort years in descending order (newest first)
        return sorted(list(years), reverse=True)
    except Exception as e:
        logger.error(f"Error getting available years: {str(e)}")
        return []

def process_pdf_file(s3_manager, pdf_path, parser_type):
    """Process a PDF file from S3 and return the extracted text."""
    try:
        # Download PDF from S3
        pdf_bytes = s3_manager.download_file(pdf_path)
        if not pdf_bytes:
            return None, f"Could not download {pdf_path} from S3"
            
        # Create BytesIO object
        pdf_io = io.BytesIO(pdf_bytes)
        
        # Get the appropriate parser
        if parser_type == ParserType.BASIC.value:
            parser = BasicPDFParser()
        elif parser_type == ParserType.MISTRAL.value:
            parser = MistralPDFParser()
        elif parser_type == ParserType.DOCLING.value:
            parser = DoclingPDFParser()
        else:
            return None, f"Unknown parser type: {parser_type}"
        
        # Extract file base name for output path
        filename = pdf_path.split('/')[-1].replace('.pdf', '')
        base_path = f"processed/{filename}"
        
        # Process the PDF
        output_path, metadata = parser.extract_text_from_pdf(pdf_io, base_path, s3_manager)
        
        # Read the processed text
        text_content = s3_manager.read_text(output_path)
        
        return text_content, metadata
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
        return None, str(e)

def get_rag_method(method_type, s3_manager, year_filter=None):
    """Get the appropriate RAG method based on type and load documents."""
    # Get Gemini API key for the generator component
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    # Initialize RAG method
    if method_type == "naive":
        # Initialize NaiveRAG with auto-loading from S3
        rag = NaiveRAG(load_from_s3=True)
        
        # Set year filter if specified
        if year_filter:
            rag.year_filter = year_filter
    elif method_type == "pinecone":
        # Initialize PineconeRAG
        rag = PineconeRAG(api_key=gemini_api_key)
    elif method_type == "chroma":
        # Initialize ChromaRAG
        rag = ChromaRAG(api_key=gemini_api_key)
    else:
        raise ValueError(f"Unknown RAG method: {method_type}")
    
    return rag

def query_via_api(query, rag_method, year=None, chunking_strategy="fixed_size"):
    """Query using the FastAPI backend."""
    try:
        # Prepare quarters filter if year is specified
        quarters = []
        if year:
            # Create filter for specified year (all quarters)
            quarters = [f"{year}_Q1", f"{year}_Q2", f"{year}_Q3", f"{year}_Q4"]
        
        # Create request payload
        payload = {
            "query": query,
            "rag_method": rag_method,
            "quarters": quarters,
            "top_k": 5,
            "chunking_strategy": chunking_strategy
        }
        
        # Call API
        response = requests.post(
            f"{FASTAPI_URL}/api/query",
            json=payload,
            timeout=60  # Allow up to 60 seconds for RAG processing
        )
        
        # Parse response
        if response.status_code == 200:
            result = response.json()
            return {
                "answer": result["answer"],
                "sources": result["sources"],
                "processing_time": result["processing_time"]
            }
        else:
            error_msg = f"API error: {response.status_code} - {response.text}"
            logger.error(error_msg)
            return {"error": error_msg}
            
    except Exception as e:
        logger.error(f"Error querying API: {str(e)}")
        return {"error": f"Failed to query API: {str(e)}"}

def query_page():
    """Query page for analyzing processed reports."""
    st.title("Query NVIDIA Reports")
    
    # Check if API is available
    api_available = check_api_health()
    
    # Initialize S3 file manager
    s3_manager = None
    if S3_AVAILABLE:
        try:
            s3_manager = S3FileManager()
            st.success("Connected to S3")
            
            # Get all files first to check what's available
            all_files = []
            pdf_files = []
            text_files = []
            
            for prefix in ["processed/", "reports/", "downloaded_reports/", "processed_reports/"]:
                files = s3_manager.list_files(prefix=prefix)
                if files:
                    all_files.extend(files)
                    
                    # Separate PDFs and text files
                    pdf_files.extend([f for f in files if f.endswith('.pdf')])
                    text_files.extend([f for f in files if f.endswith('.txt') or f.endswith('.md') or f.endswith('.json')])
            
            # Get available years
            available_years = get_available_years(s3_manager)
            
            # Create three columns for selectors
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Select RAG method
                selected_rag = rag_selector("query_page")
            
            with col2:
                # Year selector - only show if we have years available
                if available_years:
                    selected_year = st.selectbox(
                        "Filter by Year",
                        options=["All Years"] + available_years,
                        help="Filter reports by year"
                    )
                else:
                    selected_year = "All Years"
                    st.warning("No NVIDIA reports found. Please upload or download reports first.")
            
            with col3:
                # Add API mode toggle if API is available
                if api_available:
                    use_api = st.checkbox("Use FastAPI Backend", value=True, 
                                         help="When enabled, queries will be processed by the FastAPI backend instead of direct implementation")
                    if use_api:
                        st.success("FastAPI Backend is available")
                    
                    # Chunking strategy selector (only for API mode)
                    if use_api:
                        chunking_options = {
                            "fixed_size": "Fixed Size (Default)",
                            "sliding_window": "Sliding Window",
                            "semantic": "Semantic"
                        }
                        selected_chunking = st.selectbox(
                            "Chunking Strategy",
                            options=list(chunking_options.keys()),
                            format_func=lambda x: chunking_options[x],
                            help="Strategy for dividing documents into chunks"
                        )
                else:
                    use_api = False
                    if api_available is False:
                        st.warning("FastAPI Backend is not available")
            
            # Display document count 
            year_filter = None if selected_year == "All Years" else selected_year
            filtered_text_files = text_files
            if year_filter:
                filtered_text_files = [f for f in text_files if year_filter in f]
                
            filtered_pdf_files = pdf_files
            if year_filter:
                filtered_pdf_files = [f for f in pdf_files if year_filter in f]
            
            # Show what we found
            if filtered_text_files:
                st.info(f"Found {len(filtered_text_files)} processed text documents{' for ' + selected_year if year_filter else ''}")
            elif filtered_pdf_files:
                st.warning(f"Found {len(filtered_pdf_files)} PDF files but no processed text documents. You need to process these PDFs first.")
                
                # Add option to process PDFs on the fly
                st.subheader("Process PDFs")
                st.write("You can process these PDFs now to extract text for querying:")
                
                # Parser selection
                selected_parser = parser_selector("query_page")
                
                if st.button("Process PDFs Now", type="primary"):
                    with st.spinner(f"Processing {len(filtered_pdf_files)} PDF files..."):
                        # Process each PDF file
                        processed_texts = []
                        progress_bar = st.progress(0)
                        
                        for i, pdf_path in enumerate(filtered_pdf_files):
                            st.text(f"Processing {pdf_path}...")
                            text, metadata = process_pdf_file(s3_manager, pdf_path, selected_parser)
                            
                            if text:
                                processed_texts.append((text, metadata))
                                st.success(f"Processed {pdf_path} successfully")
                            else:
                                st.error(f"Failed to process {pdf_path}: {metadata}")
                                
                            # Update progress
                            progress_bar.progress((i + 1) / len(filtered_pdf_files))
                        
                        # Show number of processed documents
                        if processed_texts:
                            st.success(f"Successfully processed {len(processed_texts)} PDF files.")
                            filtered_text_files = [f for f in s3_manager.list_files(prefix="processed/") 
                                                if f.endswith('.txt') or f.endswith('.md') or f.endswith('.json')]
                            st.info(f"Now found {len(filtered_text_files)} processed text documents.")
                        else:
                            st.error("Failed to process any PDF files.")
            else:
                st.error("No documents found. Please upload or download NVIDIA reports first.")
            
            # Query input
            query = st.text_area(
                "Enter your query",
                placeholder="Example: What were NVIDIA's key financial metrics in the latest quarter?",
                help="Enter your question about NVIDIA's reports"
            )
            
            if query and st.button("Submit Query", type="primary" if filtered_text_files else "secondary"):
                if not filtered_text_files:
                    st.error("No processed text documents available. Please process PDF files first.")
                    return
                    
                with st.spinner(f"Processing query{' for ' + selected_year if year_filter else ''}..."):
                    try:
                        # Decide whether to use API or direct implementation
                        if use_api and api_available:
                            # Use FastAPI backend
                            st.info("Using FastAPI backend for query processing")
                            chunking_strategy = selected_chunking if 'selected_chunking' in locals() else "fixed_size"
                            
                            # Query via API
                            start_time = datetime.now()
                            result = query_via_api(
                                query=query, 
                                rag_method=selected_rag,
                                year=year_filter,
                                chunking_strategy=chunking_strategy
                            )
                            query_time = (datetime.now() - start_time).total_seconds()
                            
                            # Handle response
                            if "error" in result:
                                st.error(result["error"])
                                # Fall back to direct implementation
                                st.warning("Falling back to direct implementation...")
                                use_api = False
                            else:
                                # Display API response
                                st.success(f"Query processed in {result.get('processing_time', query_time):.2f} seconds")
                                
                                # Display the answer with proper markdown formatting
                                st.markdown("## Answer")
                                
                                # Check if this contains a table
                                answer_text = result["answer"]
                                if "|" in answer_text and any(marker in answer_text for marker in ["Revenue", "Profit", "Income", "$"]):
                                    # This likely contains financial tables - apply special formatting
                                    st.markdown('<div class="financial-tables">', unsafe_allow_html=True)
                                    st.markdown(answer_text)
                                    st.markdown('</div>', unsafe_allow_html=True)
                                else:
                                    # Regular rendering
                                    st.markdown(answer_text)
                                
                                # Display sources directly below the answer
                                sources = result.get("sources", [])
                                if sources:
                                    st.markdown("## Sources")
                                    for i, source in enumerate(sources):
                                        # Create an expandable section for each source
                                        st.markdown(f"**Source {i+1}**: {source.get('document', 'Unknown')}")
                                        st.caption(f"Relevance score: {source.get('score', 0):.2f}")
                                        
                                        # Display context in a code block for better formatting
                                        st.markdown("**Context:**")
                                        st.code(source.get('context', ''), language=None)
                                        
                                        # Add a separator between sources
                                        if i < len(sources) - 1:
                                            st.divider()
                                else:
                                    st.info("No specific sources provided for this answer.")
                        
                        # If not using API or API failed, use direct implementation
                        if not use_api or not api_available:
                            # Get RAG method instance and load documents
                            rag_method = get_rag_method(selected_rag, s3_manager, year_filter)
                            
                            # Check if documents are loaded
                            if hasattr(rag_method, 'documents'):
                                if not rag_method.documents:
                                    st.error("No documents have been loaded. Please upload or download reports first.")
                                    return
                                else:
                                    st.success(f"Successfully loaded {len(rag_method.documents)} documents for querying.")
                                    
                                    # Display a sample of the documents for debugging
                                    with st.expander("Debug: Document Details"):
                                        st.write(f"Number of documents loaded: {len(rag_method.documents)}")
                                        if year_filter:
                                            st.write(f"Year filter: {year_filter}")
                                            
                                        # Show a sample of the loaded documents
                                        st.write("Sample document sources:")
                                        sample_count = min(5, len(rag_method.documents))
                                        for i, (doc_id, doc_info) in enumerate(list(rag_method.documents.items())[:sample_count]):
                                            metadata = doc_info.get("metadata", {})
                                            st.write(f"- {metadata.get('source', 'Unknown')} (Year: {metadata.get('year', 'Unknown')})")
                            
                            # Process query and get response
                            start_time = datetime.now()
                            
                            # Check if the RAG method accepts year_filter parameter directly
                            if selected_rag == "naive" and hasattr(rag_method, 'query') and 'year_filter' in rag_method.query.__code__.co_varnames:
                                response = rag_method.query(query, year_filter=year_filter)
                            else:
                                # Fallback to regular query
                                response = rag_method.query(query)
                            
                            query_time = (datetime.now() - start_time).total_seconds()
                            
                            # Display response
                            st.success(f"Query processed in {query_time:.2f} seconds")
                            
                            # Display answer with proper markdown formatting
                            st.markdown("## Answer")
                            
                            # Check if this contains a table
                            if "|" in response and any(marker in response for marker in ["Revenue", "Profit", "Income", "$"]):
                                # This likely contains financial tables - apply special formatting
                                st.markdown('<div class="financial-tables">', unsafe_allow_html=True)
                                st.markdown(response)
                                st.markdown('</div>', unsafe_allow_html=True)
                            else:
                                # Regular rendering
                                st.markdown(response)
                            
                            # Display sources if available
                            if hasattr(rag_method, 'get_citations') and callable(getattr(rag_method, 'get_citations')):
                                citations = rag_method.get_citations()
                                if citations:
                                    st.markdown("## Sources")
                                    for i, citation in enumerate(citations):
                                        # Split the citation to get document and context parts
                                        doc_parts = citation.split('\n', 1)
                                        document = doc_parts[0] if len(doc_parts) > 0 else "Unknown"
                                        context = doc_parts[1] if len(doc_parts) > 1 else citation
                                        
                                        # Display document name
                                        st.markdown(f"**Source {i+1}**: {document}")
                                        st.caption(f"Relevance score: {1.0 - (i * 0.1):.2f}")
                                        
                                        # Display context in a code block for better formatting
                                        st.markdown("**Context:**")
                                        st.code(context, language=None)
                                        
                                        # Add separator
                                        if i < len(citations) - 1:
                                            st.divider()
                        
                    except Exception as e:
                        st.error(f"Error processing query: {str(e)}")
                        logger.error(f"Error in query processing: {str(e)}", exc_info=True)
        
        except Exception as e:
            st.error(f"Error connecting to S3: {str(e)}")
    else:
        st.error("S3 connection not available. Please check your environment.")

if __name__ == "__main__":
    query_page()

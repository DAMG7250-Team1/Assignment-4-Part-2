import streamlit as st
import pandas as pd
import json
import base64
import io
import logging
import os
import sys

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Direct imports from components
from components.parser_selector import parser_selector, ParserType
from components.pdf_uploader import pdf_uploader
from components.rag_selector import rag_selector, RAGMethod
from components.chunking_selector import chunking_selector, get_chunking_options

# Imports from src
from src.data.ingestion.pdf_downloader import NvidiaReportDownloader
from src.data.storage.s3_handler import S3FileManager
from src.parsers.mistral_parser import MistralPDFParser
from src.parsers.basic_parser import BasicPDFParser
from src.parsers.docling_parser import DoclingPDFParser
from src.rag.methods.naive_rag import NaiveRAG
from src.rag.methods.pinecone_rag import PineconeRAG
from src.rag.methods.chroma_rag import ChromaRAG

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_parser(parser_type: str):
    """Get the appropriate parser instance based on selected type."""
    if parser_type == ParserType.BASIC.value:
        return BasicPDFParser()
    elif parser_type == ParserType.MISTRAL.value:
        return MistralPDFParser()
    elif parser_type == ParserType.DOCLING.value:
        return DoclingPDFParser()
    else:
        raise ValueError(f"Unknown parser type: {parser_type}")

def get_rag_instance(rag_method: str, chunking_strategy: str, chunking_options: dict):
    """Get the appropriate RAG instance based on selected type."""
    # Get Gemini API key for the generator
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    if rag_method == RAGMethod.NAIVE.value:
        return NaiveRAG()
    elif rag_method == RAGMethod.PINECONE.value:
        # Pinecone uses its own API key from environment variables
        return PineconeRAG(
            chunking_strategy=chunking_strategy, 
            api_key=gemini_api_key,  # This is for the Gemini generator, not for Pinecone itself
            **chunking_options
        )
    elif rag_method == RAGMethod.CHROMA.value:
        # ChromaDB doesn't need an API key, but we pass Gemini key for the generator
        return ChromaRAG(
            chunking_strategy=chunking_strategy, 
            api_key=gemini_api_key,  # This is for the Gemini generator, not for ChromaDB
            **chunking_options
        )
    else:
        raise ValueError(f"Unknown RAG method: {rag_method}")

def display_file_content(output_path, s3_obj, metadata, parser_type):
    """Display processed file content based on parser type."""
    try:
        if not output_path or not s3_obj:
            st.warning("No content available to display.")
            return
        
        st.success(f"Successfully processed file!")
        
        # Display file information
        with st.expander("File Information", expanded=True):
            st.json(metadata)
        
        # Get file content
        content = s3_obj.read_text(output_path)
        if not content:
            st.warning("Could not retrieve file content.")
            return
        
        with st.expander("Processed Content", expanded=True):
            # Check file extension to determine how to display
            is_markdown = output_path.endswith('.md')
            is_json = output_path.endswith('.json')
            
            # For Mistral parser
            if parser_type == ParserType.MISTRAL.value:
                if is_json:
                    try:
                        json_data = json.loads(content)
                        st.json(json_data)
                        
                        # Check if we also have markdown file from Mistral
                        if 'markdown_file' in metadata:
                            markdown_content = s3_obj.read_text(metadata['markdown_file'])
                            if markdown_content:
                                st.markdown("### Markdown Version")
                                st.markdown(markdown_content)
                    except json.JSONDecodeError:
                        st.text(content)
                elif is_markdown:
                    st.markdown(content)
                else:
                    st.text(content)
            
            # For Docling parser (already generates markdown)
            elif parser_type == ParserType.DOCLING.value:
                if is_markdown:
                    st.markdown(content)
                else:
                    st.text(content)
            
            # For Basic parser
            elif parser_type == ParserType.BASIC.value:
                if is_markdown:
                    # If it's the markdown file, render it
                    st.markdown(content)
                else:
                    # For the text file, display as text
                    st.text(content)
                    
                    # Check if we also have markdown file from Basic parser
                    if 'markdown_file' in metadata:
                        markdown_content = s3_obj.read_text(metadata['markdown_file'])
                        if markdown_content:
                            st.markdown("### Markdown Version")
                            st.markdown(markdown_content)
            
            # Fallback for any other parser types
            else:
                if is_markdown:
                    st.markdown(content)
                elif is_json:
                    try:
                        st.json(json.loads(content))
                    except json.JSONDecodeError:
                        st.text(content)
                else:
                    st.text(content)
        
        # Display images if available (for any parser)
        if metadata and 'image_count' in metadata and metadata['image_count'] > 0:
            # Check if markdown content already has an image gallery section
            has_gallery_section = "## Image Gallery" in content if is_markdown else False
            
            # Only show the separate Images expander if there's no gallery section already
            if not has_gallery_section:
                with st.expander("Images", expanded=True):
                    st.write(f"Found {metadata['image_count']} images in document")
                    # We'll rely on the markdown rendering for images, as they're already included
                    st.info("Images are included in the markdown content above.")
    
    except Exception as e:
        st.error(f"Error displaying content: {str(e)}")
        logging.error(f"Error in display_file_content: {str(e)}", exc_info=True)

def add_to_rag_system(text, rag_instance, metadata=None):
    """Add text to the selected RAG system."""
    try:
        if not text:
            return False, "No text provided"
        
        if isinstance(rag_instance, NaiveRAG):
            doc_id = rag_instance.add_document(text, metadata)
            return True, f"Document added to Naive RAG system with ID: {doc_id[:8]}..."
        elif isinstance(rag_instance, (PineconeRAG, ChromaRAG)):
            if hasattr(rag_instance, 'add_document'):
                chunk_ids = rag_instance.add_document(text, metadata)
                return True, f"Document added to {rag_instance.__class__.__name__} with {len(chunk_ids)} chunks"
            else:
                # Fallback to pipeline process_document
                chunk_ids = rag_instance.pipeline.processor.process_text(text, metadata)
                return True, f"Document added to {rag_instance.__class__.__name__} with {len(chunk_ids)} chunks"
        else:
            return False, f"Unsupported RAG instance type: {type(rag_instance)}"
    
    except Exception as e:
        logger.error(f"Error adding to RAG system: {str(e)}")
        return False, f"Error: {str(e)}"

def upload_page():
    """Streamlit page for uploading and processing NVIDIA reports."""
    st.title("Upload & Process NVIDIA Reports")
    
    # Initialize S3 file manager
    try:
        s3_manager = S3FileManager()
        st.success("Successfully connected to S3")
    except Exception as e:
        st.error(f"Error connecting to S3: {str(e)}")
        return
    
    # Create tabs for different upload methods
    tab1, tab2 = st.tabs(["Upload PDF Files", "Download from NVIDIA"])
    
    with tab1:
        st.header("Upload PDF Files")
        uploaded_files = pdf_uploader("Upload one or more PDF files")
        
        if uploaded_files:
            st.write(f"Uploaded {len(uploaded_files)} file(s)")
            
            # Parser, RAG, and Chunking selection
            col1, col2, col3 = st.columns(3)
            with col1:
                selected_parser = parser_selector("upload_tab")
            with col2:
                selected_rag = rag_selector("upload_tab")
            with col3:
                selected_chunking = chunking_selector("upload_tab")
            
            # Get chunking options
            chunking_options = get_chunking_options("upload_tab")
            
            # Process button
            if st.button("Process Files"):
                with st.spinner("Processing files..."):
                    try:
                        # Get appropriate parser
                        parser = get_parser(selected_parser)
                        
                        # Initialize RAG system
                        rag_instance = get_rag_instance(selected_rag, selected_chunking, chunking_options)
                        
                        # Process each file
                        for uploaded_file in uploaded_files:
                            st.subheader(f"Processing: {uploaded_file.name}")
                            bytes_data = uploaded_file.getvalue()
                            
                            # Create file-specific base path
                            filename = uploaded_file.name.replace(".pdf", "").replace(" ", "_")
                            base_path = f"processed/{filename}"
                            
                            # Create BytesIO object
                            bytes_io = io.BytesIO(bytes_data)
                            
                            # Extract text and other content
                            try:
                                output_path, metadata = parser.extract_text_from_pdf(bytes_io, base_path, s3_manager)
                                st.success(f"Successfully processed {uploaded_file.name}")
                                
                                # Display processed content
                                st.markdown("### Processed Content")
                                display_file_content(output_path, s3_manager, metadata, selected_parser)
                                
                                # Add to RAG system
                                text_content = s3_manager.read_text(output_path)
                                if text_content:
                                    # Add extracted metadata
                                    file_metadata = {
                                        "source": uploaded_file.name,
                                        "parser": selected_parser,
                                        **metadata
                                    }
                                    
                                    success, message = add_to_rag_system(text_content, rag_instance, file_metadata)
                                    if success:
                                        st.success(f"Added to RAG system: {message}")
                                    else:
                                        st.warning(f"Failed to add to RAG system: {message}")
                                
                            except Exception as e:
                                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                                logging.error(f"Error in file processing: {str(e)}", exc_info=True)
                    
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        logging.error(f"Error in upload processing: {str(e)}", exc_info=True)
    
    with tab2:
        st.header("Download NVIDIA Reports")
        
        # Download options
        st.markdown("""
        Download and process the latest NVIDIA financial reports directly from their website.
        This will download annual reports, quarterly reports, and other financial documents.
        """)
        
        # Year selector
        available_years = [str(year) for year in range(2024, 2018, -1)]  # Last 6 years
        selected_year = st.selectbox(
            "Select Year to Analyze",
            options=["All Years"] + available_years,
            help="Choose which year of NVIDIA reports to analyze, or select 'All Years' for all available reports"
        )
        
        # Parser, RAG, and Chunking selection
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_parser = parser_selector("download_tab")
        with col2:
            selected_rag = rag_selector("download_tab")
        with col3:
            selected_chunking = chunking_selector("download_tab")
            
        # Get chunking options
        chunking_options = get_chunking_options("download_tab")
        
        # Download button
        if st.button("Download & Process Reports"):
            with st.spinner("Downloading and processing reports..."):
                try:
                    # Get appropriate parser
                    parser = get_parser(selected_parser)
                    
                    # Initialize RAG system
                    rag_instance = get_rag_instance(selected_rag, selected_chunking, chunking_options)
                    
                    # Create output directory
                    output_dir = "downloaded_reports"
                    
                    # Process downloaded reports with year filter
                    year_filter = None if selected_year == "All Years" else selected_year
                    processed_files = download_nvidia_reports(
                        output_dir, 
                        parser, 
                        s3_manager, 
                        year_filter=year_filter
                    )
                    
                    if processed_files:
                        st.success(f"Successfully downloaded and processed {len(processed_files)} reports")
                        
                        # Display each processed file
                        for s3_path, metadata in processed_files:
                            st.markdown(f"### {s3_path.split('/')[-1]}")
                            display_file_content(s3_path, s3_manager, metadata, selected_parser)
                            
                            # Add to RAG system
                            text_content = s3_manager.read_text(s3_path)
                            if text_content:
                                # Add extracted metadata
                                file_metadata = {
                                    "source": s3_path.split('/')[-1],
                                    "parser": selected_parser,
                                    "year": year_filter if year_filter else "All",
                                    **metadata
                                }
                                
                                success, message = add_to_rag_system(text_content, rag_instance, file_metadata)
                                if success:
                                    st.success(f"Added to RAG system: {message}")
                                else:
                                    st.warning(f"Failed to add to RAG system: {message}")
                    else:
                        st.error("No reports were downloaded")
                        
                except Exception as e:
                    st.error(f"Error downloading reports: {str(e)}")
                    logging.error(f"Error in report download: {str(e)}", exc_info=True)

# Define a simple download_nvidia_reports function for the Streamlit app
def download_nvidia_reports(output_dir, parser, s3_manager, year_filter=None):
    """
    Download NVIDIA reports and process them using the specified parser.
    
    Args:
        output_dir (str): Directory to save downloaded reports
        parser: Parser instance to process the reports
        s3_manager: S3FileManager instance for uploading content
        year_filter (str): Filter reports by year
        
    Returns:
        list: List of tuples containing (s3_path, metadata) for each processed report
    """
    try:
        # Initialize downloader
        downloader = NvidiaReportDownloader(s3_manager)
        
        # Download latest reports directly to S3
        logger.info(f"Downloading NVIDIA reports for {'all years' if not year_filter else year_filter}")
        
        # Check if we have a year filter
        if year_filter:
            # Download only reports for the specified year
            s3_paths = downloader.download_all_reports_s3_only(
                max_reports=10,  # Increase max reports when filtering by year
                filter_year=year_filter
            )
        else:
            # Download reports for all years
            s3_paths = downloader.download_all_reports_s3_only(max_reports=5)
        
        if not s3_paths:
            if year_filter:
                logger.warning(f"No reports were found for year {year_filter}")
            else:
                logger.warning("No reports were downloaded")
            return []
        
        logger.info(f"Downloaded {len(s3_paths)} reports")
        
        # Process each downloaded file from S3
        processed_files = []
        for s3_path in s3_paths:
            try:
                logger.info(f"Processing {s3_path}")
                
                # Download from S3 to memory for processing
                pdf_bytes = s3_manager.download_file(s3_path)
                file_bytes = io.BytesIO(pdf_bytes)
                
                # Generate a base path for S3 storage
                filename = s3_path.split('/')[-1].replace('.pdf', '').replace(' ', '_')
                base_path = f"{output_dir}/{filename}"
                
                # Extract year from filename for metadata
                file_year = "Unknown"
                if "NVIDIA_" in filename:
                    year_part = filename.split("NVIDIA_")[1].split("_")[0]
                    if year_part.isdigit() and len(year_part) == 4:
                        file_year = year_part
                
                # Process with the parser
                processed_s3_path, metadata = parser.extract_text_from_pdf(file_bytes, base_path, s3_manager)
                
                # Add year to metadata
                metadata["year"] = file_year
                
                processed_files.append((processed_s3_path, metadata))
                
                logger.info(f"Successfully processed {s3_path}")
            except Exception as e:
                logger.error(f"Error processing {s3_path}: {str(e)}")
        
        return processed_files
        
    except Exception as e:
        logger.error(f"Error in download_nvidia_reports: {str(e)}")
        return []

if __name__ == "__main__":
    upload_page()

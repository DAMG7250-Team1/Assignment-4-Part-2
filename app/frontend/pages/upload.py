import streamlit as st
import pandas as pd
import json
import base64
import io
import logging
from app.frontend.components.parser_selector import parser_selector, ParserType
from app.frontend.components.pdf_uploader import pdf_uploader
from app.frontend.components.rag_selector import rag_selector
from src.data.ingestion.pdf_downloader import NvidiaReportDownloader
from src.data.storage.s3_handler import S3FileManager
from src.parsers.mistral_parser import MistralPDFParser  # Uncommented to use the real parser
from src.parsers.basic_parser import BasicPDFParser
from src.parsers.docling_parser import DoclingPDFParser

# The DummyMistralPDFParser class is no longer needed since we're using the real parser
# Code commented out for reference
#
# class DummyMistralPDFParser:
#     """Temporary placeholder for the Mistral parser."""
#     def __init__(self):
#         self.logger = logging.getLogger(__name__)
#     
#     def extract_text_from_pdf(self, file_bytes, base_path, s3_manager):
#         # Return a simple message that Mistral parser is not available
#         error_msg = "Mistral parser is not available. Please use a different parser."
#         self.logger.warning(error_msg)
#         
#         # Create a simple JSON and upload it
#         metadata = {
#             "error": error_msg,
#             "timestamp": str(logging.Formatter().converter()),
#             "status": "failed"
#         }
#         
#         # Save metadata to S3
#         output_path = f"{base_path}/error.json"
#         s3_manager.write_text(output_path, json.dumps(metadata, indent=2))
#         
#         return output_path, metadata
#
# # Use the dummy parser instead
# MistralPDFParser = DummyMistralPDFParser

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
            
            # Parser and RAG selection
            col1, col2 = st.columns(2)
            with col1:
                selected_parser = parser_selector("upload_tab")
            with col2:
                selected_rag = rag_selector("upload_tab")
            
            # Process button
            if st.button("Process Files"):
                with st.spinner("Processing files..."):
                    try:
                        # Get appropriate parser
                        parser = get_parser(selected_parser)
                        
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
        
        # Parser and RAG selection
        col1, col2 = st.columns(2)
        with col1:
            selected_parser = parser_selector("download_tab")
        with col2:
            selected_rag = rag_selector("download_tab")
        
        # Download button
        if st.button("Download & Process Reports"):
            with st.spinner("Downloading and processing reports..."):
                try:
                    # Get appropriate parser
                    parser = get_parser(selected_parser)
                    
                    # Create output directory
                    output_dir = "downloaded_reports"
                    
                    # Process downloaded reports
                    processed_files = download_nvidia_reports(output_dir, parser, s3_manager)
                    
                    if processed_files:
                        st.success(f"Successfully downloaded and processed {len(processed_files)} reports")
                        
                        # Display each processed file
                        for s3_path, metadata in processed_files:
                            st.markdown(f"### {s3_path.split('/')[-1]}")
                            display_file_content(s3_path, s3_manager, metadata, selected_parser)
                    else:
                        st.error("No reports were downloaded")
                        
                except Exception as e:
                    st.error(f"Error downloading reports: {str(e)}")
                    logging.error(f"Error in report download: {str(e)}", exc_info=True)

# Define a simple download_nvidia_reports function for the Streamlit app
def download_nvidia_reports(output_dir, parser, s3_manager):
    """
    Download NVIDIA reports and process them using the specified parser.
    
    Args:
        output_dir (str): Directory to save downloaded reports
        parser: Parser instance to process the reports
        s3_manager: S3FileManager instance for uploading content
        
    Returns:
        list: List of tuples containing (s3_path, metadata) for each processed report
    """
    try:
        # Initialize downloader
        downloader = NvidiaReportDownloader(s3_manager)
        
        # Download latest reports directly to S3
        logger.info("Downloading NVIDIA reports directly to S3")
        s3_paths = downloader.download_all_reports_s3_only(max_reports=5)
        
        if not s3_paths:
            logger.warning("No reports were downloaded")
            return []
        
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
                
                # Process with the parser
                processed_s3_path, metadata = parser.extract_text_from_pdf(file_bytes, base_path, s3_manager)
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

import streamlit as st
import pandas as pd
import json
import base64
from app.frontend.components.parser_selector import parser_selector
from app.frontend.components.pdf_uploader import pdf_uploader
from app.frontend.components.rag_selector import rag_selector
from src.data.ingestion.pdf_downloader import NvidiaReportDownloader
from src.data.storage.s3_handler import S3FileManager
from src.parsers.mistral_parser import MistralPDFParser
from src.parsers.basic_parser import BasicPDFParser
from src.parsers.docling_parser import DoclingPDFParser

def get_parser(parser_type: str):
    """Get the appropriate parser instance based on type."""
    if parser_type == "mistral":
        return MistralPDFParser()
    elif parser_type == "basic":
        return BasicPDFParser()
    elif parser_type == "docling":
        return DoclingPDFParser()
    else:
        raise ValueError(f"Unknown parser type: {parser_type}")

def display_file_content(s3_manager: S3FileManager, file_path: str, parser_type: str):
    """Display the content of a processed file."""
    try:
        content = s3_manager.download_file(file_path).decode('utf-8')
        
        with st.expander(f"Processed Content ({parser_type})", expanded=True):
            if parser_type == "mistral":
                # Parse and display JSON content nicely
                try:
                    content_json = json.loads(content)
                    st.json(content_json)
                except:
                    st.text(content)
            elif parser_type == "docling":
                # First display the raw markdown content
                st.markdown(content)
                
                # Then display images separately
                import re
                image_urls = re.findall(r'!\[.*?\]\((.*?)\)', content)
                
                if image_urls:
                    st.subheader("Document Images")
                    cols = st.columns(2)  # Create two columns for images
                    for i, url in enumerate(image_urls):
                        with cols[i % 2]:  # Alternate between columns
                            try:
                                # Download image data directly
                                img_data = s3_manager.download_file(url.replace(f"https://{s3_manager.bucket_name}.s3.amazonaws.com/", ""))
                                st.image(img_data, caption=f"Image {i+1}", use_container_width=True)
                            except Exception as img_err:
                                st.warning(f"Could not display image {i+1}")
                                st.markdown(f"[Click to view Image {i+1}]({url})")
                
            else:  # basic parser
                # Display text content
                st.text(content)
                
                # Get and display associated images
                base_path = "/".join(file_path.split("/")[:-1])
                doc_filename = file_path.split("/")[-1].replace("basic_extracted_", "").replace(".txt", "")
                image_prefix = f"{base_path}/images/{doc_filename}"
                
                try:
                    image_files = s3_manager.list_files(prefix=image_prefix)
                    if image_files:
                        st.subheader("Document Images")
                        cols = st.columns(2)  # Create two columns for images
                        for i, img_file in enumerate(image_files):
                            with cols[i % 2]:  # Alternate between columns
                                try:
                                    img_data = s3_manager.download_file(img_file)
                                    st.image(img_data, caption=img_file.split('/')[-1], use_container_width=True)
                                except Exception as img_err:
                                    st.warning(f"Could not display image")
                                    image_url = f"https://{s3_manager.bucket_name}.s3.amazonaws.com/{img_file}"
                                    st.markdown(f"[Click to view image]({image_url})")
                except Exception as img_err:
                    st.warning(f"Could not list images: {str(img_err)}")
    except Exception as e:
        st.error(f"Error displaying file content: {str(e)}")

def upload_page():
    """NVIDIA Report Upload & Processing Page"""
    st.title("NVIDIA Report Upload & Processing")
    
    # Initialize S3 manager
    s3_manager = S3FileManager()
    
    # Tabs for different upload methods
    tab1, tab2 = st.tabs(["Upload PDF", "Download from NVIDIA"])
    
    with tab1:
        st.header("Upload PDF Files")
        uploaded_files = pdf_uploader()
        selected_parser = parser_selector("upload_tab")
        selected_rag = rag_selector("upload_tab")
        
        if uploaded_files and st.button("Process Uploaded Files"):
            parser = get_parser(selected_parser)
            
            with st.spinner("Processing uploaded files..."):
                for uploaded_file in uploaded_files:
                    try:
                        # Process file using selected parser
                        s3_path, metadata = parser.extract_text_from_pdf(
                            uploaded_file, 
                            "processed/nvidia_reports",
                            s3_manager
                        )
                        
                        st.success(f"Successfully processed {uploaded_file.name}")
                        
                        # Display processed content
                        st.markdown("### Processed Content")
                        display_file_content(s3_manager, s3_path, selected_parser)
                        
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    
    with tab2:
        st.header("Download from NVIDIA")
        selected_parser = parser_selector("download_tab")
        selected_rag = rag_selector("download_tab")
        
        col1, col2 = st.columns(2)
        with col1:
            num_years = st.slider(
                "Number of Years",
                min_value=1,
                max_value=10,
                value=5,
                help="Select how many years of reports to download"
            )
        
        with col2:
            download_type = st.radio(
                "Download Type",
                ["Latest Report Only", "All Reports"],
                help="Choose whether to download only the latest report or all reports for the selected years"
            )
        
        if st.button("Download and Process"):
            with st.spinner("Downloading and processing reports..."):
                downloader = NvidiaReportDownloader(s3_manager)
                parser = get_parser(selected_parser)
                
                try:
                    if download_type == "Latest Report Only":
                        pdf_path = downloader.get_latest_report()
                        if pdf_path:
                            with open(pdf_path, 'rb') as f:
                                s3_path, metadata = parser.extract_text_from_pdf(
                                    f, 
                                    "processed/nvidia_reports",
                                    s3_manager
                                )
                            st.success("Successfully processed latest report")
                            
                            # Display processed content
                            st.markdown("### Processed Content")
                            display_file_content(s3_manager, s3_path, selected_parser)
                            
                        else:
                            st.error("Failed to download latest report")
                    else:
                        pdf_paths = downloader.download_reports(num_years=num_years)
                        if pdf_paths:
                            processed_files = []
                            for path in pdf_paths:
                                with open(path, 'rb') as f:
                                    s3_path, metadata = parser.extract_text_from_pdf(
                                        f,
                                        "processed/nvidia_reports",
                                        s3_manager
                                    )
                                    processed_files.append((s3_path, metadata))
                            
                            st.success(f"Successfully processed {len(pdf_paths)} reports")
                            
                            # Display each processed file
                            for s3_path, metadata in processed_files:
                                st.markdown(f"### {s3_path.split('/')[-1]}")
                                display_file_content(s3_manager, s3_path, selected_parser)
                        else:
                            st.error("No reports were downloaded")
                            
                except Exception as e:
                    st.error(f"Error during download and processing: {str(e)}")

if __name__ == "__main__":
    upload_page()

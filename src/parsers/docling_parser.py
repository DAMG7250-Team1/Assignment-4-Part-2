from pathlib import Path
import io
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc import ImageRefMode, PictureItem
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)
from tempfile import NamedTemporaryFile
from docling.datamodel.pipeline_options import PdfPipelineOptions
from src.data.storage.s3_handler import S3FileManager
from datetime import datetime
import logging
import re
from selenium.webdriver.support.ui import Select


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Add console handler if not already present
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

class DoclingPDFParser:
    """
    A PDF parser that uses Docling for advanced text extraction with OCR capabilities.
    Handles text, tables, and images extraction from PDFs.
    """
    
    def __init__(self):
        self.metadata = {}
        
    def extract_text_from_pdf(self, pdf_stream: io.BytesIO, base_path: str, s3_obj: S3FileManager):
        """
        Extract text, tables, and images from a PDF file using Docling.
        
        Args:
            pdf_stream (io.BytesIO): The PDF file as a bytes stream
            base_path (str): Base path for storing extracted content
            s3_obj (S3FileManager): S3 storage manager instance
            
        Returns:
            tuple: (file path, metadata dictionary)
        """
        try:
            logger.info("Starting PDF extraction process")
            
            # Prepare pipeline options with OCR and table detection
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            pipeline_options.do_table_structure = True
            pipeline_options.images_scale = 2.0
            pipeline_options.generate_page_images = True
            pipeline_options.generate_picture_images = True

            logger.debug("Pipeline options configured")

            # Initialize the DocumentConverter with all features enabled
            doc_converter = DocumentConverter(
                allowed_formats=[InputFormat.PDF],
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options,
                    ),
                },
            )
            logger.debug("DocumentConverter initialized")
    
            # Process the PDF
            pdf_stream.seek(0)
            with NamedTemporaryFile(suffix=".pdf", delete=True) as temp_file:
                temp_file.write(pdf_stream.read())
                temp_file.flush()
                logger.debug(f"Temporary PDF file created at: {temp_file.name}")
                
                # Get PDF filename (if available) for more meaningful output filename
                pdf_name = "document"
                if hasattr(pdf_stream, 'name'):
                    pdf_name = Path(pdf_stream.name).stem
                
                # Convert PDF to markdown with embedded images
                timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                output_filename = f"{base_path}/{pdf_name}_docling_{timestamp}.md"
                logger.info(f"Will save output to: {output_filename}")
                
                # Convert and process the document
                logger.info("Starting document conversion")
                conv_result = doc_converter.convert(temp_file.name)
                logger.info("Document conversion completed")
                
                # Extract document metadata
                doc_metadata = self._extract_document_metadata(conv_result, pdf_name)
                
                final_md_content = self._process_conversion_result(conv_result, base_path, s3_obj)
                logger.debug("Conversion result processed")

                # Add document info section at the top with enhanced metadata
                doc_info = f"""# {doc_metadata.get('title', pdf_name)}

## Document Information
- **Processed On**: {timestamp}
- **Parser**: Docling PDF Parser
- **Document Type**: {doc_metadata.get('doc_type', 'Unknown')}
- **Pages**: {doc_metadata.get('page_count', 'Unknown')}
- **OCR**: Enabled
- **Table Detection**: Enabled

"""
                final_md_content = doc_info + final_md_content

                # Try to upload to S3 first, fall back to local saving if needed
                try:
                    # Upload the processed content to S3
                    logger.info("Uploading markdown content to S3")
                    if not s3_obj.upload_text(final_md_content, output_filename):
                        raise Exception("Failed to upload markdown content to S3")
                    logger.info(f"Successfully uploaded processed content to {output_filename}")
                except Exception as s3_error:
                    logger.warning(f"S3 upload failed: {s3_error}. Falling back to local file save.")
                    local_path = self._save_markdown_locally(final_md_content, base_path, pdf_name, timestamp)
                    output_filename = local_path
                
                # Update metadata with enhanced information
                image_count = self._count_images(conv_result)
                self.metadata = {
                    'timestamp': datetime.now().isoformat(),
                    'parser': 'docling_parser',
                    'title': doc_metadata.get('title', pdf_name),
                    'doc_type': doc_metadata.get('doc_type', 'Unknown'),
                    'page_count': doc_metadata.get('page_count', 0),
                    'ocr_enabled': True,
                    'table_structure_enabled': True,
                    'images_processed': True,
                    'num_images': image_count,
                    'image_count': image_count,  # Add image_count for frontend consistency
                    'markdown_file': output_filename  # Add markdown file path for reference
                }
                logger.debug(f"Updated metadata: {self.metadata}")
                
                return output_filename, self.metadata
                
        except Exception as e:
            logger.error(f"Error in Docling PDF extraction: {str(e)}", exc_info=True)
            raise
            
    def _extract_document_metadata(self, conv_result, default_title="document"):
        """
        Extract metadata from the document conversion result.
        
        Args:
            conv_result: Docling conversion result
            default_title (str): Default title if none found
            
        Returns:
            dict: Document metadata
        """
        metadata = {
            'title': default_title,
            'doc_type': 'PDF Document',
            'page_count': 0
        }
        
        try:
            # Extract title from first heading if available
            try:
                for element, level in conv_result.document.iterate_items():
                    if hasattr(element, 'text') and level == 1:  # Top level headings
                        potential_title = element.text.strip()
                        if potential_title:
                            metadata['title'] = potential_title
                            break
            except Exception as e:
                logger.warning(f"Failed to extract title: {e}")
                
            # Get page count
            try:
                if hasattr(conv_result.document, 'pages'):
                    metadata['page_count'] = len(conv_result.document.pages)
                elif hasattr(conv_result.input, 'page_count'):
                    metadata['page_count'] = conv_result.input.page_count
            except Exception as e:
                logger.warning(f"Failed to extract page count: {e}")
                
            # Try to determine document type
            if default_title.lower().find('10-k') >= 0:
                metadata['doc_type'] = 'Annual Report (10-K)'
            elif default_title.lower().find('10-q') >= 0:
                metadata['doc_type'] = 'Quarterly Report (10-Q)'
            elif default_title.lower().find('8-k') >= 0:
                metadata['doc_type'] = 'Current Report (8-K)'
            elif default_title.lower().find('nvidia') >= 0:
                metadata['doc_type'] = 'NVIDIA Document'
                
            return metadata
            
        except Exception as e:
            logger.warning(f"Error extracting document metadata: {e}")
            return metadata
            
    def _save_markdown_locally(self, content, base_path, pdf_name, timestamp):
        """
        Save markdown content to a local file when S3 upload fails.
        
        Args:
            content (str): Markdown content to save
            base_path (str): Base directory path
            pdf_name (str): Original PDF name
            timestamp (str): Timestamp for unique filename
            
        Returns:
            str: Path to saved file
        """
        try:
            import os
            local_dir = os.path.join(base_path, "local_markdown")
            os.makedirs(local_dir, exist_ok=True)
            
            local_path = os.path.join(local_dir, f"{pdf_name}_docling_{timestamp}.md")
            
            with open(local_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            logger.info(f"Saved markdown content locally to: {local_path}")
            return local_path
            
        except Exception as e:
            logger.error(f"Failed to save markdown locally: {e}")
            return f"{base_path}/{pdf_name}_docling_{timestamp}.md"  # Return original path

    def _process_conversion_result(self, conv_result, base_path: str, s3_obj: S3FileManager) -> str:
        """
        Process the conversion result and handle images.
        
        Args:
            conv_result: Docling conversion result
            base_path (str): Base path for storing content
            s3_obj (S3FileManager): S3 storage manager instance
            
        Returns:
            str: Processed markdown content with image links
        """
        try:
            logger.info("Starting conversion result processing")
            final_md_content = conv_result.document.export_to_markdown(image_mode=ImageRefMode.PLACEHOLDER)
            doc_filename = conv_result.input.file.stem

            # Count total images for logging
            total_images = self._count_images(conv_result)
            logger.info(f"Found {total_images} images in document")
            
            # If no images found, return the markdown content as is
            if total_images == 0:
                logger.info("No images found in document, skipping image processing")
                return final_md_content
            
            # Generate timestamp for unique image filenames
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            
            # Create image directory if needed (for both S3 and local)
            image_dir = f"{base_path}/images/docling_{timestamp}"
            local_image_dir = None
            
            # Create local directory if needed (for fallback)
            try:
                import os
                local_image_dir = os.path.join(base_path, "local_images", f"docling_{timestamp}")
                os.makedirs(local_image_dir, exist_ok=True)
                logger.debug(f"Created local image directory: {local_image_dir}")
            except Exception as dir_error:
                logger.warning(f"Failed to create local image directory: {dir_error}")
            
            logger.debug(f"Image directory will be: {image_dir}")
            
            # Process each image
            picture_counter = 0
            successful_uploads = 0
            image_urls = []  # Store successful image URLs for gallery
            
            for element, _level in conv_result.document.iterate_items():
                if isinstance(element, PictureItem):
                    picture_counter += 1
                    logger.debug(f"Processing image {picture_counter} of {total_images}")
                    
                    # Process and upload each image with timestamp for uniqueness
                    element_image_filename = f"{image_dir}/image_{picture_counter}.png"
                    logger.debug(f"Image will be saved as: {element_image_filename}")
                    
                    try:
                        # Get the image from the document
                        image = element.get_image(conv_result.document)
                        
                        # Log image dimensions for debugging
                        width, height = image.size
                        logger.debug(f"Original image dimensions: {width}x{height}")
                        
                        # Optimize image size if needed
                        img_size_kb = width * height * 4 / 1024  # Estimate size in KB
                        if img_size_kb > 1000:  # If larger than ~1MB
                            # Scale down the image to reduce size
                            scale_factor = min(1, 1000 / img_size_kb)
                            new_width = int(width * scale_factor)
                            new_height = int(height * scale_factor)
                            image = image.resize((new_width, new_height))
                            logger.debug(f"Resized image to {new_width}x{new_height}")
                            width, height = new_width, new_height
                        
                        # Try to upload to S3 first
                        s3_upload_success = False
                        element_image_link = None
                        local_image_path = None
                        
                        # Save to temporary file
                        with NamedTemporaryFile(suffix=".png", delete=True) as image_file:
                            # Save with optimal settings for web viewing
                            image.save(image_file, format="PNG", optimize=True, quality=85)
                            image_file.flush()
                            
                            # Try S3 upload first
                            try:
                                # Upload image to S3
                                with open(image_file.name, "rb") as fp:
                                    image_data = fp.read()
                                    logger.info(f"Uploading image {picture_counter} to S3 ({len(image_data)} bytes)")
                                    if s3_obj.upload_binary(image_data, element_image_filename):
                                        # Create S3 URL for the image
                                        element_image_link = f"https://{s3_obj.bucket_name}.s3.amazonaws.com/{element_image_filename}"
                                        logger.info(f"Image {picture_counter} uploaded: {element_image_link}")
                                        
                                        # Verify the image exists in S3
                                        if self._verify_image_exists(s3_obj, element_image_filename):
                                            logger.debug(f"Verified image exists at {element_image_filename}")
                                            s3_upload_success = True
                                        else:
                                            logger.warning(f"Could not verify image exists at {element_image_filename}")
                                            s3_upload_success = False
                                    else:
                                        logger.warning(f"S3 upload failed for image {picture_counter}")
                                        s3_upload_success = False
                            except Exception as s3_error:
                                logger.warning(f"Error uploading to S3: {s3_error}")
                                s3_upload_success = False
                            
                            # Fall back to local save if S3 upload failed
                            if not s3_upload_success and local_image_dir:
                                try:
                                    local_image_path = os.path.join(local_image_dir, f"image_{picture_counter}.png")
                                    with open(image_file.name, "rb") as src, open(local_image_path, "wb") as dst:
                                        dst.write(src.read())
                                    logger.info(f"Saved image {picture_counter} locally to {local_image_path}")
                                    element_image_link = f"file://{local_image_path}"
                                except Exception as local_error:
                                    logger.error(f"Error saving image locally: {local_error}")
                        
                        # If we have a valid image link (either S3 or local)
                        if element_image_link:
                            # Store URL for gallery along with dimensions
                            image_urls.append((
                                element_image_filename if s3_upload_success else local_image_path,
                                element_image_link,
                                (width, height)
                            ))
                            
                            # Replace placeholder with actual image link
                            final_md_content = final_md_content.replace(
                                "<!-- image -->", 
                                f"\n\n![Image {picture_counter}]({element_image_link})\n\n",
                                1
                            )
                            logger.debug(f"Added image link to markdown: {element_image_link}")
                            successful_uploads += 1
                        else:
                            # No valid image link, add a note about the failed image
                            final_md_content = final_md_content.replace(
                                "<!-- image -->",
                                f"\n\n[Failed to process Image {picture_counter}]\n\n",
                                1
                            )
                            logger.error(f"Failed to get valid image link for image {picture_counter}")
                            
                    except Exception as e:
                        logger.error(f"Error processing image {picture_counter}: {str(e)}")
                        # Add a note in the markdown about the failed image
                        final_md_content = final_md_content.replace(
                            "<!-- image -->",
                            f"\n\n[Failed to process Image {picture_counter}]\n\n",
                            1
                        )

            logger.info(f"Completed processing {successful_uploads} of {total_images} images successfully")
            
            # Check if all image placeholders were replaced
            remaining_placeholders = final_md_content.count("<!-- image -->")
            if remaining_placeholders > 0:
                logger.warning(f"Found {remaining_placeholders} remaining image placeholders that weren't replaced")
                # Replace any remaining placeholders with a generic message
                final_md_content = final_md_content.replace(
                    "<!-- image -->",
                    "\n\n[Image placeholder not replaced]\n\n"
                )
            
            # Add image gallery section if there were successful uploads
            if successful_uploads > 0:
                final_md_content = self._add_image_gallery(final_md_content, image_urls)
                
            return final_md_content
            
        except Exception as e:
            logger.error(f"Error processing conversion result: {str(e)}", exc_info=True)
            raise

    def _add_image_gallery(self, markdown_content, image_urls):
        """
        Add an image gallery section to the end of the markdown content.
        
        Args:
            markdown_content (str): The current markdown content
            image_urls (list): List of tuples with (image_path, image_url, dimensions)
            
        Returns:
            str: Updated markdown content with gallery
        """
        try:
            if not image_urls:
                return markdown_content
                
            # Add a gallery section separator
            gallery_content = "\n\n## Image Gallery\n\n"
            
            # Add summary information
            gallery_content += f"**Total Images**: {len(image_urls)}\n\n"
            
            # Calculate total size
            total_pixels = sum(width * height for _, _, (width, height) in image_urls)
            total_mb_approx = total_pixels * 4 / (1024 * 1024)  # Rough estimate of size in MB
            gallery_content += f"**Approximate Total Size**: {total_mb_approx:.2f} MB\n\n"
            
            # Add note about usage
            gallery_content += "Each image is displayed below in full quality. Click on the direct links to open images in a new tab.\n\n"
            
            # Create a summary table of all images
            gallery_content += "### Image Summary\n\n"
            gallery_content += "| # | Dimensions | Size (est.) | Direct Link |\n"
            gallery_content += "|---|------------|-------------|------------|\n"
            
            for i, (_, img_url, (width, height)) in enumerate(image_urls, 1):
                img_size_kb = (width * height * 4) / 1024  # Rough size estimate in KB
                size_display = f"{img_size_kb:.0f} KB" if img_size_kb < 1024 else f"{img_size_kb/1024:.2f} MB"
                gallery_content += f"| {i} | {width}x{height} | {size_display} | [Image {i}]({img_url}) |\n"
            
            gallery_content += "\n### Full Gallery\n\n"
            
            # Add all images to the gallery in a properly formatted way
            for i, (_, img_url, (width, height)) in enumerate(image_urls, 1):
                gallery_content += f"#### Image {i}\n\n"
                # Use proper markdown image syntax - important to have the full URL
                gallery_content += f"![Image {i}]({img_url})\n\n"
                # Add direct link to the image for convenient access
                gallery_content += f"**Direct link**: [Open in new tab]({img_url})  \n"
                gallery_content += f"**Dimensions**: {width}x{height}  \n"
                
                # Calculate approximate size
                img_size_kb = (width * height * 4) / 1024  # Rough size estimate
                size_display = f"{img_size_kb:.0f} KB" if img_size_kb < 1024 else f"{img_size_kb/1024:.2f} MB"
                gallery_content += f"**Approximate Size**: {size_display}\n\n"
                
                # Add horizontal line between images except after the last one
                if i < len(image_urls):
                    gallery_content += "---\n\n"
            
            return markdown_content + gallery_content
        except Exception as e:
            logger.error(f"Error adding image gallery: {str(e)}")
            return markdown_content  # Return original content if gallery addition fails

    def _count_images(self, conv_result) -> int:
        """Count the number of images in the document."""
        return sum(1 for element, _ in conv_result.document.iterate_items() if isinstance(element, PictureItem))

    def _verify_image_exists(self, s3_obj: S3FileManager, image_path: str) -> bool:
        """Verify if an image exists in S3."""
        return s3_obj.check_file_exists(image_path)

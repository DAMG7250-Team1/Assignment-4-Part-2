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
                
                # Convert PDF to markdown with embedded images
                timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                output_filename = f"{base_path}/docling_extracted_{timestamp}.md"
                logger.info(f"Will save output to: {output_filename}")
                
                # Convert and process the document
                logger.info("Starting document conversion")
                conv_result = doc_converter.convert(temp_file.name)
                logger.info("Document conversion completed")
                
                final_md_content = self._process_conversion_result(conv_result, base_path, s3_obj)
                logger.debug("Conversion result processed")

                # Upload the processed content to S3
                logger.info("Uploading markdown content to S3")
                if not s3_obj.upload_text(final_md_content, output_filename):
                    raise Exception("Failed to upload markdown content to S3")
                logger.info(f"Successfully uploaded processed content to {output_filename}")
                
                # Update metadata
                self.metadata = {
                    'timestamp': datetime.now().isoformat(),
                    'parser': 'docling_parser',
                    'ocr_enabled': True,
                    'table_structure_enabled': True,
                    'images_processed': True,
                    'num_images': self._count_images(conv_result)
                }
                logger.debug(f"Updated metadata: {self.metadata}")
                
                return output_filename, self.metadata
                
        except Exception as e:
            logger.error(f"Error in Docling PDF extraction: {str(e)}", exc_info=True)
            raise

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
            
            picture_counter = 0
            for element, _level in conv_result.document.iterate_items():
                if isinstance(element, PictureItem):
                    picture_counter += 1
                    logger.debug(f"Processing image {picture_counter}")
                    
                    # Process and upload each image
                    element_image_filename = f"{base_path}/images/{doc_filename}/image_{picture_counter}.png"
                    logger.debug(f"Image will be saved as: {element_image_filename}")
                    
                    try:
                        # Save image to temporary file
                        with NamedTemporaryFile(suffix=".png", delete=True) as image_file:
                            image = element.get_image(conv_result.document)
                            image.save(image_file, format="PNG", optimize=True, quality=85)
                            image_file.flush()
                            
                            # Upload image to S3
                            with open(image_file.name, "rb") as fp:
                                image_data = fp.read()
                                logger.info(f"Uploading image {picture_counter} to S3")
                                if not s3_obj.upload_binary(image_data, element_image_filename):
                                    raise Exception(f"Failed to upload image {picture_counter} to S3")
                            
                            # Create S3 URL for the image
                            element_image_link = f"https://{s3_obj.bucket_name}.s3.amazonaws.com/{element_image_filename}"
                            logger.info(f"Image {picture_counter} uploaded: {element_image_link}")
                            
                            # Replace placeholder with actual image link
                            final_md_content = final_md_content.replace(
                                "<!-- image -->", 
                                f"\n\n![Image {picture_counter}]({element_image_filename})\n\n",
                                1
                            )
                            logger.debug(f"Added image link to markdown: {element_image_filename}")
                    except Exception as e:
                        logger.error(f"Error processing image {picture_counter}: {str(e)}")
                        # Add a note in the markdown about the failed image
                        final_md_content = final_md_content.replace(
                            "<!-- image -->",
                            f"\n\n[Failed to process Image {picture_counter}]\n\n",
                            1
                        )

            logger.info("Completed processing conversion result")
            return final_md_content
            
        except Exception as e:
            logger.error(f"Error processing conversion result: {str(e)}", exc_info=True)
            raise

    def _count_images(self, conv_result) -> int:
        """Count the number of images in the document."""
        return sum(1 for element, _ in conv_result.document.iterate_items() if isinstance(element, PictureItem))

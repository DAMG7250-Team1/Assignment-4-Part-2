from typing import List, Dict, Any
import io
import PyPDF2
from datetime import datetime
import logging
from src.data.storage.s3_handler import S3FileManager
import fitz  # PyMuPDF for image extraction

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

class BasicPDFParser:
    """
    A basic PDF parser that extracts text using PyPDF2.
    This is the simplest parsing strategy that doesn't include OCR or advanced features.
    """
    
    def __init__(self):
        self.metadata = {}
    
    def extract_text_from_pdf(self, pdf_stream: io.BytesIO, base_path: str, s3_obj: S3FileManager):
        """
        Extract text and images from a PDF file using PyPDF2 and PyMuPDF.
        
        Args:
            pdf_stream (io.BytesIO): The PDF file as a bytes stream
            base_path (str): Base path for storing extracted content
            s3_obj (S3FileManager): S3 storage manager instance
            
        Returns:
            tuple: (extracted text file path, metadata dictionary)
        """
        try:
            logger.info("Starting PDF extraction with BasicPDFParser")
            
            # Create PDF reader object
            pdf_reader = PyPDF2.PdfReader(pdf_stream)
            logger.debug("PDF reader initialized")
            
            # Extract metadata
            self.metadata = {
                'num_pages': len(pdf_reader.pages),
                'metadata': pdf_reader.metadata,
                'timestamp': datetime.now().isoformat(),
                'parser': 'basic_pdf_parser'
            }
            logger.debug(f"Extracted metadata: {self.metadata}")
            
            # Extract text from each page
            extracted_text = []
            for page_num in range(len(pdf_reader.pages)):
                try:
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    extracted_text.append(f"--- Page {page_num + 1} ---\n{text}\n")
                    logger.debug(f"Successfully extracted text from page {page_num + 1}")
                except Exception as e:
                    logger.error(f"Error extracting text from page {page_num + 1}: {str(e)}")
                    extracted_text.append(f"--- Page {page_num + 1} ---\nError extracting text: {str(e)}\n")
            
            # Combine all extracted text
            final_text = "\n".join(extracted_text)
            
            # Generate timestamp for both text and images
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            output_filename = f"{base_path}/basic_extracted_{timestamp}.txt"
            logger.info(f"Will save output to: {output_filename}")
            
            # Upload text to S3
            try:
                logger.info("Uploading extracted text to S3")
                if not s3_obj.upload_text(final_text, output_filename):
                    raise Exception("Failed to upload text content to S3")
                logger.info(f"Successfully uploaded extracted text to {output_filename}")
            except Exception as e:
                logger.error(f"Error uploading text to S3: {str(e)}")
                raise
            
            # Extract and upload images using PyMuPDF
            try:
                logger.info("Starting image extraction with PyMuPDF")
                pdf_stream.seek(0)  # Reset stream position
                pdf_document = fitz.open(stream=pdf_stream.read(), filetype="pdf")
                
                image_count = 0
                for page_num in range(len(pdf_document)):
                    page = pdf_document[page_num]
                    image_list = page.get_images()
                    
                    for img_index, img in enumerate(image_list):
                        try:
                            xref = img[0]
                            base_image = pdf_document.extract_image(xref)
                            image_bytes = base_image["image"]
                            
                            # Generate image filename with timestamp
                            image_filename = f"{base_path}/images/basic_{timestamp}/image_{page_num + 1}_{img_index + 1}.png"
                            
                            # Upload image to S3
                            if s3_obj.upload_binary(image_bytes, image_filename):
                                image_count += 1
                                logger.debug(f"Successfully uploaded image {image_count} to {image_filename}")
                            else:
                                logger.warning(f"Failed to upload image {image_count} to {image_filename}")
                                
                        except Exception as img_err:
                            logger.error(f"Error processing image {img_index + 1} on page {page_num + 1}: {str(img_err)}")
                
                logger.info(f"Successfully extracted and uploaded {image_count} images")
                self.metadata['image_count'] = image_count
                
            except Exception as e:
                logger.error(f"Error in image extraction: {str(e)}")
                # Continue with text extraction even if image extraction fails
            
            return output_filename, self.metadata
            
        except Exception as e:
            logger.error(f"Error in PDF extraction: {str(e)}", exc_info=True)
            raise
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get the metadata from the last processed PDF.
        
        Returns:
            Dict[str, Any]: Dictionary containing metadata
        """
        return self.metadata
    
    def extract_text_by_page(self, pdf_stream: io.BytesIO) -> List[str]:
        """
        Extract text from PDF, returning a list of strings where each string is the text from one page.
        
        Args:
            pdf_stream (io.BytesIO): The PDF file as a bytes stream
            
        Returns:
            List[str]: List of extracted text strings, one per page
        """
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_stream)
            return [page.extract_text() for page in pdf_reader.pages]
        except Exception as e:
            logger.error(f"Error in page-by-page extraction: {str(e)}")
            raise

    def extract_text_by_section(self, pdf_stream: io.BytesIO, section_markers: List[str] = None) -> Dict[str, str]:
        """
        Extract text from PDF by sections, using section markers to split the content.
        
        Args:
            pdf_stream (io.BytesIO): The PDF file as a bytes stream
            section_markers (List[str]): List of section titles to look for
            
        Returns:
            Dict[str, str]: Dictionary mapping section names to their content
        """
        if section_markers is None:
            section_markers = ["Abstract", "Introduction", "Methods", "Results", "Discussion", "Conclusion"]
        
        try:
            full_text = " ".join(self.extract_text_by_page(pdf_stream))
            sections = {}
            
            # Split text into sections based on markers
            for i, marker in enumerate(section_markers):
                start = full_text.find(marker)
                if start != -1:
                    if i < len(section_markers) - 1:
                        end = full_text.find(section_markers[i + 1])
                        if end == -1:
                            sections[marker] = full_text[start:].strip()
                        else:
                            sections[marker] = full_text[start:end].strip()
                    else:
                        sections[marker] = full_text[start:].strip()
            
            return sections
            
        except Exception as e:
            logger.error(f"Error in section extraction: {str(e)}")
            raise
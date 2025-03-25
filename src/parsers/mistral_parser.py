import os
import io
import json
import logging
import time
from typing import Dict, Tuple, Any
from datetime import datetime
from dotenv import load_dotenv
from mistralai import Mistral
from src.data.storage.s3_handler import S3FileManager

class MistralPDFParser:
    """Parser that uses Mistral's OCR API for PDF processing."""
    
    def __init__(self):
        """Initialize the Mistral PDF Parser."""
        # Load environment variables
        load_dotenv()
        
        self.logger = logging.getLogger(__name__)
        self.api_key = os.getenv('MISTRAL_API_KEY')
        if not self.api_key:
            self.logger.error("MISTRAL_API_KEY not found in environment variables")
            raise ValueError("MISTRAL_API_KEY not found in environment variables. Please add it to your .env file")
        
        self.client = None
        self.ocr_model = "mistral-ocr-latest"

    def _initialize_client(self):
        """Initialize Mistral client with API key from environment."""
        try:
            if not self.client:
                self.logger.info("Initializing Mistral client...")
                self.client = Mistral(api_key=self.api_key)
                self.logger.info("Mistral client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Mistral client: {str(e)}")
            raise

    def extract_text_from_pdf(self, pdf_stream: io.BytesIO, base_path: str, s3_obj: S3FileManager) -> Tuple[str, Dict]:
        """
        Extract text from PDF using Mistral's OCR API with a simpler, direct approach.
        
        Args:
            pdf_stream: BytesIO stream of the PDF file
            base_path: Base path for storing extracted content
            s3_obj: S3FileManager instance for uploading content
            
        Returns:
            Tuple[str, Dict]: Path where content is stored and metadata dictionary
        """
        start_time = time.time()
        
        try:
            self.logger.info("Starting PDF extraction with Mistral OCR...")
            self._initialize_client()
            
            # Step 1: Upload the PDF file to Mistral's servers
            self.logger.info("Uploading PDF to Mistral...")
            uploaded_file = self.client.files.upload(
                file={
                    "file_name": "document.pdf",
                    "content": pdf_stream.read()
                },
                purpose="ocr"
            )
            
            # Step 2: Get a signed URL for the file
            self.logger.info("Getting signed URL...")
            signed_url = self.client.files.get_signed_url(file_id=uploaded_file.id)
            
            # Step 3: Process the PDF with OCR
            self.logger.info("Processing document with OCR...")
            ocr_response = self.client.ocr.process(
                model=self.ocr_model,
                document={
                    "type": "document_url",
                    "document_url": signed_url.url
                }
            )
            
            # Step 4: Format the OCR results into markdown
            self.logger.info("Formatting OCR results...")
            page_count = len(ocr_response.pages)
            table_count = 0
            
            markdown_content = f"""# Document OCR by Mistral
            
## Document Information
- **Processed On**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Parser**: Mistral OCR Parser
- **Page Count**: {page_count}
- **Processing Time**: {time.time() - start_time:.2f} seconds

## Content

"""
            
            # Add each page's content to the markdown
            for i, page in enumerate(ocr_response.pages):
                page_text = page.markdown
                table_count += page_text.count('|---')
                markdown_content += f"### Page {i+1}\n{page_text}\n\n"
            
            # Step 5: Save the results to S3
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            markdown_filename = f"mistral_ocr_{timestamp}.md"
            s3_path = f"{base_path}/mistral/{markdown_filename}"
            
            self.logger.info("Uploading results to S3...")
            s3_obj.upload_text(markdown_content, s3_path)
            
            # Step 6: Prepare and return metadata
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "parser": "mistral_parser",
                "processing_time": f"{time.time() - start_time:.2f}s",
                "table_count": table_count,
                "page_count": page_count,
                "markdown_file": s3_path
            }
            
            self.logger.info(f"PDF processing completed in {time.time() - start_time:.2f} seconds")
            return s3_path, metadata
            
        except Exception as e:
            self.logger.error(f"Error in Mistral PDF processing: {str(e)}")
            raise
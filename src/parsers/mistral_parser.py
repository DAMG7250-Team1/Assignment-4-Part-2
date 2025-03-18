import os
import io
import json
import logging
import time
from typing import Dict, Tuple, Any, Callable
from datetime import datetime
from dotenv import load_dotenv
from mistralai import Mistral
from mistralai import DocumentURLChunk, TextChunk
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
        self.timeout = 300  # 5 minutes timeout for operations
        self.max_retries = 5  # Maximum number of retries for rate-limited requests
        self.base_delay = 1  # Base delay in seconds for exponential backoff

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

    def _handle_rate_limit(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with rate limit handling and exponential backoff.
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Any: Result of the function call
        """
        retries = 0
        while retries <= self.max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if "429" in str(e) and retries < self.max_retries:
                    delay = self.base_delay * (2 ** retries)  # Exponential backoff
                    retries += 1
                    self.logger.warning(f"Rate limit hit. Retrying in {delay} seconds... (Attempt {retries}/{self.max_retries})")
                    time.sleep(delay)
                else:
                    raise

    def extract_text_from_pdf(self, pdf_stream: io.BytesIO, base_path: str, s3_obj: S3FileManager) -> Tuple[str, Dict]:
        """
        Extract text from PDF using Mistral's OCR API.
        
        Args:
            pdf_stream: BytesIO stream of the PDF file
            base_path: Base path for storing extracted content
            s3_obj: S3FileManager instance for uploading content
            
        Returns:
            Tuple[str, Dict]: Path where content is stored and metadata dictionary
        """
        start_total = time.time()
        try:
            self.logger.info("Starting PDF extraction with Mistral OCR...")
            self._initialize_client()
            
            # Upload PDF to Mistral with rate limit handling
            self.logger.info("Uploading PDF to Mistral...")
            start_time = time.time()
            try:
                uploaded_file = self._handle_rate_limit(
                    self.client.files.upload,
                    file={
                        "file_name": "document.pdf",
                        "content": pdf_stream.read(),
                    },
                    purpose="ocr"
                )
                self.logger.info(f"PDF uploaded successfully in {time.time() - start_time:.2f} seconds")
            except Exception as e:
                self.logger.error(f"Failed to upload PDF to Mistral: {str(e)}")
                raise
            
            # Get signed URL for processing
            self.logger.info("Getting signed URL...")
            try:
                signed_url = self._handle_rate_limit(
                    self.client.files.get_signed_url,
                    file_id=uploaded_file.id
                )
                self.logger.info("Signed URL obtained successfully")
            except Exception as e:
                self.logger.error(f"Failed to get signed URL: {str(e)}")
                raise
            
            # Process document with OCR
            self.logger.info("Processing document with OCR...")
            start_time = time.time()
            try:
                ocr_response = self._handle_rate_limit(
                    self.client.ocr.process,
                    model="mistral-ocr-latest",
                    document=DocumentURLChunk(document_url=signed_url.url),
                    include_image_base64=True
                )
                self.logger.info(f"OCR processing completed in {time.time() - start_time:.2f} seconds")
            except Exception as e:
                self.logger.error(f"Failed to process document with OCR: {str(e)}")
                raise
            
            # Check timeout
            if time.time() - start_total > self.timeout:
                raise TimeoutError("OCR processing exceeded timeout limit")
            
            # Extract text and process with chat model for structured output
            self.logger.info("Processing OCR results...")
            all_text = []
            table_count = 0
            
            for page in ocr_response.pages:
                page_text = page.markdown
                all_text.append(page_text)
                table_count += page_text.count('|---')
            
            combined_text = "\n\n".join(all_text)
            
            # Use chat model to structure the output with rate limit handling
            self.logger.info("Structuring output with chat model...")
            start_time = time.time()
            try:
                chat_response = self._handle_rate_limit(
                    self.client.chat.complete,
                    model="mistral-large-latest",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                TextChunk(text=f"The OCR in markdown:\n<BEGIN_OCR>\n{combined_text}\n<END_OCR>.\nConvert this into a structured JSON response with sections for text content, tables, and any numerical data found. The output should be strictly JSON with no extra commentary.")
                            ],
                        }
                    ],
                    response_format={"type": "json_object"},
                    temperature=0
                )
                self.logger.info(f"Chat model processing completed in {time.time() - start_time:.2f} seconds")
            except Exception as e:
                self.logger.error(f"Failed to structure output with chat model: {str(e)}")
                raise
            
            # Check timeout
            if time.time() - start_total > self.timeout:
                raise TimeoutError("Total processing exceeded timeout limit")
            
            # Parse structured response
            try:
                structured_content = json.loads(chat_response.choices[0].message.content)
            except Exception as e:
                self.logger.error(f"Failed to parse chat model response: {str(e)}")
                raise
            
            # Generate output path and filename
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            output_filename = f"mistral_ocr_{timestamp}.json"
            s3_path = f"{base_path}/mistral/{output_filename}"
            
            # Upload to S3
            self.logger.info("Uploading processed content to S3...")
            try:
                s3_obj.upload_text(json.dumps(structured_content, indent=2), s3_path)
                self.logger.info(f"Successfully uploaded to {s3_path}")
            except Exception as e:
                self.logger.error(f"Failed to upload to S3: {str(e)}")
                raise
            
            # Prepare metadata
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "parser": "mistral_parser",
                "processing_time": f"{time.time() - start_total:.2f}s",
                "table_count": table_count,
                "page_count": len(ocr_response.pages)
            }
            
            return s3_path, metadata
            
        except Exception as e:
            self.logger.error(f"Error in Mistral PDF processing: {str(e)}")
            raise
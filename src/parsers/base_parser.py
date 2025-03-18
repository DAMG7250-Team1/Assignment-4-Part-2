from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

class BaseParser(ABC):
    """Base class for all PDF parsers."""
    
    @abstractmethod
    def extract_text_from_pdf(self, pdf_stream, base_path: str, s3_obj) -> Tuple[str, Dict[str, Any]]:
        """
        Extract text and other content from PDF.
        
        Args:
            pdf_stream: BytesIO object containing the PDF
            base_path: Base path for storing extracted content
            s3_obj: S3FileManager instance for uploading content
        
        Returns:
            tuple: (output_path, metadata_dict)
        """
        pass 
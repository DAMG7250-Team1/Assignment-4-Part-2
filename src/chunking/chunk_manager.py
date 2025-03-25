"""
Chunk Manager that implements different text chunking strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import os
import logging

# Configure logging
logger = logging.getLogger(__name__)

class ChunkStrategy(ABC):
    """Abstract base class for text chunking strategies."""
    
    @abstractmethod
    def chunk_text(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Split text into chunks according to the strategy.
        
        Args:
            text: The text to be chunked
            **kwargs: Additional parameters specific to the chunking strategy
            
        Returns:
            List of chunk dictionaries with at least 'text' and 'metadata' keys
        """
        pass

class ChunkManager:
    """Manager class for text chunking strategies."""
    
    def __init__(self, strategy: Optional[ChunkStrategy] = None):
        """
        Initialize ChunkManager with a chunking strategy.
        
        Args:
            strategy: A chunking strategy (optional, can be set later)
        """
        self.strategy = strategy
    
    def set_strategy(self, strategy: ChunkStrategy) -> None:
        """
        Change the chunking strategy.
        
        Args:
            strategy: A chunking strategy to use
        """
        self.strategy = strategy
    
    def chunk_text(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Chunk text using the current strategy.
        
        Args:
            text: The text to be chunked
            **kwargs: Additional parameters to pass to the chunking strategy
            
        Returns:
            List of chunk dictionaries
            
        Raises:
            ValueError: If no strategy is set
        """
        if not self.strategy:
            raise ValueError("No chunking strategy set. Use set_strategy() first.")
        
        logger.info(f"Chunking text with strategy: {self.strategy.__class__.__name__}")
        return self.strategy.chunk_text(text, **kwargs)
    
    def chunk_document(self, document_path: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Read a document and chunk its content.
        
        Args:
            document_path: Path to the document file
            **kwargs: Additional parameters to pass to the chunking strategy
            
        Returns:
            List of chunk dictionaries
            
        Raises:
            FileNotFoundError: If the document file doesn't exist
            ValueError: If no strategy is set
        """
        if not os.path.exists(document_path):
            raise FileNotFoundError(f"Document not found: {document_path}")
        
        if not self.strategy:
            raise ValueError("No chunking strategy set. Use set_strategy() first.")
        
        # Determine file type and read content
        file_extension = os.path.splitext(document_path)[1].lower()
        
        # Read file based on extension
        try:
            with open(document_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            # Add document metadata to kwargs
            kwargs['metadata'] = kwargs.get('metadata', {})
            kwargs['metadata']['source'] = document_path
            kwargs['metadata']['file_type'] = file_extension
            
            logger.info(f"Chunking document: {document_path}")
            return self.strategy.chunk_text(content, **kwargs)
        
        except Exception as e:
            logger.error(f"Error chunking document {document_path}: {str(e)}")
            raise

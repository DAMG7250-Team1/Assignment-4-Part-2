"""
Fixed-size chunking strategy that splits documents into chunks of specified token or character length.
"""

import re
from typing import List, Dict, Any, Union, Optional
import logging

logger = logging.getLogger(__name__)

class FixedSizeChunker:
    """
    Chunker that splits text into fixed-size chunks.
    
    This chunker can operate in either character mode or token mode (approximate).
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        length_function: str = "character"
    ):
        """
        Initialize the fixed size chunker.
        
        Args:
            chunk_size: Size of each chunk in characters or tokens
            chunk_overlap: Amount of overlap between chunks in characters or tokens
            length_function: Method to measure text length - "character" or "token"
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if length_function not in ["character", "token"]:
            raise ValueError("length_function must be either 'character' or 'token'")
        
        self.length_function = length_function
        logger.info(f"Initialized FixedSizeChunker with chunk_size={chunk_size}, "
                   f"chunk_overlap={chunk_overlap}, length_function={length_function}")
    
    def _calculate_length(self, text: str) -> int:
        """Calculate the length of text based on the chosen length function."""
        if self.length_function == "character":
            return len(text)
        elif self.length_function == "token":
            # Simple token count approximation using whitespace
            return len(text.split())
    
    def _split_text(self, text: str) -> List[str]:
        """Split text into appropriate units for chunking."""
        if self.length_function == "character":
            # For character mode, we can just use the raw text
            return [text]
        elif self.length_function == "token":
            # For token mode, split by whitespace first
            return text.split()
    
    def _join_text(self, units: List[str]) -> str:
        """Join text units back into a string."""
        if self.length_function == "character":
            return "".join(units)
        elif self.length_function == "token":
            return " ".join(units)
    
    def create_chunks(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Split text into fixed-size chunks with specified overlap.
        
        Args:
            text: Text to be chunked
            metadata: Optional metadata to be included with each chunk
            
        Returns:
            List of dictionaries with 'text' and 'metadata' keys
        """
        if not text:
            logger.warning("Empty text provided to chunker")
            return []
        
        # Handle empty or None metadata
        if metadata is None:
            metadata = {}
        
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = []
        current_size = 0
        
        for paragraph in paragraphs:
            # Skip empty paragraphs
            if not paragraph.strip():
                continue
                
            paragraph_size = self._calculate_length(paragraph)
            
            # If a single paragraph is larger than the chunk size, we need to split it
            if paragraph_size > self.chunk_size:
                # If there's already content in the current chunk, add it to chunks
                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            **metadata,
                            "chunk_idx": len(chunks),
                            "chunk_size": self._calculate_length(chunk_text)
                        }
                    })
                    current_chunk = []
                    current_size = 0
                
                # Split the large paragraph
                units = self._split_text(paragraph)
                start_idx = 0
                
                while start_idx < len(units):
                    end_idx = start_idx
                    current_size = 0
                    
                    # Find how many units fit in a chunk
                    while end_idx < len(units) and current_size + self._calculate_length(units[end_idx]) <= self.chunk_size:
                        current_size += self._calculate_length(units[end_idx])
                        if self.length_function == "token":
                            # Add space for token mode
                            current_size += 1
                        end_idx += 1
                    
                    # Create the chunk
                    if start_idx < end_idx:
                        chunk_text = self._join_text(units[start_idx:end_idx])
                        chunks.append({
                            "text": chunk_text,
                            "metadata": {
                                **metadata,
                                "chunk_idx": len(chunks),
                                "chunk_size": self._calculate_length(chunk_text)
                            }
                        })
                    
                    # Move start index for next chunk, accounting for overlap
                    overlap_units = int(self.chunk_overlap / (1 if self.length_function == "character" else 2))
                    start_idx = max(start_idx + 1, end_idx - overlap_units)
            else:
                # If adding this paragraph would exceed the chunk size, start a new chunk
                if current_size + paragraph_size > self.chunk_size and current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            **metadata,
                            "chunk_idx": len(chunks),
                            "chunk_size": self._calculate_length(chunk_text)
                        }
                    })
                    
                    # Keep some paragraphs for overlap
                    overlap_needed = self.chunk_overlap
                    overlap_paragraphs = []
                    total_overlap_size = 0
                    
                    for p in reversed(current_chunk):
                        p_size = self._calculate_length(p)
                        if total_overlap_size + p_size <= overlap_needed:
                            overlap_paragraphs.insert(0, p)
                            total_overlap_size += p_size
                        else:
                            break
                    
                    current_chunk = overlap_paragraphs
                    current_size = total_overlap_size
                
                # Add the paragraph to the current chunk
                current_chunk.append(paragraph)
                current_size += paragraph_size
        
        # Add the last chunk if it has content
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_idx": len(chunks),
                    "chunk_size": self._calculate_length(chunk_text)
                }
            })
        
        logger.info(f"Created {len(chunks)} chunks from input text")
        return chunks

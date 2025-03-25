"""
Sliding window chunking strategy that provides more fine-grained control over document chunking.
"""

import re
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class SlidingWindowChunker:
    """
    Chunker that implements a sliding window approach for text chunking.
    
    This chunker advances through the text in small increments (the slide amount),
    creating chunks of specified size with each slide.
    """
    
    def __init__(
        self, 
        window_size: int = 1000,
        slide_amount: int = 200,
        length_function: str = "character"
    ):
        """
        Initialize the sliding window chunker.
        
        Args:
            window_size: Size of each chunk window in characters or tokens
            slide_amount: Amount to slide the window for each new chunk
            length_function: Method to measure text length - "character" or "token"
        """
        self.window_size = window_size
        self.slide_amount = slide_amount
        
        if length_function not in ["character", "token"]:
            raise ValueError("length_function must be either 'character' or 'token'")
        
        self.length_function = length_function
        logger.info(f"Initialized SlidingWindowChunker with window_size={window_size}, "
                    f"slide_amount={slide_amount}, length_function={length_function}")
    
    def _measure_text(self, text: str) -> int:
        """Measure text length according to configured length function."""
        if self.length_function == "character":
            return len(text)
        elif self.length_function == "token":
            return len(text.split())
    
    def _split_text(self, text: str) -> List[str]:
        """Split text into units for sliding window operation."""
        if self.length_function == "character":
            # For character mode, use individual characters
            return list(text)
        elif self.length_function == "token":
            # For token mode, split by whitespace
            return text.split()
    
    def _join_units(self, units: List[str]) -> str:
        """Join text units back into a unified string."""
        if self.length_function == "character":
            return "".join(units)
        elif self.length_function == "token":
            return " ".join(units)
    
    def create_chunks(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Split text into chunks using a sliding window approach.
        
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
        
        # Split text into units according to length function
        units = self._split_text(text)
        total_units = len(units)
        
        # If text is smaller than window size, return it as a single chunk
        if total_units <= self.window_size:
            return [{
                "text": text,
                "metadata": {
                    **metadata,
                    "chunk_idx": 0,
                    "chunk_size": self._measure_text(text)
                }
            }]
        
        chunks = []
        window_start = 0
        
        # Slide the window through the text
        while window_start < total_units:
            # Calculate window end, ensuring we don't exceed array bounds
            window_end = min(window_start + self.window_size, total_units)
            
            # Create chunk from current window
            window_units = units[window_start:window_end]
            chunk_text = self._join_units(window_units)
            
            # Skip empty chunks
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_idx": len(chunks),
                        "window_start": window_start,
                        "window_end": window_end - 1,  # Inclusive end index
                        "chunk_size": self._measure_text(chunk_text)
                    }
                })
            
            # Slide the window forward
            window_start += self.slide_amount
        
        logger.info(f"Created {len(chunks)} chunks using sliding window approach")
        return chunks
    
    def create_chunks_with_sentence_boundaries(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Split text into chunks using sliding window while respecting sentence boundaries.
        
        This method attempts to adjust chunk boundaries to coincide with the end of sentences
        to create more semantically coherent chunks.
        
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
        
        # Split text into sentences first
        # This regex attempts to match sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = self._measure_text(sentence)
            
            # If a single sentence is larger than the window size, use regular chunking
            if sentence_size > self.window_size:
                if current_chunk:
                    # Add the accumulated sentences as a chunk
                    chunk_text = " ".join(current_chunk)
                    chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            **metadata,
                            "chunk_idx": len(chunks),
                            "chunk_size": self._measure_text(chunk_text)
                        }
                    })
                    current_chunk = []
                    current_size = 0
                
                # Process oversized sentence with basic sliding window
                sentence_units = self._split_text(sentence)
                window_start = 0
                
                while window_start < len(sentence_units):
                    window_end = min(window_start + self.window_size, len(sentence_units))
                    window_text = self._join_units(sentence_units[window_start:window_end])
                    
                    chunks.append({
                        "text": window_text,
                        "metadata": {
                            **metadata,
                            "chunk_idx": len(chunks),
                            "chunk_size": self._measure_text(window_text),
                            "oversized_sentence": True
                        }
                    })
                    
                    window_start += self.slide_amount
            else:
                # If adding this sentence would exceed the window size
                if current_size + sentence_size > self.window_size and current_chunk:
                    # Add current chunk to chunks
                    chunk_text = " ".join(current_chunk)
                    chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            **metadata,
                            "chunk_idx": len(chunks),
                            "chunk_size": self._measure_text(chunk_text)
                        }
                    })
                    
                    # Determine sentences to keep for overlap
                    overlap_needed = self.slide_amount
                    overlap_sentences = []
                    total_overlap_size = 0
                    
                    # Build overlap from the end of the current chunk
                    for s in reversed(current_chunk):
                        s_size = self._measure_text(s)
                        if total_overlap_size + s_size <= overlap_needed:
                            overlap_sentences.insert(0, s)
                            total_overlap_size += s_size
                        else:
                            break
                    
                    current_chunk = overlap_sentences
                    current_size = total_overlap_size
                
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Add the last chunk if it has content
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_idx": len(chunks),
                    "chunk_size": self._measure_text(chunk_text)
                }
            })
        
        logger.info(f"Created {len(chunks)} chunks with sentence boundaries")
        return chunks

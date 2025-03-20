"""
Semantic chunking strategy that attempts to keep related content together
based on semantic similarity and natural topic boundaries.
"""

import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class SemanticChunker:
    """
    Chunker that attempts to identify semantic boundaries in text.
    
    This chunker uses text analysis to identify natural topic transitions
    and creates chunks that preserve semantic coherence.
    """
    
    def __init__(
        self,
        max_chunk_size: int = 1500,
        min_chunk_size: int = 500,
        similarity_threshold: float = 0.5,
        embedding_function: Optional[Callable] = None
    ):
        """
        Initialize the semantic chunker.
        
        Args:
            max_chunk_size: Maximum size of each chunk in characters
            min_chunk_size: Minimum size of each chunk in characters
            similarity_threshold: Threshold for determining semantic similarity
            embedding_function: Optional function to use for embedding computation
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.similarity_threshold = similarity_threshold
        self.embedding_function = embedding_function
        
        # Initialize a default embedding function using TF-IDF if none provided
        if self.embedding_function is None:
            self.vectorizer = TfidfVectorizer(stop_words='english')
            
        logger.info(f"Initialized SemanticChunker with max_chunk_size={max_chunk_size}, "
                   f"min_chunk_size={min_chunk_size}, similarity_threshold={similarity_threshold}")
    
    def _default_embedding(self, texts: List[str]) -> np.ndarray:
        """Compute embeddings using TF-IDF if no custom embedding function is provided."""
        try:
            return self.vectorizer.fit_transform(texts).toarray()
        except Exception as e:
            logger.error(f"Error computing TF-IDF embeddings: {str(e)}")
            # Fallback to a simple embedding if TF-IDF fails
            return np.array([[len(set(text.split())) for _ in range(10)] for text in texts])
    
    def _compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """Compute embeddings for a list of text segments."""
        if not texts:
            return np.array([])
            
        if self.embedding_function:
            try:
                return self.embedding_function(texts)
            except Exception as e:
                logger.error(f"Error using custom embedding function: {str(e)}")
                return self._default_embedding(texts)
        else:
            return self._default_embedding(texts)
    
    def _calculate_similarities(self, embeddings: np.ndarray) -> np.ndarray:
        """Calculate pairwise cosine similarities between adjacent text segments."""
        if len(embeddings) <= 1:
            return np.array([])
            
        # Calculate similarity between adjacent paragraphs
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
            similarities.append(sim)
            
        return np.array(similarities)
    
    def _identify_chunk_boundaries(self, similarities: np.ndarray, paragraph_lengths: List[int]) -> List[int]:
        """
        Identify optimal chunk boundaries based on semantic similarity and size constraints.
        
        This algorithm looks for natural topic transitions (low similarity points)
        while respecting the minimum and maximum chunk size constraints.
        """
        if len(similarities) == 0:
            return [0]
            
        # Find potential boundary points (where similarity drops below threshold)
        potential_boundaries = [0]  # Always include start point
        current_size = 0
        
        for i, (sim, length) in enumerate(zip(similarities, paragraph_lengths[:-1])):
            current_size += length
            
            # If we exceed min_chunk_size and find a similarity below threshold,
            # or we exceed max_chunk_size, create a boundary
            if ((current_size >= self.min_chunk_size and sim < self.similarity_threshold) or 
                current_size >= self.max_chunk_size):
                potential_boundaries.append(i + 1)  # The boundary is after the current paragraph
                current_size = 0
        
        # Add the end point if not already included
        if potential_boundaries[-1] != len(similarities):
            potential_boundaries.append(len(similarities) + 1)
            
        return potential_boundaries
    
    def create_chunks(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Split text into semantically coherent chunks.
        
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
        
        # Split text into paragraphs for semantic analysis
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        if not paragraphs:
            logger.warning("No valid paragraphs found in text")
            return []
        
        # For very short texts, return as a single chunk
        if len(text) < self.min_chunk_size:
            return [{
                "text": text,
                "metadata": {
                    **metadata,
                    "chunk_idx": 0,
                    "chunk_size": len(text)
                }
            }]
        
        # Compute paragraph lengths and embeddings
        paragraph_lengths = [len(p) for p in paragraphs]
        embeddings = self._compute_embeddings(paragraphs)
        
        # Calculate similarities between adjacent paragraphs
        similarities = self._calculate_similarities(embeddings)
        
        # Identify optimal chunk boundaries
        boundaries = self._identify_chunk_boundaries(similarities, paragraph_lengths)
        
        # Create chunks based on identified boundaries
        chunks = []
        
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            
            chunk_paragraphs = paragraphs[start_idx:end_idx]
            chunk_text = '\n\n'.join(chunk_paragraphs)
            
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_idx": len(chunks),
                    "chunk_size": len(chunk_text),
                    "paragraph_count": len(chunk_paragraphs),
                    "semantic_boundary": True
                }
            })
        
        logger.info(f"Created {len(chunks)} semantically coherent chunks")
        return chunks
    
    def create_chunks_with_headers(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Split text into chunks while respecting header-based structure.
        
        This method uses document structure (headers) as primary chunk boundaries
        and then applies semantic chunking within large sections.
        
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
        
        # Pattern to identify headers (markdown or similar formats)
        header_pattern = r'(?:^|\n)(#{1,6}\s+.+?|[^\n]+?\n[=-]{2,})'
        
        # Split text by headers
        sections = []
        last_end = 0
        current_header = "Introduction"  # Default header for the beginning
        
        for match in re.finditer(header_pattern, text, re.MULTILINE):
            # Add the preceding text as a section
            if last_end > 0:  # Skip for the first header
                section_text = text[last_end:match.start()]
                if section_text.strip():
                    sections.append((current_header, section_text))
            
            # Update the current header
            current_header = match.group(1).strip()
            last_end = match.end()
        
        # Add the last section
        if last_end < len(text):
            section_text = text[last_end:]
            if section_text.strip():
                sections.append((current_header, section_text))
        
        # If no sections were identified or text is very short, use regular semantic chunking
        if not sections:
            return self.create_chunks(text, metadata)
        
        # Process each section
        chunks = []
        
        for header, section_text in sections:
            # If section is small enough, keep it as one chunk
            if len(section_text) <= self.max_chunk_size:
                chunks.append({
                    "text": section_text,
                    "metadata": {
                        **metadata,
                        "chunk_idx": len(chunks),
                        "section_header": header,
                        "chunk_size": len(section_text)
                    }
                })
            else:
                # For large sections, apply semantic chunking
                section_chunks = self.create_chunks(section_text)
                
                # Add the section header to each chunk's metadata
                for chunk in section_chunks:
                    chunk_metadata = chunk["metadata"]
                    chunk_metadata.update({
                        "section_header": header,
                        "chunk_idx": len(chunks)  # Update chunk index
                    })
                    chunks.append({
                        "text": chunk["text"],
                        "metadata": {**metadata, **chunk_metadata}
                    })
        
        logger.info(f"Created {len(chunks)} header-based chunks")
        return chunks

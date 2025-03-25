from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import os
import numpy as np
import logging
import hashlib
import re
from collections import defaultdict
import json
import tempfile

# Import S3 manager for loading documents
try:
    from src.data.storage.s3_handler import S3FileManager
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class Source:
    document: str
    context: str
    score: float

@dataclass
class Response:
    answer: str
    sources: List[Source]

class NaiveRAG:
    """
    Simple similarity-based retrieval implementation without using a vector database.
    This implementation computes embeddings and performs cosine similarity manually.
    """
    
    def __init__(self, documents_dir: str = None, load_from_s3: bool = True):
        """
        Initialize the naive RAG method.
        
        Args:
            documents_dir: Optional directory containing text documents to load
            load_from_s3: Whether to attempt loading documents from S3
        """
        # Initialize document store (text -> metadata)
        self.documents = {}
        self.embeddings = {}
        self.dimension = 384  # Dimension for embeddings
        self.citations = []  # Store citations from last query
        
        # Load documents if specified
        if documents_dir and os.path.isdir(documents_dir):
            self._load_from_directory(documents_dir)
        
        # Load from S3 if enabled
        if load_from_s3 and S3_AVAILABLE:
            self._load_from_s3()
            
        logger.info(f"Initialized NaiveRAG with {len(self.documents)} documents")
    
    def _load_from_s3(self) -> None:
        """Load processed documents from S3."""
        try:
            s3_manager = S3FileManager()
            loaded_count = 0
            
            # Log what we're doing
            logger.info("Attempting to load documents from S3...")
            
            # Try multiple directories where documents might be stored
            for prefix in ["processed/", "reports/", "downloaded_reports/", "processed_reports/"]:
                # List files in the current directory
                files = s3_manager.list_files(prefix=prefix)
                
                if not files:
                    logger.info(f"No files found in {prefix}")
                    continue
                else:
                    logger.info(f"Found {len(files)} files in {prefix}")
                
                # Filter for text files first
                text_files = [f for f in files if f.endswith('.txt') or f.endswith('.md') or f.endswith('.json')]
                
                if text_files:
                    logger.info(f"Found {len(text_files)} text files in {prefix}")
                
                # If no text files but there are PDFs, just log it and continue
                if not text_files:
                    pdf_files = [f for f in files if f.endswith('.pdf')]
                    if pdf_files:
                        logger.warning(f"Found {len(pdf_files)} PDF files in {prefix} but no processed text. PDFs need processing first.")
                    continue
                
                # Process each text file
                for file_path in text_files:
                    try:
                        # Get document content from S3
                        text_content = s3_manager.read_text(file_path)
                        if not text_content:
                            logger.warning(f"Empty text content in {file_path}")
                            continue
                            
                        # Extract year from filename for metadata using a more flexible pattern
                        year = "Unknown"
                        # Look for any 4-digit sequence that could be a year
                        match = re.search(r'(\d{4})', file_path)
                        if match:
                            year = match.group(1)
                        
                        # Create metadata
                        metadata = {
                            "source": os.path.basename(file_path),
                            "s3_path": file_path,
                            "year": year
                        }
                        
                        # Add to documents
                        self.add_document(text_content, metadata)
                        loaded_count += 1
                        logger.debug(f"Loaded document: {file_path} (Year: {year})")
                        
                    except Exception as e:
                        logger.error(f"Error loading document {file_path} from S3: {str(e)}")
            
            if loaded_count > 0:
                logger.info(f"Successfully loaded {loaded_count} documents from S3")
            else:
                logger.warning("No documents were loaded from S3. Check that your processed documents exist and are in the correct format.")
            
        except Exception as e:
            logger.error(f"Error loading documents from S3: {str(e)}")
    
    def _load_documents(self, directory: str) -> None:
        """
        Load documents from a directory.
        
        Args:
            directory: Directory containing text documents
        """
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                filepath = os.path.join(directory, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text = f.read()
                    
                    # Add document to store with metadata
                    self.add_document(text, {"source": filename})
                    
                except Exception as e:
                    logger.error(f"Error loading document {filename}: {str(e)}")
    
    def add_document(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """
        Add a document to the RAG system.
        
        Args:
            text: The document text
            metadata: Optional metadata for the document
            
        Returns:
            Document ID
        """
        # Generate a document ID
        doc_id = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        # Store document with metadata
        self.documents[doc_id] = {
            "text": text,
            "metadata": metadata or {}
        }
        
        # Generate and store embedding
        self.embeddings[doc_id] = self._generate_embedding(text)
        
        logger.info(f"Added document with ID {doc_id[:8]}...")
        return doc_id
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate a simple deterministic embedding for the text.
        
        This is a simplified implementation for demonstration purposes.
        In a real application, you would use a proper embedding model.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        # Create a deterministic seed from the text
        text_hash = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16) % 10**8
        np.random.seed(text_hash)
        
        # Generate a random embedding vector
        embedding = np.random.rand(self.dimension).astype(np.float32)
        
        # Normalize the vector for cosine similarity
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def _compute_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        # Since vectors are normalized, dot product equals cosine similarity
        return float(np.dot(vec1, vec2))
    
    def _chunk_text(self, text: str, chunk_size: int = 100, overlap: int = 20) -> List[Tuple[str, int]]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            chunk_size: Maximum chunk size in words
            overlap: Number of overlapping words between chunks
            
        Returns:
            List of (chunk, start_position) tuples
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append((chunk, i))
            
            if i + chunk_size >= len(words):
                break
                
        return chunks
    
    def query(self, query_text: str, top_k: int = 5, threshold: float = 0.0, year_filter: str = None) -> str:
        """
        Process a query and return an answer.
        
        Args:
            query_text: Query text
            top_k: Number of top documents to consider
            threshold: Minimum similarity threshold
            year_filter: Optional year to filter documents by
            
        Returns:
            str: Generated answer
        """
        # Check if we have documents
        if not self.documents:
            return "No documents have been loaded. Please add documents first."
        
        # Apply year filter if specified
        filtered_docs = self.documents
        
        # First check the parameter
        if year_filter:
            filtered_docs = {
                doc_id: doc_info 
                for doc_id, doc_info in self.documents.items() 
                if doc_info.get('metadata', {}).get('year') == year_filter
            }
        # Then check the class attribute as fallback
        elif hasattr(self, 'year_filter') and self.year_filter:
            filtered_docs = {
                doc_id: doc_info 
                for doc_id, doc_info in self.documents.items() 
                if doc_info.get('metadata', {}).get('year') == self.year_filter
            }
            
        # If no documents match the filter
        if not filtered_docs:
            if year_filter:
                return f"No documents found for year {year_filter}. Please try another year or remove the filter."
            elif hasattr(self, 'year_filter') and self.year_filter:
                return f"No documents found for year {self.year_filter}. Please try another year or remove the filter."
            else:
                return "No relevant documents found for your query."
        
        # Compute query embedding
        query_embedding = self._generate_embedding(query_text)
        
        # Find top-k most similar documents
        similarities = {}
        for doc_id, doc_info in filtered_docs.items():
            if doc_id in self.embeddings:
                similarity = self._compute_cosine_similarity(query_embedding, self.embeddings[doc_id])
                similarities[doc_id] = similarity
        
        # Sort by similarity (descending) and filter by threshold
        sorted_docs = sorted(
            [(doc_id, sim) for doc_id, sim in similarities.items() if sim >= threshold],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Get top-k documents
        top_docs = sorted_docs[:top_k]
        
        if not top_docs:
            return "No relevant documents found for your query."
        
        # Extract text and create sources
        sources = []
        all_text = ""
        
        for doc_id, similarity in top_docs:
            doc_info = filtered_docs[doc_id]
            doc_text = doc_info["text"]
            doc_metadata = doc_info.get("metadata", {})
            
            # Extract document name from metadata
            doc_name = doc_metadata.get("source", doc_id[:8])
            
            # Extract most relevant context
            context = self._extract_relevant_context(query_text, doc_text)
            
            source = Source(
                document=doc_name,
                context=context,
                score=similarity
            )
            
            sources.append(source)
            all_text += f"\n\n{context}"
        
        # Generate citations for the sources
        self.citations = []
        for i, source in enumerate(sources):
            metadata = None
            for doc_id, doc_info in filtered_docs.items():
                if doc_info.get("metadata", {}).get("source") == source.document:
                    metadata = doc_info.get("metadata", {})
                    break
            
            if metadata:
                year = metadata.get("year", "Unknown Year")
                source_path = metadata.get("s3_path", "")
                citation = f"**{source.document}** (NVIDIA {year})\n\nRelevant excerpt:\n\n{source.context}\n\nScore: {source.score:.2f}"
            else:
                citation = f"**Source {i+1}**\n\n{source.context}\n\nScore: {source.score:.2f}"
            
            self.citations.append(citation)
        
        # Generate a better answer based on the top documents
        answer = self._generate_improved_answer(query_text, all_text)
        
        return answer
    
    def _format_context(self, text: str, query: str) -> str:
        """Format context text for better readability."""
        # Limit to 500 characters and add ellipsis if needed
        max_length = 500
        if len(text) > max_length:
            return text[:max_length] + "..."
        return text
    
    def _generate_improved_answer(self, query: str, text: str) -> str:
        """Generate a more readable answer from the text."""
        # Identify key financial metrics
        financial_metrics = self._extract_financial_metrics(text)
        
        # Find relevant sentences
        sentences = re.split(r'[.!?]+', text)
        query_terms = set(query.lower().split())
        relevant_sentences = []
        
        # Keywords to look for in financial reports
        financial_keywords = [
            "revenue", "income", "profit", "margin", "earnings", "growth", 
            "billion", "million", "quarter", "year", "increase", "decrease",
            "eps", "dividend", "guidance", "outlook", "forecast"
        ]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if sentence contains query terms or financial keywords
            sentence_terms = set(sentence.lower().split())
            term_overlap = len(query_terms.intersection(sentence_terms))
            
            has_financial_keyword = any(keyword in sentence.lower() for keyword in financial_keywords)
            
            if term_overlap > 0 or has_financial_keyword:
                # Clean up the sentence (remove excessive whitespace, etc.)
                cleaned = re.sub(r'\s+', ' ', sentence).strip()
                if cleaned and len(cleaned) > 10:  # Avoid tiny fragments
                    relevant_sentences.append(cleaned)
        
        if financial_metrics:
            summary = "Based on the documents, NVIDIA's key financial metrics include:\n\n"
            for metric, value in financial_metrics.items():
                summary += f"- {metric}: {value}\n"
                
            if relevant_sentences:
                summary += "\nAdditional information:\n"
                for i, sentence in enumerate(relevant_sentences[:5]):  # Limit to 5 sentences
                    summary += f"- {sentence}\n"
            
            return summary
        elif relevant_sentences:
            summary = "Based on the documents analyzed, here are the key points about NVIDIA's financial performance:\n\n"
            for i, sentence in enumerate(relevant_sentences[:7]):  # Limit to 7 sentences
                summary += f"- {sentence}\n"
            return summary
        else:
            return "I found information related to NVIDIA's financial performance, but couldn't extract specific key metrics. Please check the source documents for more details."
    
    def _extract_financial_metrics(self, text: str) -> Dict[str, str]:
        """Extract financial metrics from text."""
        metrics = {}
        
        # Regular expressions for common financial metrics
        patterns = {
            "Revenue": r"revenue\s+of\s+\$?([\d.,]+)\s*(billion|million|B|M)?",
            "Net Income": r"net\s+income\s+of\s+\$?([\d.,]+)\s*(billion|million|B|M)?",
            "EPS": r"earnings\s+per\s+share\s+of\s+\$?([\d.,]+)",
            "Gross Margin": r"gross\s+margin\s+of\s+([\d.,]+)\s*%",
            "Operating Income": r"operating\s+income\s+of\s+\$?([\d.,]+)\s*(billion|million|B|M)?",
        }
        
        # Search for metrics
        for metric, pattern in patterns.items():
            match = re.search(pattern, text.lower())
            if match:
                value = match.group(1)
                unit = match.group(2) if len(match.groups()) > 1 else ""
                metrics[metric] = f"${value} {unit}".strip()
        
        # Look for year-over-year growth
        growth_match = re.search(r"(increased|decreased|grew|declined)\s+by\s+([\d.,]+)\s*%", text.lower())
        if growth_match:
            direction = growth_match.group(1)
            percentage = growth_match.group(2)
            metrics["Year-over-year growth"] = f"{direction} by {percentage}%"
        
        return metrics
    
    def get_citations(self) -> List[str]:
        """Get citations from the last query."""
        return self.citations

    def _extract_relevant_context(self, query: str, text: str, max_length: int = 2000) -> str:
        """
        Extract the most relevant context from a document for a given query.
        
        Args:
            query: The query text
            text: The document text
            max_length: Maximum length of context to return
            
        Returns:
            str: The most relevant context
        """
        # For short documents, just return the text
        if len(text) <= max_length:
            return text
            
        # Split text into paragraphs
        paragraphs = [p for p in re.split(r'\n\s*\n', text) if p.strip()]
        
        if not paragraphs:
            # If no clear paragraphs, split by sentences
            sentences = [s for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
            if not sentences:
                # Last resort: return the first max_length characters
                return text[:max_length]
            
            # Try to find paragraphs with query terms
            query_terms = [term.lower() for term in query.split() if len(term) > 3]
            relevant_sentences = []
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(term in sentence_lower for term in query_terms):
                    relevant_sentences.append(sentence)
            
            # If we found relevant sentences, join them
            if relevant_sentences:
                context = " ".join(relevant_sentences)
                if len(context) <= max_length:
                    return context
                else:
                    return context[:max_length]
            
            # Otherwise return first portion of text
            return " ".join(sentences[:10])
            
        # Try to find paragraphs with query terms
        query_terms = [term.lower() for term in query.split() if len(term) > 3]
        paragraph_scores = []
        
        for i, paragraph in enumerate(paragraphs):
            paragraph_lower = paragraph.lower()
            score = sum(1 for term in query_terms if term in paragraph_lower)
            paragraph_scores.append((i, score))
        
        # Sort paragraphs by relevance score (descending)
        paragraph_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get top paragraphs until we reach max_length
        context = ""
        for i, _ in paragraph_scores:
            paragraph = paragraphs[i]
            if len(context) + len(paragraph) + 2 <= max_length:
                context += paragraph + "\n\n"
            else:
                # If adding the full paragraph would exceed max_length,
                # add as much as we can
                remaining = max_length - len(context)
                if remaining > 50:  # Only add if we can add a meaningful amount
                    context += paragraph[:remaining]
                break
                
        return context.strip()

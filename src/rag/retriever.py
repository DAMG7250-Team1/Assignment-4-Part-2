"""
Retriever for RAG pipeline.

This module provides functionality to retrieve relevant documents from 
a vector store based on a query.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

from src.vectorstore.base import BaseVectorStore

logger = logging.getLogger(__name__)

class Retriever:
    """
    Retriever for RAG pipeline.
    
    This class handles retrieving relevant documents from a vector store
    based on a query.
    """
    
    def __init__(self, 
                vector_store: BaseVectorStore,
                top_k: int = 4,
                score_threshold: Optional[float] = None,
                reranker: Optional[Callable] = None):
        """
        Initialize the retriever.
        
        Args:
            vector_store: Vector store to retrieve documents from
            top_k: Number of documents to retrieve
            score_threshold: Optional threshold for similarity scores
            reranker: Optional function to rerank retrieved documents
        """
        self.vector_store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.reranker = reranker
        
        logger.info(f"Initialized Retriever with top_k={top_k}")
    
    def retrieve(self, 
                query: str, 
                filter: Optional[Dict[str, Any]] = None,
                top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents based on a query.
        
        Args:
            query: The query text
            filter: Optional filter for the search
            top_k: Optional override for the number of documents to retrieve
            
        Returns:
            List of retrieved documents with metadata and scores
        """
        if not query:
            logger.warning("Empty query provided to retrieve")
            return []
        
        # Use instance top_k if not provided
        if top_k is None:
            top_k = self.top_k
        
        # Query vector store
        try:
            results = self.vector_store.similarity_search(
                query=query,
                k=top_k,
                filter=filter
            )
            
            if not results:
                logger.info("No results found for query")
                return []
            
            # Format results
            formatted_results = []
            
            for text, metadata, score in results:
                # Apply score threshold if specified
                if self.score_threshold is not None and score < self.score_threshold:
                    continue
                    
                formatted_results.append({
                    "text": text,
                    "metadata": metadata,
                    "score": score
                })
            
            # Apply reranker if specified
            if self.reranker and formatted_results:
                try:
                    formatted_results = self.reranker(query, formatted_results)
                except Exception as e:
                    logger.error(f"Error applying reranker: {str(e)}")
            
            logger.info(f"Retrieved {len(formatted_results)} documents for query")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def retrieve_with_mmr(self, 
                         query: str, 
                         filter: Optional[Dict[str, Any]] = None,
                         top_k: Optional[int] = None,
                         lambda_mult: float = 0.5,
                         fetch_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using Maximum Marginal Relevance.
        
        This provides a more diverse set of results by balancing relevance
        with information diversity.
        
        Args:
            query: The query text
            filter: Optional filter for the search
            top_k: Optional override for the number of documents to retrieve
            lambda_mult: Controls trade-off between relevance and diversity
            fetch_k: Number of initial documents to consider before reranking
            
        Returns:
            List of retrieved documents with metadata and scores
        """
        # This is a simple MMR implementation
        # Get initial results
        if top_k is None:
            top_k = self.top_k
            
        if fetch_k is None:
            fetch_k = top_k * 3  # Default: fetch 3x more docs than needed
        
        # Get initial results
        initial_results = self.retrieve(
            query=query,
            filter=filter,
            top_k=fetch_k
        )
        
        if len(initial_results) <= top_k:
            return initial_results
        
        # Apply MMR algorithm
        selected = [initial_results[0]]  # Start with the most relevant document
        remaining = initial_results[1:]
        
        while len(selected) < top_k and remaining:
            # Calculate MMR scores for remaining documents
            next_best_score = -1
            next_best_doc = None
            next_best_index = -1
            
            for i, doc in enumerate(remaining):
                # Calculate relevance component
                relevance = doc["score"]
                
                # Calculate diversity component (max similarity to any already selected doc)
                max_similarity = 0
                for selected_doc in selected:
                    # Simplified similarity calculation based on overlapping content
                    # In a real implementation, you'd compute actual similarity
                    similarity = self._calculate_text_similarity(doc["text"], selected_doc["text"])
                    max_similarity = max(max_similarity, similarity)
                
                # Calculate MMR score
                mmr_score = lambda_mult * relevance - (1 - lambda_mult) * max_similarity
                
                if mmr_score > next_best_score:
                    next_best_score = mmr_score
                    next_best_doc = doc
                    next_best_index = i
            
            if next_best_doc is not None:
                selected.append(next_best_doc)
                remaining.pop(next_best_index)
            else:
                break
        
        logger.info(f"Retrieved {len(selected)} documents using MMR")
        return selected
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate a simple text similarity score.
        
        This is a basic implementation for the MMR algorithm. In practice,
        you'd use a proper similarity measure.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score
        """
        # Convert texts to sets of words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Calculate Jaccard similarity
        if not words1 or not words2:
            return 0.0
            
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0 
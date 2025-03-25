"""
Base interface for vector store implementations.

This module defines the abstract base class that all vector store implementations
must inherit from to ensure a consistent interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple

class BaseVectorStore(ABC):
    """
    Abstract base class for vector stores.
    
    This class defines the interface that all vector store implementations
    must adhere to for compatibility with the RAG pipeline.
    """
    
    @abstractmethod
    def __init__(self, **kwargs):
        """
        Initialize the vector store with configuration parameters.
        
        Args:
            **kwargs: Implementation-specific configuration parameters
        """
        pass
    
    @abstractmethod
    def add_texts(self, 
                  texts: List[str], 
                  metadatas: Optional[List[Dict[str, Any]]] = None, 
                  ids: Optional[List[str]] = None, 
                  **kwargs) -> List[str]:
        """
        Add texts to the vector store.
        
        Args:
            texts: List of text strings to add
            metadatas: Optional list of metadata dictionaries associated with each text
            ids: Optional list of ids to associate with each text
            **kwargs: Additional implementation-specific parameters
            
        Returns:
            List of ids of the added texts
        """
        pass
    
    @abstractmethod
    def similarity_search(self, 
                          query: str, 
                          k: int = 4, 
                          filter: Optional[Dict[str, Any]] = None, 
                          **kwargs) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        Search for texts similar to the query text.
        
        Args:
            query: The query text
            k: Number of results to return
            filter: Optional filter to apply to the search
            **kwargs: Additional implementation-specific parameters
            
        Returns:
            List of tuples containing (text, metadata, score)
        """
        pass
    
    @abstractmethod
    def delete(self, 
               ids: Optional[List[str]] = None, 
               filter: Optional[Dict[str, Any]] = None, 
               **kwargs) -> bool:
        """
        Delete texts from the vector store.
        
        Args:
            ids: Optional list of ids to delete
            filter: Optional filter to apply when selecting texts to delete
            **kwargs: Additional implementation-specific parameters
            
        Returns:
            True if the deletion was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_by_id(self, id: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Retrieve a text by its id.
        
        Args:
            id: The id of the text to retrieve
            
        Returns:
            Tuple containing (text, metadata) if found, None otherwise
        """
        pass
    
    @abstractmethod
    def update_text(self, 
                    id: str, 
                    text: str, 
                    metadata: Optional[Dict[str, Any]] = None, 
                    **kwargs) -> bool:
        """
        Update a text in the vector store.
        
        Args:
            id: The id of the text to update
            text: The new text
            metadata: Optional new metadata
            **kwargs: Additional implementation-specific parameters
            
        Returns:
            True if the update was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def count(self, filter: Optional[Dict[str, Any]] = None) -> int:
        """
        Count the number of texts in the vector store.
        
        Args:
            filter: Optional filter to apply when counting
            
        Returns:
            The number of texts in the vector store
        """
        pass
    
    def search_by_metadata(self, 
                          metadata_filter: Dict[str, Any], 
                          limit: int = 10) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Search for texts based on metadata.
        
        This is a default implementation that can be overridden by subclasses
        that have more efficient ways to search by metadata.
        
        Args:
            metadata_filter: Filter to apply to the metadata
            limit: Maximum number of results to return
            
        Returns:
            List of tuples containing (text, metadata)
        """
        # Default implementation might be less efficient, so subclasses
        # are encouraged to override this with more optimized versions
        pass 
"""
ChromaDB vector store implementation.

This module implements the BaseVectorStore interface using ChromaDB
as the underlying vector database.
"""

import logging
import os
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from src.vectorstore.base import BaseVectorStore

logger = logging.getLogger(__name__)

class ChromaDBVectorStore(BaseVectorStore):
    """
    ChromaDB implementation of the BaseVectorStore interface.
    
    This class provides methods to interact with a ChromaDB instance
    for vector search and storage.
    """
    
    def __init__(self, 
                collection_name: str = "documents",
                persist_directory: Optional[str] = None,
                embedding_function_name: str = "openai",
                embedding_function: Optional[Any] = None,
                distance_func: str = "cosine",
                **kwargs):
        """
        Initialize the ChromaDB vector store.
        
        Args:
            collection_name: Name of the collection to use
            persist_directory: Directory to persist the database (if None, in-memory DB is used)
            embedding_function_name: Name of the embedding function to use if no function provided
            embedding_function: Custom embedding function (optional)
            distance_func: Distance function for similarity search
            **kwargs: Additional parameters to pass to ChromaDB
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.distance_func = distance_func
        
        # Initialize embedding function
        if embedding_function:
            self.embedding_function = embedding_function
        else:
            self._setup_embedding_function(embedding_function_name)
        
        # Initialize ChromaDB client
        self._initialize_client()
        
        # Get or create the collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": distance_func}
        )
        
        logger.info(f"Initialized ChromaDB vector store with collection '{collection_name}'")
    
    def _setup_embedding_function(self, embedding_function_name: str):
        """Set up the embedding function based on the provided name."""
        ef_provider = embedding_functions.DefaultEmbeddingFunction
        
        if embedding_function_name == "openai":
            try:
                ef_provider = embedding_functions.OpenAIEmbeddingFunction
                openai_api_key = os.environ.get("OPENAI_API_KEY")
                if not openai_api_key:
                    logger.warning("OPENAI_API_KEY not found in environment, switching to default embedding function")
                    ef_provider = embedding_functions.DefaultEmbeddingFunction
                else:
                    self.embedding_function = ef_provider(api_key=openai_api_key, model_name="text-embedding-3-small")
                    return
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI embedding function: {str(e)}")
                ef_provider = embedding_functions.DefaultEmbeddingFunction
        
        elif embedding_function_name == "cohere":
            try:
                ef_provider = embedding_functions.CohereEmbeddingFunction
                cohere_api_key = os.environ.get("COHERE_API_KEY")
                if not cohere_api_key:
                    logger.warning("COHERE_API_KEY not found in environment, switching to default embedding function")
                    ef_provider = embedding_functions.DefaultEmbeddingFunction
                else:
                    self.embedding_function = ef_provider(api_key=cohere_api_key)
                    return
            except Exception as e:
                logger.error(f"Failed to initialize Cohere embedding function: {str(e)}")
                ef_provider = embedding_functions.DefaultEmbeddingFunction
        
        # Default to the all-MiniLM-L6-v2 model
        try:
            self.embedding_function = ef_provider()
        except Exception as e:
            logger.error(f"Failed to initialize default embedding function: {str(e)}")
            # Use None, which will make ChromaDB use its internal default
            self.embedding_function = None
    
    def _initialize_client(self):
        """Initialize the ChromaDB client based on configuration."""
        client_settings = Settings()
        
        if self.persist_directory:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=client_settings
            )
            logger.info(f"Initialized persistent ChromaDB client at {self.persist_directory}")
        else:
            self.client = chromadb.Client(settings=client_settings)
            logger.info("Initialized in-memory ChromaDB client")
    
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
            **kwargs: Additional parameters for ChromaDB
            
        Returns:
            List of ids of the added texts
        """
        if not texts:
            logger.warning("No texts provided to add_texts")
            return []
        
        # Generate ids if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        
        # If metadatas is not provided, create empty metadata for each text
        if metadatas is None:
            metadatas = [{} for _ in range(len(texts))]
        
        # Ensure all metadatas have the required structure
        for i, metadata in enumerate(metadatas):
            # Make sure metadata is a dictionary
            if metadata is None:
                metadatas[i] = {}
            
            # Convert non-string values to strings for ChromaDB compatibility
            for key, value in list(metadatas[i].items()):
                if not isinstance(value, (str, int, float, bool)):
                    metadatas[i][key] = str(value)
        
        try:
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(texts)} texts to ChromaDB collection '{self.collection_name}'")
            return ids
        except Exception as e:
            logger.error(f"Error adding texts to ChromaDB: {str(e)}")
            raise
    
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
            **kwargs: Additional parameters for ChromaDB
            
        Returns:
            List of tuples containing (text, metadata, score)
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                where=filter,
                **kwargs
            )
            
            # Process results to match the expected return format
            formatted_results = []
            
            if results["documents"] and len(results["documents"]) > 0:
                documents = results["documents"][0]  # First query results
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]
                
                for doc, meta, dist in zip(documents, metadatas, distances):
                    # Convert distance to similarity score (1 - distance for cosine)
                    if self.distance_func == "cosine":
                        similarity = 1 - dist
                    else:
                        # For other distance metrics, you might need different conversions
                        similarity = 1 / (1 + dist)
                    
                    formatted_results.append((doc, meta, similarity))
            
            logger.info(f"Found {len(formatted_results)} similar documents for query")
            return formatted_results
        
        except Exception as e:
            logger.error(f"Error performing similarity search in ChromaDB: {str(e)}")
            return []
    
    def delete(self, 
              ids: Optional[List[str]] = None, 
              filter: Optional[Dict[str, Any]] = None, 
              **kwargs) -> bool:
        """
        Delete texts from the vector store.
        
        Args:
            ids: Optional list of ids to delete
            filter: Optional filter to apply when selecting texts to delete
            **kwargs: Additional parameters for ChromaDB
            
        Returns:
            True if the deletion was successful, False otherwise
        """
        try:
            if ids:
                self.collection.delete(ids=ids)
                logger.info(f"Deleted {len(ids)} documents by IDs from ChromaDB")
            elif filter:
                # Get IDs matching the filter
                matching_results = self.collection.get(where=filter)
                if matching_results and matching_results["ids"]:
                    matching_ids = matching_results["ids"]
                    self.collection.delete(ids=matching_ids)
                    logger.info(f"Deleted {len(matching_ids)} documents by filter from ChromaDB")
            else:
                # Nothing to delete
                logger.warning("No IDs or filter provided for deletion")
                return False
                
            return True
        
        except Exception as e:
            logger.error(f"Error deleting documents from ChromaDB: {str(e)}")
            return False
    
    def get_by_id(self, id: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Retrieve a text by its id.
        
        Args:
            id: The id of the text to retrieve
            
        Returns:
            Tuple containing (text, metadata) if found, None otherwise
        """
        try:
            result = self.collection.get(ids=[id])
            
            if result and result["documents"] and len(result["documents"]) > 0:
                return (result["documents"][0], result["metadatas"][0])
            else:
                logger.warning(f"No document found with ID '{id}'")
                return None
        
        except Exception as e:
            logger.error(f"Error retrieving document from ChromaDB: {str(e)}")
            return None
    
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
            **kwargs: Additional parameters for ChromaDB
            
        Returns:
            True if the update was successful, False otherwise
        """
        try:
            # Prepare metadata
            if metadata is None:
                # Try to preserve existing metadata
                existing = self.get_by_id(id)
                metadata = existing[1] if existing else {}
            
            # Convert non-string values to strings for ChromaDB compatibility
            for key, value in list(metadata.items()):
                if not isinstance(value, (str, int, float, bool)):
                    metadata[key] = str(value)
            
            # Update the document
            self.collection.update(
                ids=[id],
                documents=[text],
                metadatas=[metadata]
            )
            
            logger.info(f"Updated document with ID '{id}' in ChromaDB")
            return True
        
        except Exception as e:
            logger.error(f"Error updating document in ChromaDB: {str(e)}")
            return False
    
    def count(self, filter: Optional[Dict[str, Any]] = None) -> int:
        """
        Count the number of texts in the vector store.
        
        Args:
            filter: Optional filter to apply when counting
            
        Returns:
            The number of texts in the vector store
        """
        try:
            if filter:
                # Get IDs matching the filter
                matching_results = self.collection.get(where=filter)
                if matching_results and matching_results["ids"]:
                    return len(matching_results["ids"])
                return 0
            else:
                # Get all documents count
                return self.collection.count()
        
        except Exception as e:
            logger.error(f"Error counting documents in ChromaDB: {str(e)}")
            return 0
    
    def search_by_metadata(self, 
                         metadata_filter: Dict[str, Any], 
                         limit: int = 10) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Search for texts based on metadata.
        
        Args:
            metadata_filter: Filter to apply to the metadata
            limit: Maximum number of results to return
            
        Returns:
            List of tuples containing (text, metadata)
        """
        try:
            results = self.collection.get(
                where=metadata_filter,
                limit=limit
            )
            
            if results and results["documents"]:
                return [
                    (doc, meta) 
                    for doc, meta in zip(results["documents"], results["metadatas"])
                ]
            else:
                logger.info(f"No documents found matching metadata filter")
                return []
        
        except Exception as e:
            logger.error(f"Error searching by metadata in ChromaDB: {str(e)}")
            return [] 
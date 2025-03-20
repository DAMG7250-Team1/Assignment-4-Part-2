"""
Pinecone vector store implementation.

This module implements the BaseVectorStore interface using Pinecone
as the underlying vector database.
"""

import logging
import os
import uuid
from typing import Dict, List, Any, Optional, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential

# Import Pinecone with proper error handling
try:
    # The package is called 'pinecone' not 'pinecone-client'
    import pinecone
except ImportError:
    raise ImportError("Could not import pinecone. Please install it with 'pip install pinecone'")

from src.vectorstore.base import BaseVectorStore

logger = logging.getLogger(__name__)

class PineconeVectorStore(BaseVectorStore):
    """
    Pinecone implementation of the BaseVectorStore interface.
    
    This class provides methods to interact with a Pinecone index
    for vector search and storage.
    """
    
    def __init__(self, 
                api_key: Optional[str] = None,
                index_name: str = "nvidia-reports",
                cloud: str = "aws", 
                region: str = "us-east-1"):
        """
        Initialize the Pinecone vector store.
        
        Args:
            api_key: Pinecone API key. If None, it will be read from the PINECONE_API_KEY environment variable.
            index_name: Name of the Pinecone index to use.
            cloud: Cloud provider where the index is hosted (aws, gcp, azure).
            region: Region where the index is hosted. Free plan supports regions like "us-east-1" on AWS.
        """
        # Get API key from environment if not provided
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("Pinecone API key must be provided or set as PINECONE_API_KEY environment variable")
        
        self.index_name = index_name
        self.dimension = 384  # Dimension for embeddings
        
        logger.info(f"Initializing Pinecone vector store with index '{index_name}'")
        
        # Initialize Pinecone client with v6 API
        pc = pinecone.Pinecone(api_key=self.api_key)
        
        # List existing indexes and create if not exists
        try:
            # Check if index exists
            existing_indexes = [index.name for index in pc.list_indexes()]
            logger.info(f"Existing Pinecone indexes: {existing_indexes}")
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating new Pinecone index '{index_name}'")
                try:
                    # Create the index with proper configuration
                    pc.create_index(
                        name=index_name,
                        dimension=self.dimension,
                        metric="cosine",
                        spec=pinecone.ServerlessSpec(
                            cloud=cloud,
                            region=region
                        )
                    )
                    logger.info(f"Waiting for Pinecone index '{index_name}' to be initialized...")
                    # Wait for index to be ready
                    import time
                    time.sleep(10)
                except Exception as e:
                    error_msg = str(e).lower()
                    # If the index already exists, we can ignore this error
                    if "already exists" in error_msg:
                        logger.info(f"Index '{index_name}' already exists")
                    else:
                        logger.error(f"Error creating Pinecone index: {e}")
                        raise
            
            # Connect to the index
            self.index = pc.Index(self.index_name)
            logger.info(f"Successfully connected to Pinecone index '{index_name}'")
            
        except Exception as e:
            logger.error(f"Error connecting to Pinecone: {e}")
            raise
        
        # Initialize simple embedding function
        try:
            import numpy as np
            
            def simple_encode(self, text):
                """Simple function to generate consistent embeddings for testing."""
                # Generate a deterministic but simple embedding based on the text
                # This is only for testing - in production, use a proper embedding model
                import hashlib
                # Use hash of text as seed for random generator
                text_hash = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16) % 10**8
                np.random.seed(text_hash)
                # Generate a random embedding vector
                embedding = np.random.rand(self.dimension).astype(np.float32)
                # Normalize the vector
                embedding = embedding / np.linalg.norm(embedding)
                return embedding
                
            # Create a class with the encode method
            class SimpleEncoder:
                def __init__(self, dimension):
                    self.dimension = dimension
                    
                def encode(self, text):
                    # Generate a deterministic but simple embedding based on the text
                    import hashlib
                    # Use hash of text as seed for random generator
                    text_hash = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16) % 10**8
                    np.random.seed(text_hash)
                    # Generate a random embedding vector
                    embedding = np.random.rand(self.dimension).astype(np.float32)
                    # Normalize the vector
                    embedding = embedding / np.linalg.norm(embedding)
                    return embedding
                
            self.model = SimpleEncoder(self.dimension)
            logger.info("Initialized simple embedding function for testing")
            
        except ImportError:
            logger.error("Failed to import numpy. Please install with 'pip install numpy'")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        Add documents to the Pinecone index with automatic embedding.
        
        Args:
            documents: List of document texts to add.
            metadata: Optional list of metadata dictionaries, one per document.
            
        Returns:
            List of IDs for the added documents.
        """
        if not documents:
            logger.warning("No documents provided to add_documents")
            return []
        
        # Ensure metadata is the same length as documents
        if metadata is None:
            metadata = [{} for _ in documents]
        elif len(metadata) != len(documents):
            raise ValueError(f"Length of metadata ({len(metadata)}) does not match length of documents ({len(documents)})")
        
        vectors = []
        ids = []
        
        for i, doc in enumerate(documents):
            doc_id = str(uuid.uuid4())
            ids.append(doc_id)
            
            # Generate embedding
            embedding = self.model.encode(doc)
            
            # Prepare the vector with text for automatic embedding
            vectors.append({
                "id": doc_id,
                "values": embedding.tolist(),
                "metadata": {
                    "text": doc,  # Store original text in metadata
                    **metadata[i]  # Include any additional metadata
                }
            })
        
        # Upsert vectors in batches of 100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            try:
                self.index.upsert(vectors=batch)
                logger.info(f"Successfully added batch of {len(batch)} documents to Pinecone")
            except Exception as e:
                logger.error(f"Error adding documents to Pinecone: {e}")
                raise
        
        return ids
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def search(self, query: str, top_k: int = 5, filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents using the query text.
        
        Args:
            query: The query text.
            top_k: Number of results to return.
            filter: Optional metadata filter to apply to the search.
            
        Returns:
            List of dictionaries containing id, text, metadata and score for each match.
        """
        try:
            # Generate embedding for query
            query_embedding = self.model.encode(query)
            
            # Query using embedding
            response = self.index.query(
                vector=query_embedding.tolist(),
                top_k=top_k,
                filter=filter,
                include_metadata=True
            )
            
            matches = []
            for match in response.matches:
                matches.append({
                    "id": match.id,
                    "text": match.metadata.get("text", ""),
                    "metadata": match.metadata,
                    "score": match.score
                })
            
            logger.info(f"Found {len(matches)} matches for query: {query[:50]}...")
            return matches
        
        except Exception as e:
            logger.error(f"Error searching Pinecone: {e}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
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
        try:
            if ids:
                # Delete specific IDs
                self.index.delete(ids=ids)
                logger.info(f"Successfully deleted {len(ids)} documents")
            elif filter:
                # Fetch IDs that match the filter
                # We use a zero vector to match based on metadata only
                zero_vector = [0.0] * self.dimension
                response = self.index.query(
                    vector=zero_vector,
                    filter=filter,
                    top_k=10000,  # Use a high number to get most matches
                    include_metadata=False
                )
                
                # Extract IDs from results
                ids_to_delete = [match.id for match in response.matches]
                
                if ids_to_delete:
                    # Delete the matching IDs
                    self.index.delete(ids=ids_to_delete)
                    logger.info(f"Successfully deleted {len(ids_to_delete)} documents matching filter")
                else:
                    logger.info("No documents matched the filter criteria for deletion")
            else:
                # No IDs or filter provided, clear the entire index
                self.index.delete(delete_all=True)
                logger.info("Successfully cleared all documents from the index")
                
            return True
            
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False
    
    def close(self) -> None:
        """
        Close the connection to Pinecone.
        """
        logger.info(f"Closing connection to Pinecone index '{self.index_name}'")
        # No explicit close method needed for Pinecone
        pass
    
    def clear(self) -> None:
        """
        Clear all vectors from the index.
        """
        try:
            self.index.delete(delete_all=True)
            logger.info(f"Successfully cleared all vectors from Pinecone index '{self.index_name}'")
        except Exception as e:
            logger.error(f"Error clearing Pinecone index: {e}")
            raise
            
    # Helper methods for compatibility with the BaseVectorStore interface
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None, 
                 ids: Optional[List[str]] = None, **kwargs) -> List[str]:
        """Add texts to the vector store."""
        # This is just a wrapper around add_documents for compatibility
        return self.add_documents(texts, metadatas)
    
    def get_by_id(self, id: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Retrieve a document by its ID."""
        try:
            response = self.index.fetch(ids=[id])
            
            # Extract the vector data for the given ID
            if id in response.vectors:
                vector_data = response.vectors[id]
                
                # Extract text from metadata
                text = vector_data.metadata.get("text", "")
                metadata = {k: v for k, v in vector_data.metadata.items() if k != "text"}
                
                return (text, metadata)
            else:
                logger.warning(f"No document found with ID '{id}'")
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving document from Pinecone: {e}")
            return None
            
    def count(self, filter: Optional[Dict[str, Any]] = None) -> int:
        """Count the number of documents in the index."""
        try:
            # Get index stats
            stats = self.index.describe_index_stats()
            
            # Get total vector count
            total_count = stats.total_vector_count
            
            logger.info(f"Total document count in Pinecone index: {total_count}")
            return total_count
            
        except Exception as e:
            logger.error(f"Error counting documents in Pinecone: {e}")
            return 0
            
    def search_by_metadata(self, metadata_filter: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """Search for documents based on metadata."""
        try:
            # Create a zero vector of the correct dimension
            zero_vector = [0.0] * self.dimension
            
            # Query with the filter
            response = self.index.query(
                vector=zero_vector,
                top_k=limit,
                filter=metadata_filter,
                include_metadata=True
            )
            
            # Process results
            matches = []
            for match in response.matches:
                matches.append({
                    "id": match.id,
                    "text": match.metadata.get("text", ""),
                    "metadata": match.metadata,
                    "score": match.score
                })
            
            logger.info(f"Found {len(matches)} documents matching metadata filter")
            return matches
            
        except Exception as e:
            logger.error(f"Error searching by metadata in Pinecone: {e}")
            return []
    
    # Add these new methods to implement the required abstract methods
    
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
        try:
            # Generate embedding for query
            query_embedding = self.model.encode(query)
            
            # Query using embedding
            response = self.index.query(
                vector=query_embedding.tolist(),
                top_k=k,
                filter=filter,
                include_metadata=True
            )
            
            # Format results as required by the abstract method
            formatted_results = []
            for match in response.matches:
                text = match.metadata.get("text", "")
                # Remove text from metadata to avoid duplication
                metadata = {k: v for k, v in match.metadata.items() if k != "text"}
                score = match.score
                formatted_results.append((text, metadata, score))
            
            logger.info(f"Found {len(formatted_results)} matches for query: {query[:50]}...")
            return formatted_results
        
        except Exception as e:
            logger.error(f"Error in similarity_search: {e}")
            raise
    
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
        try:
            # Generate embedding for the new text
            embedding = self.model.encode(text)
            
            # Prepare metadata
            if metadata is None:
                metadata = {}
                
            # Include the text in metadata
            metadata["text"] = text
            
            # Upsert vector with new embedding and metadata
            self.index.upsert(
                vectors=[{
                    "id": id,
                    "values": embedding.tolist(),
                    "metadata": metadata
                }]
            )
            
            logger.info(f"Successfully updated document with ID '{id}'")
            return True
            
        except Exception as e:
            logger.error(f"Error updating document with ID '{id}': {e}")
            return False 
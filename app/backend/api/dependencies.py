from fastapi import Depends, HTTPException, status
from typing import Generator, Optional
import os
import logging

from src.data.storage.s3_handler import S3FileManager
from src.vectorstore.vector_store_factory import VectorStoreFactory
from src.chunking.chunker_factory import ChunkerFactory
from app.backend.core.config import get_settings, Settings

# Configure logging
logger = logging.getLogger(__name__)

def get_s3_manager() -> S3FileManager:
    """
    Provides an S3FileManager instance.
    
    Returns:
        S3FileManager: Instance for handling S3 storage operations
    
    Raises:
        HTTPException: If S3 initialization fails
    """
    try:
        return S3FileManager()
    except Exception as e:
        logger.error(f"Failed to initialize S3 manager: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="S3 storage service unavailable"
        )

def get_vector_store(store_type: Optional[str] = None):
    """
    Creates and provides a vector store instance.
    
    Args:
        store_type: Type of vector store to create (chromadb, pinecone)
            If None, uses the default from settings
    
    Returns:
        BaseVectorStore: Vector store instance
    
    Raises:
        HTTPException: If vector store initialization fails
    """
    settings = get_settings()
    store_type = store_type or settings.DEFAULT_VECTOR_STORE
    
    try:
        logger.info(f"Creating {store_type} vector store")
        
        if store_type == "pinecone":
            return VectorStoreFactory.get_vector_store(
                store_type="pinecone",
                index_name=settings.PINECONE_INDEX,
                cloud=settings.PINECONE_CLOUD,
                region=settings.PINECONE_REGION
            )
        elif store_type == "chromadb":
            return VectorStoreFactory.get_vector_store(
                store_type="chromadb",
                collection_name=settings.CHROMA_COLLECTION,
                persist_directory=settings.CHROMA_PERSIST_DIR
            )
        else:
            raise ValueError(f"Unsupported vector store type: {store_type}")
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Vector store service unavailable: {str(e)}"
        )

def get_chunker(strategy: Optional[str] = None, **kwargs):
    """
    Creates and provides a chunker instance.
    
    Args:
        strategy: Chunking strategy to use
            If None, uses the default from settings
        **kwargs: Additional parameters for the chunker
    
    Returns:
        Chunker instance
    
    Raises:
        HTTPException: If chunker initialization fails
    """
    settings = get_settings()
    strategy = strategy or settings.DEFAULT_CHUNKING_STRATEGY
    
    try:
        logger.info(f"Creating chunker with strategy: {strategy}")
        return ChunkerFactory.get_chunker(strategy, **kwargs)
    except Exception as e:
        logger.error(f"Failed to initialize chunker: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid chunking strategy: {str(e)}"
        )

def verify_api_keys():
    """
    Verify that required API keys are present in environment variables.
    
    Raises:
        HTTPException: If required API keys are missing
    """
    # List of API keys that should be available
    required_keys = ["GEMINI_API_KEY", "PINECONE_API_KEY"]
    
    # Check for missing keys
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        logger.error(f"Missing required API keys: {', '.join(missing_keys)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Missing required API keys: {', '.join(missing_keys)}"
        )
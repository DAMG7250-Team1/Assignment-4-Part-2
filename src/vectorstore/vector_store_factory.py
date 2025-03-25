"""
Factory class for creating and retrieving vector store instances.
"""

import logging
from typing import Dict, Any, Optional, List, Union, Type, Callable
import os

# Import vector store implementations
from src.vectorstore.base import BaseVectorStore
from src.vectorstore.pinecone_store import PineconeVectorStore

# Try to import ChromaDB, but don't fail if it's not available
try:
    from src.vectorstore.chromadb_store import ChromaDBVectorStore
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("ChromaDB module not found. ChromaDB vector store will not be available.")

logger = logging.getLogger(__name__)

class VectorStoreFactory:
    """
    Factory class for creating vector store instances.
    
    This factory provides a unified interface for selecting and using
    different vector store implementations.
    """
    
    AVAILABLE_STORES = {
        "pinecone": PineconeVectorStore
    }
    
    # Add ChromaDB to available stores only if it's available
    if CHROMADB_AVAILABLE:
        AVAILABLE_STORES["chromadb"] = ChromaDBVectorStore
    
    @classmethod
    def get_vector_store(cls, 
                        store_type: str, 
                        **kwargs) -> BaseVectorStore:
        """
        Get a vector store instance based on the store type.
        
        Args:
            store_type: Type of vector store to create
            **kwargs: Additional parameters to pass to the vector store constructor
            
        Returns:
            An instance of the requested vector store
            
        Raises:
            ValueError: If the requested store type is not available
        """
        store_type = store_type.lower()
        
        if store_type not in cls.AVAILABLE_STORES:
            available_stores = ", ".join(cls.AVAILABLE_STORES.keys())
            raise ValueError(f"Unknown vector store type: '{store_type}'. "
                            f"Available types: {available_stores}")
        
        store_class = cls.AVAILABLE_STORES[store_type]
        logger.info(f"Creating vector store of type '{store_type}'")
        
        try:
            return store_class(**kwargs)
        except Exception as e:
            logger.error(f"Error creating vector store of type '{store_type}': {str(e)}")
            raise
    
    @classmethod
    def get_available_store_types(cls) -> List[str]:
        """
        Get a list of available vector store types.
        
        Returns:
            List of available vector store type names
        """
        return list(cls.AVAILABLE_STORES.keys())
    
    @classmethod
    def create_openai_embedding_function(cls, 
                                       model_name: str = "text-embedding-3-small", 
                                       batch_size: int = 8,
                                       **kwargs) -> Callable:
        """
        Create an OpenAI embedding function for use with vector stores.
        
        Args:
            model_name: Name of the OpenAI embedding model to use
            batch_size: Batch size for embedding requests
            **kwargs: Additional parameters for the OpenAI client
            
        Returns:
            Embedding function that converts text to vectors
        """
        try:
            from openai import OpenAI
            import numpy as np
            
            client = OpenAI(**kwargs)
            
            def get_embeddings(texts):
                """Get embeddings for a list of texts using the OpenAI API."""
                if isinstance(texts, str):
                    texts = [texts]
                
                all_embeddings = []
                
                # Process in batches
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    
                    response = client.embeddings.create(
                        model=model_name,
                        input=batch
                    )
                    
                    # Extract embeddings from response
                    embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(embeddings)
                
                # If a single text was provided, return a single embedding
                if len(all_embeddings) == 1 and isinstance(texts, str):
                    return np.array(all_embeddings[0])
                
                return all_embeddings
            
            logger.info(f"Created OpenAI embedding function using model '{model_name}'")
            return get_embeddings
            
        except Exception as e:
            logger.error(f"Error creating OpenAI embedding function: {str(e)}")
            raise
    
    @classmethod
    def create_huggingface_embedding_function(cls, 
                                            model_name: str = "BAAI/bge-small-en-v1.5", 
                                            device: str = "cpu",
                                            **kwargs) -> Callable:
        """
        Create a HuggingFace embedding function for use with vector stores.
        
        Args:
            model_name: Name of the HuggingFace model to use
            device: Device to use for inference (cpu or cuda)
            **kwargs: Additional parameters for the model
            
        Returns:
            Embedding function that converts text to vectors
        """
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            
            # Move model to specified device
            device = torch.device(device)
            model = model.to(device)
            
            def get_embeddings(texts):
                """Get embeddings for a list of texts using the HuggingFace model."""
                if isinstance(texts, str):
                    texts = [texts]
                
                # Tokenize texts
                encoded_input = tokenizer(
                    texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=512, 
                    return_tensors="pt"
                ).to(device)
                
                # Compute token embeddings
                with torch.no_grad():
                    model_output = model(**encoded_input)
                    
                # Mean pooling
                attention_mask = encoded_input["attention_mask"]
                token_embeddings = model_output.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                
                # Convert to numpy array
                embeddings = embeddings.cpu().numpy()
                
                # If a single text was provided, return a single embedding
                if len(embeddings) == 1 and isinstance(texts, str):
                    return embeddings[0]
                
                return embeddings
            
            logger.info(f"Created HuggingFace embedding function using model '{model_name}'")
            return get_embeddings
            
        except Exception as e:
            logger.error(f"Error creating HuggingFace embedding function: {str(e)}")
            raise 

    @staticmethod
    def create_google_embedding_function(model_name: str = "embedding-001", batch_size: int = 10, **kwargs):
        """
        Create an embedding function using Google's embedding API.
        
        Args:
            model_name: Name of the Google embedding model to use
            batch_size: Batch size for embedding requests
            **kwargs: Additional parameters for the Google client
            
        Returns:
            Embedding function that converts text to vectors
        """
        try:
            import google.generativeai as genai
            import numpy as np
            
            # Configure genai with API key from kwargs or environment
            api_key = kwargs.get("api_key") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("Google API key not provided and not found in environment")
                
            genai.configure(api_key=api_key)
            
            # Wrapper class to make the embedding function compatible with ChromaDB
            class GoogleEmbeddingFunction:
                def __init__(self):
                    self.model_name = model_name
                    self.batch_size = batch_size
                    
                def __call__(self, input):
                    """Get embeddings for a list of texts using the Google API."""
                    if isinstance(input, str):
                        input = [input]
                    
                    all_embeddings = []
                    
                    # Process in batches
                    for i in range(0, len(input), batch_size):
                        batch = input[i:i+batch_size]
                        
                        # Get embeddings for batch
                        batch_embeddings = []
                        for text in batch:
                            result = genai.embed_content(
                                model=f"models/{model_name}",
                                content=text,
                                task_type="retrieval_document"
                            )
                            if hasattr(result, "embedding"):
                                batch_embeddings.append(result.embedding)
                            else:
                                # Fallback if the response format is different
                                batch_embeddings.append(result["embedding"])
                        
                        all_embeddings.extend(batch_embeddings)
                    
                    return all_embeddings
            
            logger.info(f"Created Google embedding function using model '{model_name}'")
            return GoogleEmbeddingFunction()
            
        except ImportError:
            logger.error("Failed to import google.generativeai. Install with 'pip install google-generativeai'")
            raise
        except Exception as e:
            logger.error(f"Error creating Google embedding function: {str(e)}")
            raise 
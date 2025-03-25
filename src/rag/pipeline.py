"""
RAG (Retrieval-Augmented Generation) Pipeline.

This module provides a full RAG pipeline that combines document processing,
retrieval, and generation components.
"""

import logging
import os
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

from src.processor.document_processor import DocumentProcessor
from src.rag.retriever import Retriever
from src.rag.generator import Generator
from src.vectorstore.base import BaseVectorStore
from src.vectorstore.vector_store_factory import VectorStoreFactory

logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    RAG (Retrieval-Augmented Generation) Pipeline.
    
    This class combines document processing, retrieval, and generation components
    to provide a complete RAG pipeline.
    """
    
    def __init__(self, 
                processor: Optional[DocumentProcessor] = None,
                retriever: Optional[Retriever] = None,
                generator: Optional[Generator] = None,
                vector_store: Optional[BaseVectorStore] = None,
                vector_store_type: str = "chromadb",
                vector_store_args: Optional[Dict[str, Any]] = None,
                embedding_function: Optional[Callable] = None,
                embedding_args: Optional[Dict[str, Any]] = None,
                chunking_strategy: str = "fixed_size",
                chunking_args: Optional[Dict[str, Any]] = None,
                retrieval_top_k: int = 4,
                retrieval_score_threshold: Optional[float] = None,
                model_provider: str = "openai",
                model_name: str = "gpt-3.5-turbo",
                model_args: Optional[Dict[str, Any]] = None):
        """
        Initialize the RAG pipeline.
        
        Args:
            processor: Optional pre-configured document processor
            retriever: Optional pre-configured retriever
            generator: Optional pre-configured generator
            vector_store: Optional pre-configured vector store
            vector_store_type: Type of vector store to use if not provided
            vector_store_args: Arguments for vector store initialization
            embedding_function: Optional embedding function
            embedding_args: Arguments for embedding function initialization
            chunking_strategy: Strategy to use for chunking documents
            chunking_args: Arguments for chunking strategy
            retrieval_top_k: Number of documents to retrieve
            retrieval_score_threshold: Optional threshold for similarity scores
            model_provider: Provider of the language model
            model_name: Name of the language model to use
            model_args: Arguments for language model initialization
        """
        # Initialize vector store if not provided
        if vector_store is None and processor is None:
            vector_store_args = vector_store_args or {}
            
            # Add embedding function to vector store args if provided
            if embedding_function is not None and "embedding_function" not in vector_store_args:
                vector_store_args["embedding_function"] = embedding_function
                
            vector_store = VectorStoreFactory.get_vector_store(
                store_type=vector_store_type,
                **vector_store_args
            )
        
        # Initialize processor if not provided
        if processor is None:
            processor_args = {
                "vector_store": vector_store,
                "chunking_strategy": chunking_strategy,
                "chunking_args": chunking_args or {},
                "embedding_function": embedding_function,
                "embedding_args": embedding_args
            }
            
            processor = DocumentProcessor(**processor_args)
        
        # Use processor's vector store if vector_store is not provided
        if vector_store is None:
            vector_store = processor.vector_store
        
        # Initialize retriever if not provided
        if retriever is None:
            retriever = Retriever(
                vector_store=vector_store,
                top_k=retrieval_top_k,
                score_threshold=retrieval_score_threshold
            )
        
        # Initialize generator if not provided
        if generator is None:
            generator_args = {
                "model_provider": model_provider,
                "model_name": model_name,
                **(model_args or {})
            }
            
            generator = Generator(**generator_args)
        
        # Set components
        self.processor = processor
        self.retriever = retriever
        self.generator = generator
        
        logger.info("Initialized RAG pipeline")
    
    def process_document(self, 
                        document: Union[str, Dict[str, Any]], 
                        document_type: str = "text",
                        metadata: Optional[Dict[str, Any]] = None,
                        **kwargs) -> List[str]:
        """
        Process a document for the RAG pipeline.
        
        Args:
            document: Document to process (text or file path)
            document_type: Type of document (text, file, or json)
            metadata: Optional metadata for the document
            **kwargs: Additional arguments for the processor
            
        Returns:
            List of chunk IDs stored in the vector database
        """
        if document_type == "text":
            return self.processor.process_text(document, metadata, **kwargs)
        elif document_type == "file":
            return self.processor.process_file(document, metadata, **kwargs)
        elif document_type == "json":
            return self.processor.process_json(document, **kwargs)
        else:
            raise ValueError(f"Unsupported document type: {document_type}")
    
    def process_documents(self, 
                         documents: List[Union[str, Dict[str, Any]]],
                         document_type: str = "text",
                         common_metadata: Optional[Dict[str, Any]] = None,
                         **kwargs) -> Dict[str, List[str]]:
        """
        Process multiple documents for the RAG pipeline.
        
        Args:
            documents: List of documents to process
            document_type: Type of documents (text, file, or json)
            common_metadata: Metadata to apply to all documents
            **kwargs: Additional arguments for the processor
            
        Returns:
            Dictionary mapping document identifiers to lists of chunk IDs
        """
        results = {}
        
        if document_type == "file":
            # For files, we can use the bulk processor
            return self.processor.bulk_process_files(
                file_paths=documents,
                common_metadata=common_metadata,
                **kwargs
            )
        else:
            # For text and JSON, process each document individually
            for i, doc in enumerate(documents):
                # Create a unique identifier for each document
                doc_id = f"doc_{i+1}"
                
                # Prepare document-specific metadata
                doc_metadata = common_metadata.copy() if common_metadata else {}
                doc_metadata["doc_id"] = doc_id
                
                if document_type == "text":
                    chunk_ids = self.processor.process_text(
                        text=doc,
                        metadata=doc_metadata,
                        **kwargs
                    )
                elif document_type == "json":
                    chunk_ids = self.processor.process_json(
                        json_data=doc,
                        **kwargs
                    )
                else:
                    # Should never reach here due to earlier check
                    continue
                
                results[doc_id] = chunk_ids
        
        return results
    
    def query(self, 
             query: str,
             filter: Optional[Dict[str, Any]] = None,
             top_k: Optional[int] = None,
             use_mmr: bool = False,
             mmr_lambda: float = 0.5,
             **kwargs) -> Dict[str, Any]:
        """
        Query the RAG pipeline.
        
        Args:
            query: User query
            filter: Optional filter for document retrieval
            top_k: Optional override for the number of documents to retrieve
            use_mmr: Whether to use Maximum Marginal Relevance for retrieval
            mmr_lambda: Lambda parameter for MMR
            **kwargs: Additional arguments for the generator
            
        Returns:
            Dictionary containing the response and metadata
        """
        start_time = time.time()
        
        # Retrieve relevant documents
        if use_mmr:
            retrieved_docs = self.retriever.retrieve_with_mmr(
                query=query,
                filter=filter,
                top_k=top_k,
                lambda_mult=mmr_lambda
            )
        else:
            retrieved_docs = self.retriever.retrieve(
                query=query,
                filter=filter,
                top_k=top_k
            )
        
        retrieval_time = time.time() - start_time
        
        # Generate response
        generation_start_time = time.time()
        result = self.generator.generate(query, retrieved_docs)
        generation_time = time.time() - generation_start_time
        
        # Add timing and retrieved documents to metadata
        result["metadata"]["timing"] = {
            "retrieval": retrieval_time,
            "generation": generation_time,
            "total": time.time() - start_time
        }
        
        result["metadata"]["retrieval"] = {
            "num_docs": len(retrieved_docs),
            "docs": retrieved_docs
        }
        
        return result
    
    def process_and_query(self, 
                         document: Union[str, Dict[str, Any]],
                         query: str,
                         document_type: str = "text",
                         metadata: Optional[Dict[str, Any]] = None,
                         **kwargs) -> Dict[str, Any]:
        """
        Process a document and then query it in a single operation.
        
        This is useful for ad-hoc querying of documents not stored in the vector database.
        
        Args:
            document: Document to process
            query: User query
            document_type: Type of document (text, file, or json)
            metadata: Optional metadata for the document
            **kwargs: Additional arguments for processing and querying
            
        Returns:
            Dictionary containing the response and metadata
        """
        # Process the document
        chunk_ids = self.process_document(
            document=document,
            document_type=document_type,
            metadata=metadata
        )
        
        if not chunk_ids:
            logger.warning("No chunks created from document, generating response without context")
            return self.generator.generate_without_context(query)
        
        # Create a filter to only search the document we just processed
        if metadata and "doc_id" in metadata:
            filter = {"doc_id": metadata["doc_id"]}
        else:
            # No specific doc_id, use chunk_ids
            # Note: This filtering approach depends on the vector store implementation
            filter = None
        
        # Query the pipeline
        return self.query(
            query=query,
            filter=filter,
            **kwargs
        )
    
    def get_vector_store(self) -> BaseVectorStore:
        """Get the vector store used by the pipeline."""
        return self.retriever.vector_store
    
    def save_pipeline_config(self, file_path: str) -> None:
        """
        Save the pipeline configuration to a file.
        
        Args:
            file_path: Path to save the configuration file
        """
        # Create a simplified configuration that doesn't include objects
        config = {
            "chunking_strategy": self.processor.chunking_strategy,
            "chunking_args": self.processor.chunking_args,
            "retrieval_top_k": self.retriever.top_k,
            "retrieval_score_threshold": self.retriever.score_threshold,
            "model_provider": self.generator.model_provider,
            "model_name": self.generator.model_name,
            "model_temperature": self.generator.temperature,
            "model_max_tokens": self.generator.max_tokens,
            "system_prompt": self.generator.system_prompt,
            "timestamp": time.time()
        }
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        logger.info(f"Saved pipeline configuration to {file_path}")
    
    @classmethod
    def load_from_config(cls, 
                        config_file: str, 
                        vector_store: Optional[BaseVectorStore] = None,
                        vector_store_type: str = "chromadb",
                        vector_store_args: Optional[Dict[str, Any]] = None) -> 'RAGPipeline':
        """
        Load a pipeline from a configuration file.
        
        Args:
            config_file: Path to the configuration file
            vector_store: Optional pre-configured vector store
            vector_store_type: Type of vector store to use if not provided
            vector_store_args: Arguments for vector store initialization
            
        Returns:
            RAGPipeline instance
        """
        # Load config from file
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Create pipeline with loaded config
        return cls(
            vector_store=vector_store,
            vector_store_type=vector_store_type,
            vector_store_args=vector_store_args,
            chunking_strategy=config.get("chunking_strategy", "fixed_size"),
            chunking_args=config.get("chunking_args", {}),
            retrieval_top_k=config.get("retrieval_top_k", 4),
            retrieval_score_threshold=config.get("retrieval_score_threshold"),
            model_provider=config.get("model_provider", "openai"),
            model_name=config.get("model_name", "gpt-3.5-turbo"),
            model_args={
                "temperature": config.get("model_temperature", 0.7),
                "max_tokens": config.get("model_max_tokens", 1000),
                "system_prompt": config.get("system_prompt")
            }
        ) 
"""
Document processor for ingesting and processing documents for the RAG pipeline.

This module provides functionality to process documents, chunk them, and store them
in a vector database for later retrieval.
"""

import logging
import os
import json
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

from src.chunking.chunker_factory import ChunkerFactory
from src.vectorstore.base import BaseVectorStore
from src.vectorstore.vector_store_factory import VectorStoreFactory

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Process documents for ingestion into the RAG pipeline.
    
    This class handles document loading, chunking, and storage in the vector database.
    """
    
    def __init__(self, 
                vector_store: Optional[BaseVectorStore] = None,
                vector_store_type: str = "chromadb",
                vector_store_args: Optional[Dict[str, Any]] = None,
                chunking_strategy: str = "fixed_size",
                chunking_args: Optional[Dict[str, Any]] = None,
                embedding_function: Optional[Callable] = None,
                embedding_args: Optional[Dict[str, Any]] = None):
        """
        Initialize the document processor.
        
        Args:
            vector_store: Optional pre-configured vector store
            vector_store_type: Type of vector store to use if not provided
            vector_store_args: Arguments for vector store initialization
            chunking_strategy: Strategy to use for chunking documents
            chunking_args: Arguments for the chunking strategy
            embedding_function: Optional embedding function
            embedding_args: Arguments for creating an embedding function if not provided
        """
        self.chunking_strategy = chunking_strategy
        self.chunking_args = chunking_args or {}
        
        # Map frontend parameter names to chunker parameter names if needed
        mapped_chunking_args = self._map_chunking_args(chunking_strategy, self.chunking_args)
        
        # Set up chunker
        self.chunker = ChunkerFactory.get_chunker(
            strategy_name=chunking_strategy,
            **mapped_chunking_args
        )
        
        # Set up embedding function if not provided
        if embedding_function:
            self.embedding_function = embedding_function
        else:
            self._setup_embedding_function(embedding_args or {})
        
        # Set up vector store
        if vector_store:
            self.vector_store = vector_store
        else:
            vector_store_args = vector_store_args or {}
            
            # Add embedding function to vector store args if not explicitly provided
            if "embedding_function" not in vector_store_args and hasattr(self, "embedding_function"):
                vector_store_args["embedding_function"] = self.embedding_function
                
            self.vector_store = VectorStoreFactory.get_vector_store(
                store_type=vector_store_type,
                **vector_store_args
            )
        
        logger.info(f"Initialized DocumentProcessor with chunking_strategy='{chunking_strategy}'")
    
    def _map_chunking_args(self, strategy: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Map frontend parameter names to chunker parameter names."""
        # Create a copy of args to avoid modifying the original
        mapped_args = args.copy()
        
        # Map parameters for each chunking strategy
        if strategy == "sliding_window":
            # For sliding window, handle window_size and slide_amount
            if "chunk_size" in mapped_args:
                mapped_args["window_size"] = mapped_args.pop("chunk_size")
            if "chunk_overlap" in mapped_args:
                mapped_args["slide_amount"] = mapped_args.pop("chunk_overlap")
        elif strategy == "semantic":
            # For semantic, handle min_size and max_size
            if "min_size" in mapped_args:
                mapped_args["min_chunk_size"] = mapped_args.pop("min_size")
            if "max_size" in mapped_args:
                mapped_args["max_chunk_size"] = mapped_args.pop("max_size")
                
        return mapped_args
    
    def _setup_embedding_function(self, embedding_args: Dict[str, Any]) -> None:
        """Set up the embedding function based on provided args."""
        # Default to OpenAI embeddings if not specified
        embedding_type = embedding_args.pop("type", "openai")
        
        try:
            if embedding_type == "openai":
                self.embedding_function = VectorStoreFactory.create_openai_embedding_function(
                    **embedding_args
                )
            elif embedding_type == "huggingface":
                self.embedding_function = VectorStoreFactory.create_huggingface_embedding_function(
                    **embedding_args
                )
            else:
                logger.warning(f"Unknown embedding type: {embedding_type}. Using OpenAI embeddings.")
                self.embedding_function = VectorStoreFactory.create_openai_embedding_function(
                    **embedding_args
                )
        except Exception as e:
            logger.error(f"Failed to initialize embedding function: {str(e)}")
            logger.warning("Proceeding without custom embedding function. Vector store may use its default.")
    
    def process_text(self, 
                    text: str, 
                    metadata: Optional[Dict[str, Any]] = None, 
                    doc_id: Optional[str] = None,
                    chunk_with_metadata: bool = True) -> List[str]:
        """
        Process a text document for the RAG pipeline.
        
        Args:
            text: The text content to process
            metadata: Optional metadata associated with the document
            doc_id: Optional document ID
            chunk_with_metadata: Whether to include metadata with chunks
            
        Returns:
            List of chunk IDs stored in the vector database
        """
        if not text:
            logger.warning("Empty text provided to process_text")
            return []
        
        # Generate document ID if not provided
        if doc_id is None:
            doc_id = str(uuid.uuid4())
        
        # Prepare metadata
        metadata = metadata or {}
        metadata["doc_id"] = doc_id
        
        # Create chunks from the text
        chunks = self.chunker.create_chunks(text, metadata)
        
        if not chunks:
            logger.warning(f"No chunks created from document {doc_id}")
            return []
        
        logger.info(f"Created {len(chunks)} chunks from document {doc_id}")
        
        # Prepare chunks for vector store
        texts = [chunk["text"] for chunk in chunks]
        chunk_metadatas = [chunk["metadata"] for chunk in chunks] if chunk_with_metadata else None
        
        # Store chunks in vector database
        chunk_ids = self.vector_store.add_texts(
            texts=texts,
            metadatas=chunk_metadatas
        )
        
        logger.info(f"Stored {len(chunk_ids)} chunks in vector store for document {doc_id}")
        return chunk_ids
    
    def process_file(self, 
                    file_path: str, 
                    metadata: Optional[Dict[str, Any]] = None,
                    doc_id: Optional[str] = None,
                    chunk_with_metadata: bool = True) -> List[str]:
        """
        Process a file for the RAG pipeline.
        
        Args:
            file_path: Path to the file to process
            metadata: Optional metadata associated with the document
            doc_id: Optional document ID
            chunk_with_metadata: Whether to include metadata with chunks
            
        Returns:
            List of chunk IDs stored in the vector database
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return []
        
        # Prepare metadata
        metadata = metadata or {}
        metadata["file_path"] = file_path
        metadata["file_name"] = os.path.basename(file_path)
        
        # Generate document ID from filename if not provided
        if doc_id is None:
            doc_id = str(uuid.uuid4())
        
        # Check file extension to determine how to read it
        file_extension = os.path.splitext(file_path)[1].lower()
        
        # Process based on file type
        try:
            # For PDF files, use PyPDF2 to extract text
            if file_extension in ['.pdf', '.PDF']:
                try:
                    import PyPDF2
                    with open(file_path, 'rb') as pdf_file:
                        pdf_reader = PyPDF2.PdfReader(pdf_file)
                        text = ""
                        for page_num in range(len(pdf_reader.pages)):
                            page = pdf_reader.pages[page_num]
                            page_text = page.extract_text() or ""
                            text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                    
                    # Convert single newlines to double for better paragraph detection
                    # First check if there are any double newlines already
                    if text.count("\n\n") < 10:  # If very few double newlines
                        logger.info(f"Converting single newlines to double newlines for {file_path}")
                        text = text.replace("\n", "\n\n")
                        
                except ImportError:
                    logger.error("PyPDF2 is not installed. Please install it with 'pip install PyPDF2'")
                    return []
                except Exception as pdf_error:
                    logger.error(f"Error extracting text from PDF {file_path}: {str(pdf_error)}")
                    return []
            # For text files, read as UTF-8 text
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
            # Process the text content
            return self.process_text(
                text=text,
                metadata=metadata,
                doc_id=doc_id,
                chunk_with_metadata=chunk_with_metadata
            )
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return []
    
    def process_json(self, 
                    json_data: Union[str, Dict[str, Any]], 
                    text_key: str = "text",
                    metadata_keys: Optional[List[str]] = None,
                    doc_id: Optional[str] = None,
                    chunk_with_metadata: bool = True) -> List[str]:
        """
        Process a JSON document for the RAG pipeline.
        
        Args:
            json_data: JSON data (string or dict) to process
            text_key: Key to extract text content from
            metadata_keys: Keys to extract metadata from
            doc_id: Optional document ID
            chunk_with_metadata: Whether to include metadata with chunks
            
        Returns:
            List of chunk IDs stored in the vector database
        """
        # Parse JSON if string
        if isinstance(json_data, str):
            try:
                json_data = json.loads(json_data)
            except Exception as e:
                logger.error(f"Error parsing JSON: {str(e)}")
                return []
        
        # Extract text
        if text_key not in json_data:
            logger.error(f"Text key '{text_key}' not found in JSON data")
            return []
        
        text = json_data[text_key]
        
        # Extract metadata
        metadata = {}
        if metadata_keys:
            for key in metadata_keys:
                if key in json_data:
                    metadata[key] = json_data[key]
        
        # Process the text content
        return self.process_text(
            text=text,
            metadata=metadata,
            doc_id=doc_id,
            chunk_with_metadata=chunk_with_metadata
        )
    
    def bulk_process_files(self, 
                          file_paths: List[str], 
                          common_metadata: Optional[Dict[str, Any]] = None,
                          chunk_with_metadata: bool = True) -> Dict[str, List[str]]:
        """
        Process multiple files in bulk.
        
        Args:
            file_paths: List of file paths to process
            common_metadata: Metadata to apply to all files
            chunk_with_metadata: Whether to include metadata with chunks
            
        Returns:
            Dictionary mapping file paths to lists of chunk IDs
        """
        results = {}
        
        for file_path in file_paths:
            # Generate document ID
            doc_id = str(uuid.uuid4())
            
            # Prepare metadata with common and file-specific data
            metadata = {
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "doc_id": doc_id
            }
            
            if common_metadata:
                metadata.update(common_metadata)
            
            # Process the file
            chunk_ids = self.process_file(
                file_path=file_path,
                metadata=metadata,
                doc_id=doc_id,
                chunk_with_metadata=chunk_with_metadata
            )
            
            results[file_path] = chunk_ids
        
        return results
    
    def query_vector_store(self, 
                          query: str, 
                          k: int = 4, 
                          filter: Optional[Dict[str, Any]] = None) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        Query the vector store for similar documents.
        
        Args:
            query: The query text
            k: Number of results to return
            filter: Optional filter for the query
            
        Returns:
            List of tuples containing (text, metadata, similarity score)
        """
        return self.vector_store.similarity_search(query, k, filter) 
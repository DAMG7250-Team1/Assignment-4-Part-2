from src.rag.methods.naive_rag import Response, Source
import os
import logging
from typing import List, Dict, Any, Optional
import tempfile

from src.vectorstore.vector_store_factory import VectorStoreFactory
from src.rag.retriever import Retriever
from src.rag.generator import Generator
from src.rag.pipeline import RAGPipeline
from src.chunking.strategies.fixed_size import FixedSizeChunker
from src.chunking.strategies.semantic import SemanticChunker
from src.chunking.strategies.sliding_window import SlidingWindowChunker

logger = logging.getLogger(__name__)

class ChromaRAG:
    """Local vector store with ChromaDB."""
    
    def __init__(self, 
                chunking_strategy: str = "fixed_size",
                persistent_directory: Optional[str] = None,
                collection_name: str = "nvidia-reports",
                model_provider: str = "google",
                model_name: str = "gemini-flash",
                api_key: Optional[str] = None):
        """
        Initialize the Chroma RAG method.
        
        Args:
            chunking_strategy: Strategy to use for chunking (fixed_size, sliding_window, semantic)
            persistent_directory: Directory for ChromaDB persistence (optional, uses in-memory if None)
            collection_name: Name of the ChromaDB collection
            model_provider: LLM provider for generation
            model_name: Model name for generation
            api_key: API key for the language model (if not provided, will use environment variable)
                    Note: This is for the LLM generator, ChromaDB doesn't require an API key
        """
        self.chunking_strategy = chunking_strategy
        self.collection_name = collection_name
        self.persistent_directory = persistent_directory
        self.model_provider = model_provider
        self.model_name = model_name
        self.api_key = api_key
        
        # Initialize ChromaDB vector store
        vector_store_args = {
            "collection_name": collection_name
        }
        
        if persistent_directory:
            vector_store_args["persist_directory"] = persistent_directory
        
        self.vector_store = VectorStoreFactory.get_vector_store(
            store_type="chromadb",
            **vector_store_args
        )
        
        # Create appropriate chunker based on strategy
        if chunking_strategy == "fixed_size":
            self.chunker = FixedSizeChunker(chunk_size=1000)
        elif chunking_strategy == "sliding_window":
            self.chunker = SlidingWindowChunker(chunk_size=1000, chunk_overlap=200)
        elif chunking_strategy == "semantic":
            self.chunker = SemanticChunker()
        else:
            raise ValueError(f"Unknown chunking strategy: {chunking_strategy}")
        
        # Create retriever
        self.retriever = Retriever(vector_store=self.vector_store)
        
        # Create generator
        self.generator = Generator(
            model_provider=self.model_provider,
            model_name=self.model_name,
            temperature=0.2,
            api_key=self.api_key
        )
        
        # Create RAG pipeline
        self.pipeline = RAGPipeline(
            vector_store=self.vector_store,
            retriever=self.retriever,
            generator=self.generator
        )
        
        logger.info(f"Initialized ChromaRAG with chunking strategy '{chunking_strategy}'")
    
    def add_document(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Add a document to ChromaDB.
        
        Args:
            text: Document text
            metadata: Optional metadata
            
        Returns:
            List of chunk IDs
        """
        # Create chunks
        chunks = self.chunker.create_chunks(text)
        
        if not chunks:
            logger.warning("No chunks created from document")
            return []
        
        # Add document chunks to vector store
        metadata_list = [metadata or {} for _ in chunks]
        ids = self.vector_store.add_documents(chunks, metadata=metadata_list)
        
        logger.info(f"Added document with {len(chunks)} chunks to ChromaDB")
        return ids
    
    def add_file(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Add a document from a file to ChromaDB.
        
        Args:
            file_path: Path to the file
            metadata: Optional metadata
            
        Returns:
            List of chunk IDs
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                
            # Add file name to metadata if not already present
            if metadata is None:
                metadata = {}
            
            if 'source' not in metadata:
                metadata['source'] = os.path.basename(file_path)
                
            return self.add_document(text, metadata)
            
        except Exception as e:
            logger.error(f"Error adding file to ChromaDB: {str(e)}")
            return []
        
    def process_query(self, query: str) -> Response:
        """
        Process a query using ChromaDB vector store.
        
        Args:
            query: The user's question
            
        Returns:
            Response object containing the answer and sources
        """
        try:
            # Retrieve relevant documents
            retrieved_docs = self.retriever.retrieve(query, top_k=3)
            
            if not retrieved_docs:
                return Response(
                    answer="I couldn't find any relevant information in the available documents.",
                    sources=[]
                )
            
            # Generate response
            response_dict = self.generator.generate(query, retrieved_docs)
            answer = response_dict.get("response", "No response generated")
            
            # Format sources
            sources = []
            for i, doc in enumerate(retrieved_docs):
                doc_id = doc.get("id", f"doc_{i}")
                text = doc.get("text", "")
                score = doc.get("score", 0.0)
                metadata = doc.get("metadata", {})
                document_name = metadata.get("source", "Unknown source")
                
                source = Source(
                    document=document_name,
                    context=text,
                    score=score
                )
                sources.append(source)
            
            return Response(
                answer=answer,
                sources=sources
            )
        
        except Exception as e:
            logger.error(f"Error processing query with ChromaRAG: {str(e)}")
            return Response(
                answer=f"An error occurred while processing your query: {str(e)}",
                sources=[]
            )

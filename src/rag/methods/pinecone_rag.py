from src.rag.methods.naive_rag import Response, Source
import os
import logging
from typing import List, Dict, Any, Optional

from src.vectorstore.pinecone_store import PineconeVectorStore
from src.rag.retriever import Retriever
from src.rag.generator import Generator
from src.rag.pipeline import RAGPipeline
from src.chunking.strategies.fixed_size import FixedSizeChunker
from src.chunking.strategies.semantic import SemanticChunker
from src.chunking.strategies.sliding_window import SlidingWindowChunker

logger = logging.getLogger(__name__)

class PineconeRAG:
    """Vector similarity search with Pinecone."""
    
    def __init__(self, 
                chunking_strategy: str = "fixed_size",
                index_name: str = "nvidia-reports",
                cloud: str = "aws", 
                region: str = "us-east-1",
                model_provider: str = "google",
                model_name: str = "gemini-flash",
                api_key: Optional[str] = None):
        """
        Initialize the Pinecone RAG method.
        
        Args:
            chunking_strategy: Strategy to use for chunking (fixed_size, sliding_window, semantic)
            index_name: Name of the Pinecone index
            cloud: Cloud provider for Pinecone
            region: Region for Pinecone
            model_provider: LLM provider for generation
            model_name: Model name for generation
            api_key: API key for the language model (if not provided, will use environment variable)
                    Note: This is NOT the Pinecone API key, which is taken from environment variables
        """
        self.chunking_strategy = chunking_strategy
        self.index_name = index_name
        self.cloud = cloud
        self.region = region
        self.model_provider = model_provider
        self.model_name = model_name
        self.api_key = api_key
        
        # Initialize Pinecone vector store
        self.vector_store = PineconeVectorStore(
            index_name=self.index_name,
            cloud=self.cloud,
            region=self.region
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
        
        logger.info(f"Initialized PineconeRAG with chunking strategy '{chunking_strategy}'")
        
    def process_query(self, query: str) -> Response:
        """
        Process a query using Pinecone vector search.
        
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
            logger.error(f"Error processing query with PineconeRAG: {str(e)}")
            return Response(
                answer=f"An error occurred while processing your query: {str(e)}",
                sources=[]
            )

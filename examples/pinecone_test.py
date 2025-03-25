#!/usr/bin/env python3
"""
Test script for Pinecone integration with NVIDIA RAG pipeline.
This script demonstrates how to use the Pinecone vector store with local embeddings
and Gemini Flash for response generation.
"""

import os
import sys
import logging
from dotenv import load_dotenv
import time

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Explicitly set the Google API key from the environment variable
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY", "")
if not os.environ["GOOGLE_API_KEY"]:
    logger.error("GEMINI_API_KEY environment variable not set")
    sys.exit(1)
else:
    logger.info(f"Using GEMINI_API_KEY (first 10 chars): {os.environ['GOOGLE_API_KEY'][:10]}...")

# Import project modules
from src.vectorstore.pinecone_store import PineconeVectorStore
from src.chunking.strategies.fixed_size import FixedSizeChunker
from src.rag.retriever import Retriever
from src.rag.generator import Generator
from src.rag.pipeline import RAGPipeline

def main():
    """Main function to test Pinecone integration with Gemini Flash."""
    logger.info("Starting Pinecone integration test with Gemini Flash")
    
    # Create a Pinecone vector store
    vector_store = PineconeVectorStore(
        index_name="nvidia-test",
        cloud="aws",
        region="us-east-1"
    )
    
    # Create a chunker
    chunker = FixedSizeChunker(chunk_size=1000)
    
    # Create a retriever
    retriever = Retriever(vector_store=vector_store)
    
    # Create a generator explicitly using Gemini Flash
    generator = Generator(
        model_provider="google",
        model_name="gemini-2.0-flash",
        temperature=0.2,
        client_args={"api_key": os.environ["GOOGLE_API_KEY"]}
    )
    
    # Create a RAG pipeline
    pipeline = RAGPipeline(
        vector_store=vector_store,
        retriever=retriever,
        generator=generator
    )
    
    # Test with some example NVIDIA data
    nvidia_text = """
    NVIDIA (NASDAQ: NVDA) today reported revenue for the second quarter ended July 30, 2023, of $13.51 billion, 
    up 101% from a year ago and up 88% from the previous quarter.
    
    GAAP earnings per diluted share for the quarter were $2.48, up 854% from a year ago and up 202% from the 
    previous quarter. Non-GAAP earnings per diluted share were $2.70, up 429% from a year ago and up 148% 
    from the previous quarter.
    
    "A new computing era has begun. Companies worldwide are transitioning from general-purpose to accelerated 
    computing and generative AI," said Jensen Huang, founder and CEO of NVIDIA.
    
    "NVIDIA GPUs connected by our Mellanox networking and switch technologies enable enterprises to build 
    generative AI factories that process and refine data into valuable intelligence," he said. "The race 
    is on among enterprises to transform their businesses with generative AI. The world's most valuable 
    companies are racing to deploy NVIDIA H100 into production."
    
    In Gaming, revenue of $2.49 billion was up 22% from a year ago and up 11% from the previous quarter, 
    driven by sales of NVIDIA GeForce RTX® 40 Series GPUs for laptops and desktops.
    
    In Data Center, revenue of $10.32 billion was up 171% from a year ago and up 141% from the previous 
    quarter. The year-on-year growth was driven by the ramp of the NVIDIA Hopper™ architecture. Strong 
    growth was driven by leadership in both training and inference for generative AI applications.
    """
    
    # Manually create chunks for testing
    chunks = [
        "NVIDIA (NASDAQ: NVDA) today reported revenue for the second quarter ended July 30, 2023, of $13.51 billion, up 101% from a year ago and up 88% from the previous quarter.",
        "In Gaming, revenue of $2.49 billion was up 22% from a year ago and up 11% from the previous quarter, driven by sales of NVIDIA GeForce RTX® 40 Series GPUs for laptops and desktops.",
        "In Data Center, revenue of $10.32 billion was up 171% from a year ago and up 141% from the previous quarter. The year-on-year growth was driven by the ramp of the NVIDIA Hopper™ architecture.",
        "A new computing era has begun. Companies worldwide are transitioning from general-purpose to accelerated computing and generative AI."
    ]
    logger.info(f"Created {len(chunks)} chunks from the NVIDIA text")
    
    # Add chunks to vector store with metadata
    metadata_list = [{"source": "NVIDIA Q2 2023 Report", "document_id": "nvidia_q2_2023"} for _ in chunks]
    vector_store.add_documents(chunks, metadata=metadata_list)
    logger.info("Added chunks to Pinecone vector store")
    
    # Give Pinecone a moment to index the documents
    logger.info("Waiting for Pinecone to index documents...")
    time.sleep(3)
    
    # Test a query using the retriever and generator directly
    query = "What was NVIDIA's revenue in Q2 2023?"
    logger.info(f"Querying: '{query}'")
    
    # Retrieve relevant chunks
    try:
        # Try standard retrieval first
        relevant_chunks = retriever.retrieve(query, top_k=3)
        logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks")
        
        # If no results, get documents directly
        if not relevant_chunks:
            logger.info("No results from retriever, falling back to direct document access")
            # Get all documents for testing purposes
            zero_vector = [0.0] * vector_store.dimension
            direct_results = vector_store.index.query(
                vector=zero_vector,
                top_k=3,
                include_metadata=True
            )
            
            # Convert to the expected format
            for match in direct_results.matches:
                relevant_chunks.append({
                    "id": match.id,
                    "text": match.metadata.get("text", ""),
                    "metadata": match.metadata,
                    "score": 1.0  # Placeholder score
                })
            
            logger.info(f"Retrieved {len(relevant_chunks)} chunks directly")
    except Exception as e:
        logger.error(f"Error during retrieval: {str(e)}")
        relevant_chunks = []
    
    # Generate response with Gemini Flash
    response = generator.generate_response(query, relevant_chunks)
    
    # Print the response
    logger.info(f"Gemini Flash Response: {response}")
    
    logger.info("Pinecone integration test completed")

if __name__ == "__main__":
    main()
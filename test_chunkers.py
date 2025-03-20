#!/usr/bin/env python3
"""
Test script to evaluate different chunking strategies in the NVIDIA RAG pipeline.
This simplified version uses PyPDF2 and ChromaDB for all tests.
"""

import os
import sys
import logging
import time
from pathlib import Path
from dotenv import load_dotenv
import json
import PyPDF2
import io

# Load environment variables from .env file
load_dotenv()

# Add the src directory to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import components
from src.chunking.chunker_factory import ChunkerFactory
from src.vectorstore.vector_store_factory import VectorStoreFactory
from src.rag.pipeline import RAGPipeline

def extract_pdf_text(file_path):
    """Extract text from a PDF file using PyPDF2."""
    try:
        with open(file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text() or ""
                text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                
            # Convert single newlines to double for better paragraph detection
            if text.count("\n\n") < 10:  # If very few double newlines
                text = text.replace("\n", "\n\n")
                
            return text
            
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        return None

def setup_rag_pipeline(chunking_strategy, persist_dir):
    """Set up a RAG pipeline with the specified chunking strategy."""
    
    # Configure chunking arguments based on strategy
    chunking_args = {}
    if chunking_strategy == "fixed_size":
        chunking_args = {
            "chunk_size": 1000,
            "chunk_overlap": 200
        }
    elif chunking_strategy == "sliding_window":
        chunking_args = {
            "window_size": 1000,
            "slide_amount": 200
        }
    elif chunking_strategy == "semantic":
        chunking_args = {
            "max_chunk_size": 1500,
            "min_chunk_size": 500
        }
    
    # Create Google embedding function
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY", "")
    embedding_function = VectorStoreFactory.create_google_embedding_function()
    
    # Create RAG pipeline
    pipeline = RAGPipeline(
        vector_store_type="chromadb",
        vector_store_args={
            "persist_directory": persist_dir,
            "embedding_function": embedding_function
        },
        chunking_strategy=chunking_strategy,
        chunking_args=chunking_args,
        model_provider="google",
        model_name="gemini-2.0-flash",
        model_args={
            "temperature": 0.2,
            "system_prompt": """You are a financial analyst assistant specializing in NVIDIA. 
            Answer questions accurately based on the provided context.
            Cite specific data points when relevant."""
        }
    )
    
    return pipeline

def evaluate_chunker(chunking_strategy, test_file, test_query):
    """
    Evaluate a specific chunking strategy.
    
    Args:
        chunking_strategy: Name of the chunking strategy
        test_file: PDF file to process
        test_query: Query to test with
        
    Returns:
        Dictionary with evaluation results
    """
    try:
        start_time = time.time()
        
        # Create unique directory for this test
        test_dir = f"data/test_{chunking_strategy}_{int(time.time())}"
        os.makedirs(test_dir, exist_ok=True)
        
        logger.info(f"Testing chunking strategy: {chunking_strategy}")
        
        # Step 1: Extract text from PDF
        extract_start = time.time()
        text = extract_pdf_text(test_file)
        extract_time = time.time() - extract_start
        
        if not text:
            return {
                "success": False,
                "error": "Failed to extract text from PDF",
                "chunker": chunking_strategy
            }
            
        # Save extracted text for reference
        with open(f"{test_dir}/extracted_text.txt", "w") as f:
            f.write(text)
        
        # Step 2: Set up RAG pipeline with specified chunker
        pipeline_start = time.time()
        pipeline = setup_rag_pipeline(chunking_strategy, f"{test_dir}/vectorstore")
        setup_time = time.time() - pipeline_start
        
        # Step 3: Process text
        process_start = time.time()
        chunk_ids = pipeline.processor.process_text(
            text=text,
            metadata={"file_name": os.path.basename(test_file)}
        )
        process_time = time.time() - process_start
        num_chunks = len(chunk_ids)
        
        # Step 4: Test query
        query_start = time.time()
        result = pipeline.query(test_query)
        query_time = time.time() - query_start
        response = result.get("response", "No response generated")
        
        total_time = time.time() - start_time
        
        # Store result details
        result_data = {
            "success": True,
            "chunker": chunking_strategy,
            "num_chunks": num_chunks,
            "timing": {
                "extract": extract_time,
                "setup": setup_time,
                "process": process_time,
                "query": query_time,
                "total": total_time
            },
            "query": test_query,
            "response": response
        }
        
        # Save result to file
        with open(f"{test_dir}/result.json", "w") as f:
            json.dump(result_data, f, indent=2)
            
        logger.info(f"Evaluation complete for {chunking_strategy}")
        logger.info(f"Created {num_chunks} chunks and processed query in {query_time:.2f} seconds")
        
        return result_data
        
    except Exception as e:
        logger.error(f"Error evaluating chunker: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "chunker": chunking_strategy
        }

def main():
    # Test file
    test_file = "data/reports/NVIDIA_2025_Third_Quarter_2025.pdf"
    if not os.path.exists(test_file):
        logger.error(f"Test file not found: {test_file}")
        return
        
    # Test query
    test_query = "What was NVIDIA's revenue in the most recent quarter?"
    
    # Available chunking strategies to test
    chunkers = ["fixed_size", "sliding_window", "semantic"]
    
    # Track results
    results = []
    
    # Test all chunking strategies
    for chunking_strategy in chunkers:
        result = evaluate_chunker(
            chunking_strategy,
            test_file,
            test_query
        )
        results.append(result)
        
        # Add a small delay between tests
        time.sleep(1)
    
    # Summarize results
    successful = [r for r in results if r.get("success", False)]
    failed = [r for r in results if not r.get("success", False)]
    
    logger.info(f"Completed {len(results)} chunker tests")
    logger.info(f"Successful: {len(successful)}, Failed: {len(failed)}")
    
    # Find best chunker based on query time and chunk count
    if successful:
        fastest = min(successful, key=lambda x: x.get("timing", {}).get("query", float("inf")))
        most_chunks = max(successful, key=lambda x: x.get("num_chunks", 0))
        
        logger.info(f"Fastest query time: {fastest['chunker']} ({fastest.get('timing', {}).get('query', 0):.2f} seconds)")
        logger.info(f"Most chunks created: {most_chunks['chunker']} ({most_chunks.get('num_chunks', 0)} chunks)")
        
        # Print responses for comparison
        logger.info("Responses from each chunker:")
        for result in successful:
            logger.info(f"\n--- {result['chunker']} ({result.get('num_chunks', 0)} chunks) ---")
            logger.info(result.get('response', 'No response'))
    
    # Save complete results
    with open("chunker_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    logger.info("Results saved to chunker_test_results.json")

if __name__ == "__main__":
    main() 
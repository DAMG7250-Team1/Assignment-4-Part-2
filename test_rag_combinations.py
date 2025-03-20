#!/usr/bin/env python3
"""
Test script to evaluate different combinations of parsers, chunkers, and vector databases
in the NVIDIA RAG pipeline.
"""

import os
import sys
import logging
import time
from pathlib import Path
from dotenv import load_dotenv
import json

# Load environment variables from .env file
load_dotenv()

# Add the src directory to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("rag_combinations_test.log")
    ]
)
logger = logging.getLogger(__name__)

# Import components
from src.parsers.basic_parser import BasicPDFParser
try:
    from src.parsers.docling_parser import DoclingPDFParser
    DOCLING_AVAILABLE = True
except ImportError:
    logger.warning("Docling parser not available. Skipping tests with this parser.")
    DOCLING_AVAILABLE = False

try:
    from src.parsers.mistral_parser import MistralPDFParser
    MISTRAL_AVAILABLE = True
except ImportError:
    logger.warning("Mistral parser not available. Skipping tests with this parser.")
    MISTRAL_AVAILABLE = False

from src.chunking.chunker_factory import ChunkerFactory
from src.vectorstore.vector_store_factory import VectorStoreFactory
from src.rag.pipeline import RAGPipeline
from src.processor.document_processor import DocumentProcessor

def setup_rag_pipeline(chunking_strategy, vector_store_type, persist_dir):
    """Set up a RAG pipeline with the specified components."""
    
    # Configure the vector store
    vector_store_args = {
        "persist_directory": persist_dir
    }
    
    if vector_store_type == "pinecone":
        # Add Pinecone-specific arguments
        vector_store_args.update({
            "index_name": "nvidia-test",
            "namespace": f"test-{int(time.time())}",
            "region": "us-west1"  # Adjust region as needed
        })
    
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
            "stride": 200
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
        vector_store_type=vector_store_type,
        vector_store_args={**vector_store_args, "embedding_function": embedding_function},
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

def process_with_parser(parser_name, file_path, output_dir):
    """Process a PDF file with the specified parser."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Open the PDF file
        with open(file_path, 'rb') as pdf_file:
            pdf_data = pdf_file.read()
            pdf_stream = io.BytesIO(pdf_data)
        
        # Set up S3 file manager (dummy for local testing)
        class DummyS3Manager:
            def upload_text(self, text, path):
                return True
                
        s3_manager = DummyS3Manager()
        
        # Process with the specified parser
        if parser_name == "basic":
            parser = BasicPDFParser()
        elif parser_name == "docling" and DOCLING_AVAILABLE:
            parser = DoclingPDFParser()
        elif parser_name == "mistral" and MISTRAL_AVAILABLE:
            parser = MistralPDFParser()
        else:
            logger.error(f"Parser {parser_name} not available")
            return None, None
            
        # Extract text using the parser
        output_path, metadata = parser.extract_text_from_pdf(pdf_stream, output_dir, s3_manager)
        
        return output_path, metadata
    except Exception as e:
        logger.error(f"Error processing with {parser_name} parser: {str(e)}")
        return None, None

def evaluate_combination(parser_name, chunking_strategy, vector_store_type, test_file, test_query):
    """
    Evaluate a specific combination of parser, chunker, and vector store.
    
    Args:
        parser_name: Name of the parser to use
        chunking_strategy: Name of the chunking strategy
        vector_store_type: Type of vector store
        test_file: PDF file to process
        test_query: Query to test with
        
    Returns:
        Dictionary with evaluation results
    """
    try:
        start_time = time.time()
        
        # Create unique directory for this combination
        combo_dir = f"data/test_{parser_name}_{chunking_strategy}_{vector_store_type}_{int(time.time())}"
        os.makedirs(combo_dir, exist_ok=True)
        
        logger.info(f"Testing combination: Parser={parser_name}, Chunker={chunking_strategy}, VectorStore={vector_store_type}")
        
        # Step 1: Process PDF with the specified parser
        parser_start = time.time()
        output_path, metadata = process_with_parser(parser_name, test_file, f"{combo_dir}/parsed")
        parser_time = time.time() - parser_start
        
        if not output_path:
            return {
                "success": False,
                "error": f"Failed to process with {parser_name} parser",
                "parser": parser_name,
                "chunker": chunking_strategy,
                "vector_store": vector_store_type
            }
        
        # Step 2: Set up and use RAG pipeline with specified chunker and vector store
        pipeline_start = time.time()
        pipeline = setup_rag_pipeline(chunking_strategy, vector_store_type, f"{combo_dir}/vectorstore")
        setup_time = time.time() - pipeline_start
        
        # Step 3: Ingest document
        ingest_start = time.time()
        chunk_ids = pipeline.process_document(output_path, document_type="file")
        ingest_time = time.time() - ingest_start
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
            "parser": parser_name,
            "chunker": chunking_strategy,
            "vector_store": vector_store_type,
            "num_chunks": num_chunks,
            "timing": {
                "parser": parser_time,
                "setup": setup_time,
                "ingestion": ingest_time,
                "query": query_time,
                "total": total_time
            },
            "query": test_query,
            "response": response
        }
        
        # Save result to file
        with open(f"{combo_dir}/result.json", "w") as f:
            json.dump(result_data, f, indent=2)
            
        logger.info(f"Evaluation complete for {parser_name}_{chunking_strategy}_{vector_store_type}")
        logger.info(f"Created {num_chunks} chunks and processed query in {query_time:.2f} seconds")
        
        return result_data
        
    except Exception as e:
        logger.error(f"Error evaluating combination: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "parser": parser_name,
            "chunker": chunking_strategy,
            "vector_store": vector_store_type
        }

def main():
    # Test file
    test_file = "data/reports/NVIDIA_2025_Third_Quarter_2025.pdf"
    if not os.path.exists(test_file):
        logger.error(f"Test file not found: {test_file}")
        return
        
    # Test query
    test_query = "What was NVIDIA's revenue in the most recent quarter?"
    
    # Available components to test
    parsers = ["basic"]
    if DOCLING_AVAILABLE:
        parsers.append("docling")
    if MISTRAL_AVAILABLE:
        parsers.append("mistral")
        
    chunkers = ["fixed_size", "sliding_window", "semantic"]
    vector_stores = ["chromadb", "pinecone"]
    
    # Track results
    results = []
    
    # Test all combinations (or a subset for testing)
    for parser_name in parsers:
        for chunking_strategy in chunkers:
            for vector_store_type in vector_stores:
                # Skip Pinecone if API key not available
                if vector_store_type == "pinecone" and not os.getenv("PINECONE_API_KEY"):
                    logger.warning("Skipping Pinecone test (no API key available)")
                    continue
                    
                result = evaluate_combination(
                    parser_name, 
                    chunking_strategy, 
                    vector_store_type,
                    test_file,
                    test_query
                )
                results.append(result)
                
                # Add a small delay between tests
                time.sleep(1)
    
    # Summarize results
    successful = [r for r in results if r.get("success", False)]
    failed = [r for r in results if not r.get("success", False)]
    
    logger.info(f"Completed {len(results)} combination tests")
    logger.info(f"Successful: {len(successful)}, Failed: {len(failed)}")
    
    # Find best combination based on query time
    if successful:
        fastest = min(successful, key=lambda x: x.get("timing", {}).get("query", float("inf")))
        most_chunks = max(successful, key=lambda x: x.get("num_chunks", 0))
        
        logger.info(f"Fastest query time: {fastest['parser']}_{fastest['chunker']}_{fastest['vector_store']} "
                   f"({fastest.get('timing', {}).get('query', 0):.2f} seconds)")
        logger.info(f"Most chunks created: {most_chunks['parser']}_{most_chunks['chunker']}_{most_chunks['vector_store']} "
                   f"({most_chunks.get('num_chunks', 0)} chunks)")
    
    # Save complete results
    with open("rag_combinations_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    logger.info("Results saved to rag_combinations_results.json")

if __name__ == "__main__":
    # Need to import io here for BytesIO
    import io
    main() 
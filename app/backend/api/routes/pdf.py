from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List, Optional, Dict, Any
import io
import os
import logging
import json
import time

from src.parsers.basic_parser import BasicPDFParser
from src.parsers.mistral_parser import MistralPDFParser
from src.parsers.docling_parser import DoclingPDFParser
from src.data.storage.s3_handler import S3FileManager
from src.rag.methods.naive_rag import NaiveRAG
from src.rag.methods.pinecone_rag import PineconeRAG
from src.rag.methods.chroma_rag import ChromaRAG
from src.chunking.chunker_factory import ChunkerFactory

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

def get_parser(parser_type: str):
    """
    Get the appropriate parser based on the selected type.
    
    Args:
        parser_type: Type of parser to use
    
    Returns:
        Parser instance
    
    Raises:
        ValueError: If the parser type is invalid
    """
    if parser_type == "basic":
        return BasicPDFParser()
    elif parser_type == "mistral":
        return MistralPDFParser()
    elif parser_type == "docling":
        return DoclingPDFParser()
    else:
        raise ValueError(f"Unknown parser type: {parser_type}")

def get_rag_method(method_type: str, chunking_strategy: str = "fixed_size", chunking_options: Dict[str, Any] = None):
    """
    Get the appropriate RAG method based on the selected type.
    
    Args:
        method_type: Type of RAG method to use
        chunking_strategy: Strategy for document chunking
        chunking_options: Additional options for the chunking strategy
    
    Returns:
        RAG method instance
    
    Raises:
        ValueError: If the RAG method type is invalid
    """
    # Get Gemini API key for the generator
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    chunking_options = chunking_options or {}
    
    if method_type == "naive":
        return NaiveRAG()
    elif method_type == "pinecone":
        return PineconeRAG(
            chunking_strategy=chunking_strategy,
            api_key=gemini_api_key,  # For the Gemini generator
            **chunking_options
        )
    elif method_type == "chroma":
        return ChromaRAG(
            chunking_strategy=chunking_strategy,
            api_key=gemini_api_key,  # For the Gemini generator
            **chunking_options
        )
    else:
        raise ValueError(f"Unknown RAG method: {method_type}")

@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    parser_type: str = Form(...),
    rag_method: str = Form(...),
    chunking_strategy: str = Form(...),
    metadata: Optional[str] = Form(None)
):
    """
    Upload and process a PDF document.
    
    Args:
        file: PDF file to process
        parser_type: Type of parser to use (basic, mistral, docling)
        rag_method: RAG method to use (naive, pinecone, chroma)
        chunking_strategy: Strategy for document chunking
        metadata: Optional JSON metadata for the document
    
    Returns:
        JSON response with processing results
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing PDF: {file.filename}")
        logger.info(f"Using parser: {parser_type}, RAG method: {rag_method}, chunking strategy: {chunking_strategy}")
        
        # Parse optional metadata
        parsed_metadata = {}
        if metadata:
            try:
                parsed_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                logger.warning("Invalid metadata JSON provided, using empty metadata")
        
        # Add basic file metadata
        parsed_metadata.update({
            "filename": file.filename,
            "upload_time": time.time(),
            "parser": parser_type,
            "rag_method": rag_method,
            "chunking_strategy": chunking_strategy
        })
        
        # Initialize S3 file manager
        s3_manager = S3FileManager()
        
        # Get appropriate parser
        parser = get_parser(parser_type)
        
        # Read file content
        file_content = await file.read()
        file_bytes = io.BytesIO(file_content)
        
        # Generate base path for S3
        filename = file.filename.replace(".pdf", "").replace(" ", "_")
        base_path = f"processed/{filename}"
        
        # Process PDF with parser
        output_path, file_metadata = parser.extract_text_from_pdf(file_bytes, base_path, s3_manager)
        
        # Combine metadata
        combined_metadata = {**parsed_metadata, **file_metadata}
        
        # Get processed text
        text_content = s3_manager.read_text(output_path)
        if not text_content:
            raise HTTPException(status_code=500, detail="Failed to retrieve processed text content")
        
        # Get chunking options from metadata if provided
        chunking_options = combined_metadata.get("chunking_options", {})
        
        # Initialize RAG method and add document
        rag_instance = get_rag_method(rag_method, chunking_strategy, chunking_options)
        
        # Add document to RAG system
        if isinstance(rag_instance, NaiveRAG):
            doc_id = rag_instance.add_document(text_content, combined_metadata)
            chunk_ids = [doc_id]
        else:
            chunk_ids = rag_instance.add_document(text_content, combined_metadata)
        
        processing_time = time.time() - start_time
        
        return {
            "status": "success",
            "filename": file.filename,
            "output_path": output_path,
            "processing_time": processing_time,
            "chunks_added": len(chunk_ids),
            "metadata": combined_metadata
        }
        
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/parsers")
async def get_parsers():
    """Get information about available parsers."""
    return {
        "parsers": [
            {
                "id": "basic",
                "name": "Basic PDF Parser",
                "description": "Simple text extraction using PyPDF2"
            },
            {
                "id": "mistral",
                "name": "Mistral OCR Parser",
                "description": "Advanced OCR using Mistral's AI capabilities"
            },
            {
                "id": "docling",
                "name": "Docling Parser",
                "description": "Document parsing with layout analysis and structure preservation"
            }
        ]
    }

@router.get("/chunking-strategies")
async def get_chunking_strategies():
    """Get information about available chunking strategies."""
    try:
        # Get strategies from the factory
        strategies = ChunkerFactory.get_available_strategies()
        
        # Format for response
        strategy_info = [
            {"id": "fixed_size", "name": "Fixed Size", "description": "Split documents into chunks of fixed size"},
            {"id": "sliding_window", "name": "Sliding Window", "description": "Create overlapping chunks with a sliding window"},
            {"id": "semantic", "name": "Semantic", "description": "Split documents at semantic boundaries"}
        ]
        
        return {"strategies": strategy_info}
    except Exception as e:
        logger.error(f"Error getting chunking strategies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/quarters")
async def get_quarters():
    """Get available document quarters for filtering."""
    return {
        "quarters": [
            "2022-Q1", "2022-Q2", "2022-Q3", "2022-Q4",
            "2023-Q1", "2023-Q2", "2023-Q3", "2023-Q4"
        ]
    }
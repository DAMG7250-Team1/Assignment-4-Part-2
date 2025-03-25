from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import time
import os

from src.rag.methods.naive_rag import NaiveRAG, Response, Source
from src.rag.methods.pinecone_rag import PineconeRAG
from src.rag.methods.chroma_rag import ChromaRAG

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Pydantic models for request/response validation
class QueryRequest(BaseModel):
    query: str
    rag_method: str
    quarters: List[str] = []
    top_k: int = 3
    chunking_strategy: str = "fixed_size"

class SourceModel(BaseModel):
    document: str
    context: str
    score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceModel]
    processing_time: float

def get_rag_method(method_type: str, chunking_strategy: str = "fixed_size"):
    """Get the appropriate RAG method instance."""
    # Get Gemini API key for the generator
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    if method_type == "naive":
        # Initialize NaiveRAG with load_from_s3=True to automatically load documents
        return NaiveRAG(load_from_s3=True)
    elif method_type == "pinecone":
        return PineconeRAG(
            chunking_strategy=chunking_strategy,
            api_key=gemini_api_key  # For the Gemini generator
        )
    elif method_type == "chroma":
        return ChromaRAG(
            chunking_strategy=chunking_strategy,
            api_key=gemini_api_key  # For the Gemini generator
        )
    else:
        raise ValueError(f"Unknown RAG method: {method_type}")

@router.post("", response_model=QueryResponse)
async def process_query(request: QueryRequest = Body(...)):
    """
    Process a query using the specified RAG method
    
    - **query**: The user question to be answered
    - **rag_method**: RAG method to use (naive, pinecone, chroma)
    - **quarters**: List of quarters to filter documents by (optional)
    - **top_k**: Number of top documents to retrieve
    - **chunking_strategy**: Strategy for document chunking
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing query: '{request.query}' with method: {request.rag_method}")
        
        # Get the appropriate RAG method
        rag_method = get_rag_method(request.rag_method, request.chunking_strategy)
        
        # Extract year information if quarters are provided
        year_filter = None
        if request.quarters:
            logger.info(f"Filtering by quarters: {request.quarters}")
            # Extract years from quarters (format: YYYY_QN)
            years = set()
            for quarter in request.quarters:
                parts = quarter.split('_')
                if len(parts) > 0 and parts[0].isdigit():
                    years.add(parts[0])
            
            if years:
                year_filter = list(years)
                logger.info(f"Extracted years for filtering: {year_filter}")
        
        # Process the query - handle different implementations
        if request.rag_method == "naive":
            # NaiveRAG uses 'query' method
            if hasattr(rag_method, 'year_filter') and year_filter:
                rag_method.year_filter = year_filter
            
            answer = rag_method.query(request.query)
            
            # Improve formatting of the answer
            answer = format_financial_answer(answer, request.query)
            
            # Get citations if available
            sources = []
            if hasattr(rag_method, 'get_citations') and callable(getattr(rag_method, 'get_citations')):
                citations = rag_method.get_citations()
                
                for i, citation in enumerate(citations):
                    # Parse the citation to extract document and context
                    # This assumes a specific format - adjust as needed
                    doc_parts = citation.split('\n', 1)
                    document = doc_parts[0] if len(doc_parts) > 0 else "Unknown"
                    context = doc_parts[1] if len(doc_parts) > 1 else citation
                    
                    sources.append(SourceModel(
                        document=document,
                        context=context,
                        score=1.0 - (i * 0.1)  # Simple scoring based on order
                    ))
            
        else:
            # Pinecone and Chroma RAG implementations
            # Create metadata filter if quarters are specified
            filter_dict = None
            if request.quarters:
                filter_dict = {"quarter": {"$in": request.quarters}}
            
            # These implementations might have process_query or query methods
            if hasattr(rag_method, 'process_query'):
                response = rag_method.process_query(
                    request.query, 
                    filter_dict=filter_dict,
                    top_k=request.top_k
                )
                answer = response.answer
                
                # Improve formatting of the answer
                answer = format_financial_answer(answer, request.query)
                
                sources = [
                    SourceModel(
                        document=source.document,
                        context=source.context,
                        score=source.score
                    ) for source in response.sources
                ]
            elif hasattr(rag_method, 'query'):
                # Fall back to query method if process_query doesn't exist
                answer = rag_method.query(request.query)
                
                # Improve formatting of the answer
                answer = format_financial_answer(answer, request.query)
                
                # Try to get sources if available
                sources = []
                if hasattr(rag_method, 'get_citations') and callable(getattr(rag_method, 'get_citations')):
                    citations = rag_method.get_citations()
                    for i, citation in enumerate(citations):
                        doc_parts = citation.split('\n', 1)
                        document = doc_parts[0] if len(doc_parts) > 0 else "Unknown"
                        context = doc_parts[1] if len(doc_parts) > 1 else citation
                        
                        sources.append(SourceModel(
                            document=document,
                            context=context,
                            score=1.0 - (i * 0.1)
                        ))
            else:
                raise ValueError(f"RAG method {request.rag_method} does not have a compatible query interface")
        
        processing_time = time.time() - start_time
        logger.info(f"Query processed in {processing_time:.2f} seconds")
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def format_financial_answer(answer, query):
    """
    Format the answer to make financial data more readable and structured.
    
    Args:
        answer (str): The raw answer from the RAG pipeline
        query (str): The user's query
        
    Returns:
        str: A formatted answer with better structure
    """
    # Check if this is a financial query
    financial_keywords = ['revenue', 'profit', 'earning', 'income', 'margin', 'financial', 'quarter', 'fiscal']
    is_financial_query = any(keyword in query.lower() for keyword in financial_keywords)
    
    if not is_financial_query:
        return answer

    import re
    
    # Simple function to clean up formatting issues
    def clean_text(text):
        # Fix broken pipe characters
        text = re.sub(r'∣', '|', text)
        
        # Clean up number formatting
        text = re.sub(r'(\d+)\s*,\s*(\d+)', r'\1,\2', text)
        
        # Clean up table formatting
        text = re.sub(r'\|\s+', '| ', text)
        text = re.sub(r'\s+\|', ' |', text)
        
        return text
    
    # Clean the answer text
    cleaned_answer = clean_text(answer)
    
    # Create a structured response with sections
    formatted_answer = "# NVIDIA Financial Analysis\n\n"
    
    # If the query is for a specific quarter, add that information
    quarter_match = re.search(r'Q[1-4]\s*20\d{2}', query, re.IGNORECASE)
    if quarter_match:
        quarter = quarter_match.group(0).upper()
        formatted_answer = f"# NVIDIA Financial Results: {quarter}\n\n"
    
    # Extract key sections from the answer
    
    # Look for tables
    if '|' in cleaned_answer:
        formatted_answer += "## Financial Data\n\n"
        
        # Extract and clean tables
        tables = re.findall(r'(\|[^\n]+\|(?:\n\|[^\n]+\|)+)', cleaned_answer)
        if tables:
            for table in tables:
                formatted_answer += f"{table}\n\n"
        
        # Remove tables from the text for further processing
        narrative = re.sub(r'(\|[^\n]+\|(?:\n\|[^\n]+\|)+)', '', cleaned_answer)
        narrative = narrative.strip()
        
        if narrative:
            formatted_answer += "## Additional Information\n\n"
            formatted_answer += narrative
    else:
        # No tables, just use the cleaned answer
        formatted_answer += cleaned_answer
    
    # Add citation footer
    formatted_answer += "\n\n*Data sourced from NVIDIA financial reports*"
    
    return formatted_answer

@router.get("/methods")
async def get_rag_methods():
    """Get information about available RAG methods"""
    return {
        "methods": [
            {
                "id": "naive",
                "name": "Naive RAG",
                "description": "Simple in-memory implementation"
            },
            {
                "id": "pinecone",
                "name": "Pinecone RAG",
                "description": "Vector similarity search with Pinecone"
            },
            {
                "id": "chroma",
                "name": "Chroma RAG",
                "description": "Local vector search with ChromaDB"
            }
        ]
    }

@router.get("/chunking-strategies")
async def get_chunking_strategies():
    """Get information about available chunking strategies"""
    return {
        "strategies": [
            {
                "id": "fixed_size",
                "name": "Fixed Size",
                "description": "Split documents into chunks of fixed size",
                "recommendation": "Best for general use cases - good balance of speed and accuracy",
                "performance": {
                    "ingestion_speed": "Medium",
                    "query_speed": "Fast (0.88 sec)",
                    "storage_impact": "Baseline",
                    "chunks_created": "~242 per document"
                },
                "default": True
            },
            {
                "id": "sliding_window",
                "name": "Sliding Window",
                "description": "Create overlapping chunks with a sliding window",
                "recommendation": "Best for critical analytics when every detail matters",
                "performance": {
                    "ingestion_speed": "Slow",
                    "query_speed": "Fastest (0.68 sec)",
                    "storage_impact": "High (~4x more storage)",
                    "chunks_created": "~969 per document"
                },
                "default": False
            },
            {
                "id": "semantic",
                "name": "Semantic",
                "description": "Split documents at semantic boundaries",
                "recommendation": "Best for complex documents with strong topical structure",
                "performance": {
                    "ingestion_speed": "Medium",
                    "query_speed": "Fast (0.73 sec)",
                    "storage_impact": "Medium (~1.4x more storage)",
                    "chunks_created": "~329 per document"
                },
                "default": False
            }
        ]
    }
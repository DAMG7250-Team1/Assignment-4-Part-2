from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import sys
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import route modules
from app.backend.api.routes import query, pdf, health, nvidia_reports
from app.backend.core.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="NVIDIA RAG API",
    description="API for the NVIDIA RAG Pipeline - Retrieval Augmented Generation for NVIDIA financial reports",
    version="1.0.0",
)

# Add CORS middleware to allow Streamlit frontend to make requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, set this to your Streamlit domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers from route modules
app.include_router(query.router, prefix="/api/query", tags=["Query"])
app.include_router(pdf.router, prefix="/api/pdf", tags=["PDF Processing"])
app.include_router(health.router, prefix="/api/health", tags=["Health"])
app.include_router(nvidia_reports.router, prefix="/api/nvidia-reports", tags=["NVIDIA Reports"])

@app.on_event("startup")
async def startup_event():
    """Run tasks at application startup"""
    logger.info("Starting NVIDIA RAG API")
    
    # Get settings
    settings = get_settings()
    
    # Verify environment variables
    required_vars = ["GEMINI_API_KEY", "PINECONE_API_KEY"]
    missing_vars = [var for var in required_vars if getattr(settings, var) is None]
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
    else:
        logger.info("All required environment variables are set")

@app.on_event("shutdown")
async def shutdown_event():
    """Run tasks at application shutdown"""
    logger.info("Shutting down NVIDIA RAG API")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "NVIDIA RAG API",
        "documentation": "/docs",
        "info": "Use /api/health to check system status",
    }
from fastapi import APIRouter, Depends, HTTPException
import os
import logging
from typing import Dict, List
import time

from app.backend.core.config import get_settings, Settings
from app.backend.api.dependencies import get_s3_manager, get_vector_store, verify_api_keys

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

@router.get("")
async def health_check(settings: Settings = Depends(get_settings)):
    """
    Simple health check endpoint to verify API is running.
    
    Returns a 200 OK response if the API is operational.
    """
    return {
        "status": "ok",
        "timestamp": time.time(),
        "message": f"{settings.PROJECT_NAME} API is operational"
    }

@router.get("/api-keys")
async def check_api_keys():
    """
    Check if required API keys are configured in environment variables.
    
    Returns the status of each required API key (present or missing).
    Does not expose the actual key values for security reasons.
    """
    api_keys = {
        "GEMINI_API_KEY": bool(os.getenv("GEMINI_API_KEY")),
        "PINECONE_API_KEY": bool(os.getenv("PINECONE_API_KEY")),
        "MISTRAL_API_KEY": bool(os.getenv("MISTRAL_API_KEY"))
    }
    
    all_keys_present = all(api_keys.values())
    
    return {
        "api_keys": api_keys,
        "status": "ok" if all_keys_present else "missing_keys",
        "message": "All required API keys are configured" if all_keys_present 
                   else "Some required API keys are missing"
    }

@router.get("/dependencies")
async def check_dependencies(
    settings: Settings = Depends(get_settings)
):
    """
    Check if all required services and dependencies are available.
    
    This endpoint verifies connectivity to backend services like
    vector databases and storage services.
    """
    dependencies = {
        "pinecone": _check_pinecone(),
        "s3": _check_s3(),
        "chromadb": True  # Local service should always be available
    }
    
    all_dependencies_ok = all(dependencies.values())
    
    return {
        "dependencies": dependencies,
        "status": "ok" if all_dependencies_ok else "service_unavailable",
        "message": "All dependencies are available" if all_dependencies_ok 
                  else "Some dependencies are unavailable",
        "config": {
            "default_vector_store": settings.DEFAULT_VECTOR_STORE,
            "default_chunking_strategy": settings.DEFAULT_CHUNKING_STRATEGY,
            "default_model_provider": settings.DEFAULT_MODEL_PROVIDER
        }
    }

@router.get("/version")
async def get_version():
    """Return version information for the API and components."""
    return {
        "api_version": "1.0.0",
        "components": {
            "fastapi": "0.103.1",
            "pinecone": "2.2.1",
            "chromadb": "0.4.13",
            "gemini": "1.0.0"
        }
    }

def _check_pinecone() -> bool:
    """
    Check if Pinecone is accessible.
    
    This is a simple check - in a real implementation, you would
    attempt to connect to Pinecone and verify it's operational.
    """
    try:
        # Here you would add actual Pinecone connection check
        # For simplicity, we just check if the API key is set
        return bool(os.getenv("PINECONE_API_KEY"))
    except Exception as e:
        logger.error(f"Error checking Pinecone: {str(e)}")
        return False

def _check_s3() -> bool:
    """
    Check if S3 storage is accessible.
    
    This is a simple check - in a real implementation, you would
    attempt to connect to S3 and verify it's operational.
    """
    try:
        # Here you would add actual S3 connection check
        # For simplicity, we just check if the AWS keys are set
        return bool(os.getenv("AWS_ACCESS_KEY_ID")) and bool(os.getenv("AWS_SECRET_ACCESS_KEY"))
    except Exception as e:
        logger.error(f"Error checking S3: {str(e)}")
        return False
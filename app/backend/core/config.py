import os
from pydantic_settings import BaseSettings
from typing import List, Dict, Any, Optional
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # Go up to the project root
ENV_FILE = PROJECT_ROOT / ".env"

class Settings(BaseSettings):
    """Application settings and configuration.
    
    This class uses Pydantic's BaseSettings to load configuration from environment 
    variables with fallback to default values.
    """
    
    # API Settings
    API_V1_STR: str = "/api"
    PROJECT_NAME: str = "NVIDIA RAG Pipeline"
    BACKEND_CORS_ORIGINS: List[str] = ["*"]  # In production, specify the domains
    
    # RAG Settings
    DEFAULT_VECTOR_STORE: str = "chromadb"
    DEFAULT_CHUNKING_STRATEGY: str = "semantic"
    DEFAULT_RAG_METHOD: str = "chroma"
    
    # LLM Settings
    DEFAULT_MODEL_PROVIDER: str = "google"
    DEFAULT_MODEL_NAME: str = "gemini-flash"
    DEFAULT_TEMPERATURE: float = 0.2
    
    # Storage Settings
    S3_BUCKET: str = os.getenv("S3_BUCKET", "rag-flow-forge")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    
    # Pinecone Settings
    PINECONE_INDEX: str = "nvidia-reports"
    PINECONE_CLOUD: str = "aws"
    PINECONE_REGION: str = "us-east-1"
    
    # ChromaDB Settings
    CHROMA_COLLECTION: str = "nvidia-reports"
    CHROMA_PERSIST_DIR: Optional[str] = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    
    # Parser Settings
    DEFAULT_PARSER: str = "basic"
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    MISTRAL_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_ENVIRONMENT: Optional[str] = None
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    
    class Config:
        """Pydantic config."""
        env_file = str(ENV_FILE)
        case_sensitive = True
        extra = "ignore"  # Allow extra fields from env file

# Initialize settings
settings = Settings()

def get_settings() -> Settings:
    """Returns the settings instance.
    This function is used as a dependency in FastAPI endpoints
    to access configuration settings.
    """
    return settings
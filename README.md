# NVIDIA RAG Pipeline

A Retrieval-Augmented Generation (RAG) pipeline for NVIDIA financial reports, enabling intelligent question answering about NVIDIA's financial performance and business operations.

## Features

- **Automated Data Collection**: Downloads NVIDIA financial reports from their investor relations website
- **Advanced Document Processing**: Processes PDF reports with multiple chunking strategies
- **Flexible Vector Storage**: Supports multiple vector databases (ChromaDB, Pinecone)
- **Semantic Search**: Retrieves relevant information from reports based on user queries
- **Intelligent Response Generation**: Uses advanced LLMs (OpenAI, Anthropic, Mistral) to generate insightful responses
- **Interactive UI**: Streamlit-based interface for easy interaction

## Architecture

The pipeline consists of several key components:

1. **Document Processor**: Ingests documents, splits them into chunks, and stores them in a vector database
2. **Chunking Strategies**: Implements various strategies (fixed-size, sliding window, semantic) for document chunking
3. **Vector Stores**: Support for different vector databases (ChromaDB, Pinecone)
4. **Retriever**: Retrieves relevant document chunks based on queries, with support for MMR reranking
5. **Generator**: Generates responses using various LLM providers (OpenAI, Anthropic, Mistral)
6. **RAG Pipeline**: Orchestrates the entire process from document ingestion to response generation

## Installation

### Prerequisites

- Python 3.8+
- pip package manager
- API keys for the services you plan to use (OpenAI, Anthropic, Mistral, Pinecone)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/nvidia-rag-pipeline.git
cd nvidia-rag-pipeline
```

2. Create a virtual environment (optional but recommended):

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Using conda
conda create -n nvidia-rag python=3.8
conda activate nvidia-rag
```

3. Install the package and dependencies:

```bash
pip install -e .
# Or to just install dependencies:
pip install -r requirements.txt
```

4. Set up environment variables:

Create a `.env` file in the project root with your API keys:

```
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
MISTRAL_API_KEY=your_mistral_api_key
PINECONE_API_KEY=your_pinecone_api_key

# Optional AWS credentials if using S3
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=your_aws_region
S3_BUCKET=your_s3_bucket
```

## Usage

### Example Script

The simplest way to use the pipeline is with the example script:

```bash
# Download NVIDIA reports and start the interactive Q&A
python examples/nvidia_rag_example.py --download --reports-dir data/reports --vector-store-dir data/vectorstore --max-reports 5

# Use existing reports
python examples/nvidia_rag_example.py --reports-dir data/reports --vector-store-dir data/vectorstore
```

### Using the Pipeline in Your Code

```python
from src.rag.pipeline import RAGPipeline
from src.vectorstore.vector_store_factory import VectorStoreFactory

# Create an embedding function
embedding_function = VectorStoreFactory.create_openai_embedding_function()

# Create a RAG pipeline
pipeline = RAGPipeline(
    vector_store_type="chromadb",
    vector_store_args={
        "collection_name": "my_documents",
        "persist_directory": "data/vectorstore",
        "embedding_function": embedding_function
    },
    chunking_strategy="semantic",
    model_provider="openai",
    model_name="gpt-4"
)

# Process a document
pipeline.process_document(
    document="path/to/document.pdf",
    document_type="file",
    metadata={"source": "example"}
)

# Query the pipeline
result = pipeline.query(
    query="What is NVIDIA's revenue growth?",
    use_mmr=True
)

# Print the response
print(result["response"])
```

## Customization

### Chunking Strategies

The pipeline supports multiple chunking strategies:

- **Fixed Size**: Splits text into chunks of a specified size
- **Sliding Window**: Creates overlapping chunks using a sliding window approach
- **Semantic**: Creates chunks based on semantic similarity and natural topic boundaries

### Vector Stores

Support for different vector databases:

- **ChromaDB**: In-memory or persistent local vector storage
- **Pinecone**: Cloud-based vector database with high scalability

### LLM Providers

Support for various LLM providers:

- **OpenAI**: GPT-3.5, GPT-4, etc.
- **Anthropic**: Claude models
- **Mistral**: Mistral AI models

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NVIDIA for publishing their financial reports
- The open-source community behind the libraries used in this project


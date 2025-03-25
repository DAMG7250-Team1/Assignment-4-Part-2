# NVIDIA Financial Reports Analysis

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
git clone https://github.com/DAMG7250-Team1/Assignment-4-Part-2.git
cd Assignment-4-Part-2
```

2. Create and activate a virtual environment:
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Using conda
conda create -n nvidia-rag python=3.8
conda activate nvidia-rag
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.template .env
# Edit .env with your API keys and credentials
```

## Project Structure

```
.
├── backend/
│   ├── agent/         # RAG agent implementation
│   ├── core/          # Core functionality
│   └── features/      # Feature implementations
├── frontend/          # Streamlit frontend
├── airflow/          # Airflow DAGs
├── .env              # Environment variables
└── requirements.txt  # Python dependencies
```

## Usage

1. Start the backend server:
```bash
cd backend
python main.py
```

2. Start the frontend:
```bash
cd frontend
streamlit run app.py
```

3. Access the application at http://localhost:8501

## Environment Variables

See `.env.template` for required environment variables:
- Mistral AI API Key
- OpenAI API Key
- Pinecone Configuration
- AWS Credentials
- ChromaDB Configuration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

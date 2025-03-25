# NVIDIA Financial Reports Analysis

This project provides a comprehensive analysis of NVIDIA's financial reports using advanced NLP and RAG (Retrieval-Augmented Generation) techniques.

## Features

- Web scraping of NVIDIA's quarterly reports
- PDF processing using Mistral OCR
- Document chunking with multiple strategies
- Vector storage using Pinecone
- Interactive query interface with Streamlit
- AWS S3 integration for document storage

## Setup

1. Clone the repository:
```bash
git clone https://github.com/DAMG7250-Team1/Assignment-4-Part-2.git
cd Assignment-4-Part-2
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
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
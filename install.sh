#!/bin/bash
# Installation script for NVIDIA RAG Pipeline

set -e  # Exit on error

echo "Installing NVIDIA RAG Pipeline dependencies..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python version must be at least 3.8. Your version is $python_version."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install the package in development mode
echo "Installing NVIDIA RAG Pipeline in development mode..."
pip install -e .

# Create necessary directories
mkdir -p data/reports data/vectorstore

# Create .env file from example if it doesn't exist
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    echo "Creating .env file from example..."
    cp .env.example .env
    echo "Please edit the .env file with your API keys and settings."
fi

echo "Installation complete!"
echo "To activate the environment, run: source venv/bin/activate"
echo "To test the pipeline, run: python examples/nvidia_rag_example.py --download" 
#!/bin/bash

# Check if AWS credentials are set in environment
if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "Warning: AWS credentials not found in environment variables."
    echo "Please set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, and BUCKET_NAME before running this script."
    echo "Example:"
    echo "export AWS_ACCESS_KEY_ID=your_access_key"
    echo "export AWS_SECRET_ACCESS_KEY=your_secret_key"
    echo "export AWS_REGION=your_region"
    echo "export BUCKET_NAME=your_bucket_name"
fi

# Set default region if not provided
if [ -z "$AWS_REGION" ]; then
    export AWS_REGION="us-east-1"
    echo "AWS_REGION not set, using default: us-east-1"
fi

# Set default bucket name if not provided
if [ -z "$BUCKET_NAME" ]; then
    echo "Warning: BUCKET_NAME not set. S3 operations may fail."
fi

# Use the Python 3.11 installation where all dependencies are installed
PYTHON_PATH="/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11"

# Verify Python exists
if [ ! -f "$PYTHON_PATH" ]; then
    echo "Error: Python 3.11 not found at $PYTHON_PATH"
    exit 1
fi

# Run the app with the appropriate Python version
cd "$(dirname "$0")"
"$PYTHON_PATH" -m streamlit run app/frontend/nvidia_app.py 
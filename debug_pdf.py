#!/usr/bin/env python3
"""
Debug script to test PDF processing and chunking
"""

import os
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the src directory to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

import PyPDF2
from src.chunking.chunker_factory import ChunkerFactory

def main():
    # File path
    pdf_path = "data/reports/NVIDIA_2025_Third_Quarter_2025.pdf"
    
    # Extract text from PDF
    logger.info(f"Extracting text from {pdf_path}")
    
    try:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text() or ""
                text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                
        logger.info(f"Extracted {len(text)} characters of text")
        logger.info(f"First 200 characters: {text[:200]}")
        
        # Analyze paragraph structure
        paragraphs = text.split("\n\n")
        logger.info(f"Found {len(paragraphs)} paragraphs using '\\n\\n' as separator")
        
        # Check if we have too few paragraphs, which suggests text might not be using \n\n
        if len(paragraphs) < 10:
            logger.info("Few paragraphs found, trying alternative splitting")
            # Try splitting by just \n
            alt_paragraphs = text.split("\n")
            logger.info(f"Found {len(alt_paragraphs)} paragraphs using '\\n' as separator")
            
            # Look at newline patterns in the text
            newline_counts = text.count("\n")
            double_newline_counts = text.count("\n\n")
            logger.info(f"Text contains {newline_counts} single newlines and {double_newline_counts} double newlines")
            
            # Create a modified text with more standard paragraph separations
            modified_text = text.replace("\n", "\n\n")
            
            # Create fixed-size chunker with modified text
            logger.info("Attempting to chunk with modified text (single newlines converted to double)")
            chunker = ChunkerFactory.get_chunker(
                strategy_name="fixed_size",
                chunk_size=1000,
                chunk_overlap=200
            )
            
            metadata = {"file_path": pdf_path, "file_name": os.path.basename(pdf_path)}
            chunks = chunker.create_chunks(modified_text, metadata)
            
            logger.info(f"Created {len(chunks)} chunks from the modified text")
            
            # Print first chunk if available
            if chunks:
                logger.info(f"First chunk text (first 200 chars): {chunks[0]['text'][:200]}")
                logger.info(f"First chunk metadata: {chunks[0]['metadata']}")
            else:
                logger.error("No chunks were created from modified text!")
        else:
            # Continue with original approach
            # Create fixed-size chunker
            chunker = ChunkerFactory.get_chunker(
                strategy_name="fixed_size",
                chunk_size=1000,
                chunk_overlap=200
            )
            
            # Create chunks
            metadata = {"file_path": pdf_path, "file_name": os.path.basename(pdf_path)}
            chunks = chunker.create_chunks(text, metadata)
            
            logger.info(f"Created {len(chunks)} chunks from the PDF")
            
            # Print first chunk
            if chunks:
                logger.info(f"First chunk text (first 200 chars): {chunks[0]['text'][:200]}")
                logger.info(f"First chunk metadata: {chunks[0]['metadata']}")
            else:
                logger.error("No chunks were created!")
            
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 
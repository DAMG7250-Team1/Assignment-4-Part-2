#!/usr/bin/env python3
"""
NVIDIA RAG Pipeline Example

This script demonstrates how to use the RAG pipeline with NVIDIA financial reports
to answer questions about NVIDIA's financial performance and business operations.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import requests
import time
import re
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.select import Select

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load environment variables from .env file at project root
    load_dotenv(Path(__file__).parent.parent / '.env')
    print("Environment variables loaded from .env file")
except ImportError:
    print("dotenv package not found. Install with 'pip install python-dotenv' to load environment variables from .env file")
    # Continue without dotenv, assuming environment variables are already set

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.rag.pipeline import RAGPipeline
from src.vectorstore.vector_store_factory import VectorStoreFactory
from src.data.ingestion.pdf_downloader import NvidiaReportDownloader
from src.data.storage.s3_handler import S3FileManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_nvidia_reports(output_dir: str, max_reports: int = 5) -> List[str]:
    """
    Download NVIDIA financial reports.
    
    Args:
        output_dir: Directory to save the reports
        max_reports: Maximum number of reports to download
        
    Returns:
        List of file paths to the downloaded reports
    """
    logger.info(f"Downloading up to {max_reports} NVIDIA financial reports...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Download reports
    downloader = NvidiaReportDownloader()
    return downloader.download_all_reports(
        output_dir=output_dir,
        max_reports=max_reports
    )

def create_rag_pipeline(persist_directory: Optional[str] = None) -> RAGPipeline:
    """
    Create a RAG pipeline.
    
    Args:
        persist_directory: Optional directory to persist vector store data
        
    Returns:
        RAGPipeline instance
    """
    # Create embedding function using Google/Vertex AI instead of OpenAI
    # Make sure to set the GOOGLE_API_KEY from the GEMINI_API_KEY in the environment
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it in your .env file.")
    
    os.environ["GOOGLE_API_KEY"] = gemini_api_key
    embedding_function = VectorStoreFactory.create_google_embedding_function()
    
    # Create vector store
    vector_store_args = {
        "collection_name": "nvidia_reports",
        "persist_directory": persist_directory,
        "embedding_function": embedding_function
    }
    
    # Create pipeline with default settings
    return RAGPipeline(
        vector_store_type="chromadb",
        vector_store_args=vector_store_args,
        chunking_strategy="fixed_size",  # Use fixed size chunking for better reliability with PDFs
        chunking_args={
            "chunk_size": 1000,
            "chunk_overlap": 200
        },
        retrieval_top_k=4,
        model_provider="google",  # Using Google for LLM
        model_name="gemini-2.0-flash",  # Using Gemini model
        model_args={
            "temperature": 0.2,  # Lower temperature for more factual responses
            "system_prompt": """You are a financial analyst assistant specializing in NVIDIA and the semiconductor industry.
            Your task is to provide accurate, detailed analyses of NVIDIA's financial performance, business operations,
            and market position based on their official financial reports. When answering questions:
            
            1. Focus on factual information from the provided context
            2. Cite specific quarters, years, figures, and growth rates when relevant
            3. Acknowledge when information might be outdated or not available in the context
            4. Avoid speculation beyond what's supported by the data
            5. Use clear, professional financial terminology
            
            Always indicate which reports or documents your information comes from."""
        }
    )

def ingest_reports(pipeline: RAGPipeline, report_files: List[str]) -> None:
    """
    Ingest reports into the RAG pipeline.
    
    Args:
        pipeline: RAG pipeline
        report_files: List of report file paths
    """
    logger.info(f"Ingesting {len(report_files)} reports into the RAG pipeline...")
    
    # Process each report
    common_metadata = {"source_type": "nvidia_financial_report"}
    result = pipeline.process_documents(
        documents=report_files,
        document_type="file",
        common_metadata=common_metadata
    )
    
    logger.info(f"Successfully ingested {len(result)} reports")

def ask_questions(pipeline: RAGPipeline) -> None:
    """
    Interactive question answering using the RAG pipeline.
    
    Args:
        pipeline: RAG pipeline
    """
    print("\n===== NVIDIA Financial Reports Q&A =====")
    print("Type 'exit' or 'quit' to end the session\n")
    
    # Sample questions to suggest
    sample_questions = [
        "What was NVIDIA's revenue in the most recent quarter?",
        "How has NVIDIA's gross margin changed over time?",
        "What are NVIDIA's main growth drivers?",
        "How is NVIDIA's AI business performing?",
        "What risks does NVIDIA face according to recent reports?",
        "How has the GPU shortage affected NVIDIA's business?"
    ]
    
    print("Sample questions you can ask:")
    for i, question in enumerate(sample_questions, 1):
        print(f"{i}. {question}")
    print()
    
    # Interactive question loop
    while True:
        query = input("Enter your question (or 'exit' to quit): ")
        
        if query.lower() in ['exit', 'quit']:
            break
        
        if not query.strip():
            continue
        
        print("\nProcessing your question...\n")
        
        # Query the pipeline
        result = pipeline.query(
            query=query,
            use_mmr=True  # Use MMR for more diverse results
        )
        
        # Print the response
        print("\n" + "=" * 80)
        print(result["response"])
        print("=" * 80)
        
        # Print source information
        print("\nSources:")
        if "retrieval" in result["metadata"]:
            for i, doc in enumerate(result["metadata"]["retrieval"]["docs"], 1):
                source = doc["metadata"].get("file_name", f"Source {i}")
                print(f"{i}. {source} (similarity: {doc['score']:.4f})")
        print()

def scrape_pdf_links():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36")
    driver = webdriver.Chrome(options=options)
 
    bucket_name = os.getenv('BUCKET_NAME')
    s3_client = S3FileManager.get_s3_client()
    try:
        # Navigate to the page
        driver.get("https://investor.nvidia.com/financial-info/quarterly-results/default.aspx")
       
        pdf_links = {}
        years_to_scrape = {"2024"}  # Changed to 2024 as per your requirement
       
        for year in years_to_scrape:
            # Wait for the dropdown to be present
            wait = WebDriverWait(driver, 10)
            dropdown_element = wait.until(EC.presence_of_element_located((By.ID, "_ctrl0_ctl75_selectEvergreenFinancialAccordionYear")))
           
            # Create Select object
            select = Select(dropdown_element)
            
            # Check if the year is available in the dropdown
            options_text = [option.text for option in select.options]
            if year not in options_text:
                print(f"Warning: Year {year} not found in dropdown. Available years: {options_text}")
                continue
            
            # Select by visible text
            select.select_by_visible_text(year)
           
            # Wait for page to update after selection
            time.sleep(3)
           
            # Find all accordion toggle buttons
            accordion_buttons = driver.find_elements(By.XPATH, "//button[contains(@class, 'evergreen-financial-accordion-toggle')]")
           
            year_links = {}
           
            # Expand each accordion section
            for i, button in enumerate(accordion_buttons):
                try:
                    # Get quarter name before clicking
                    quarter_name = button.find_element(By.XPATH, ".//span[@class='evergreen-accordion-title']").text
 
                    time.sleep(1)
                    # Try multiple click methods for better reliability
                    try:
                        # First try JavaScript click
                        driver.execute_script("arguments[0].click();", button)
                    except:
                        try:
                            # Then try regular click
                            button.click()
                        except:
                            # Finally try action chains
                            actions = ActionChains(driver)
                            actions.move_to_element(button).click().perform()
                   
                    # Wait for content to load - longer wait for first accordion
                    wait_time = 3 if i == 0 else 1
                    time.sleep(wait_time)
                   
                    # Check if panel is expanded
                    is_expanded = button.get_attribute("aria-expanded") == "true"
                    if not is_expanded:
                        print(f"Warning: {quarter_name} panel did not expand, trying again")
                        driver.execute_script("arguments[0].click();", button)
                        time.sleep(2)
                   
                    # Find the expanded content container
                    panel_id = button.get_attribute("aria-controls")
                    panel = wait.until(EC.presence_of_element_located((By.ID, panel_id)))
                   
                    # Find all links in the expanded section
                    links = panel.find_elements(By.XPATH, ".//a[contains(@class, 'evergreen-financial-accordion-link')]")
                   
                    for link in links:
                        href = link.get_attribute('href')
                        text = link.text.strip()
                       
                        # Filter for only 10-Q and 10-K documents
                        if '10-Q' in text or '10-K' in text:
                            quarter = re.sub(r"\s+", "_", quarter_name)
                            year_links.update({
                                quarter: href,
                                # 'text': text,
                                # 'url': href
                            })
                   
                    # Close this accordion section before opening the next one
                    driver.execute_script("arguments[0].click();", button)
                    time.sleep(1)
               
                except Exception as e:
                    print(f"Error processing accordion {i}: {str(e)}")
           
            pdf_links[year] = year_links
 
        json_data = json.dumps(pdf_links)  
        s3_key = "metadata/metadata.json"
        try:
            s3_client.put_object(
                Bucket=bucket_name,
                Key=s3_key,
                Body=json_data,
                ContentType='application/json'
            )
            # Also save locally as fallback
            os.makedirs("metadata", exist_ok=True)
            with open("metadata/metadata.json", "w") as f:
                f.write(json_data)
            return "metadata.json file uploaded successfully "
        except Exception as e:
            print(f"Error uploading to S3: {e}")
            # Save locally if S3 fails
            try:
                os.makedirs("metadata", exist_ok=True)
                with open("metadata/metadata.json", "w") as f:
                    f.write(json_data)
                return "metadata.json saved locally (S3 upload failed)"
            except Exception as local_error:
                print(f"Error saving locally: {local_error}")
                return "file not uploaded to s3"
       
    finally:
        # Close the browser when done
        driver.quit()

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="NVIDIA RAG Pipeline Example")
    parser.add_argument("--download", action="store_true", help="Download NVIDIA reports")
    parser.add_argument("--reports-dir", default="data/reports", help="Directory for report files")
    parser.add_argument("--vector-store-dir", default="data/vectorstore", help="Directory for vector store data")
    parser.add_argument("--max-reports", type=int, default=5, help="Maximum number of reports to download")
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.reports_dir, exist_ok=True)
    os.makedirs(args.vector_store_dir, exist_ok=True)
    
    # Download reports if requested
    report_files = []
    if args.download:
        report_files = download_nvidia_reports(args.reports_dir, args.max_reports)
    else:
        # Use existing reports
        report_files = [os.path.join(args.reports_dir, f) for f in os.listdir(args.reports_dir) 
                       if f.endswith(('.pdf', '.PDF'))]
    
    if not report_files:
        logger.error("No report files found. Use --download to download reports.")
        return
    
    logger.info(f"Found {len(report_files)} report files")
    
    # Create RAG pipeline
    pipeline = create_rag_pipeline(args.vector_store_dir)
    
    # Ingest reports
    ingest_reports(pipeline, report_files)
    
    # Start interactive Q&A
    ask_questions(pipeline)

if __name__ == "__main__":
    main() 
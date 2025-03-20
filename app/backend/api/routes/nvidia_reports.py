from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Depends, Path
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import logging
import uuid
import re
import os
from datetime import datetime
from src.data.ingestion.pdf_downloader import NvidiaReportDownloader
from src.data.storage.s3_handler import S3FileManager
from src.parsers.basic_parser import BasicPDFParser
from src.parsers.docling_parser import DoclingPDFParser
from src.parsers.mistral_parser import MistralPDFParser

logger = logging.getLogger(__name__)
router = APIRouter()

# Dictionary to store background download tasks
download_tasks: Dict[str, Dict[str, Any]] = {}

# Dictionary to store background processing tasks
processing_tasks: Dict[str, Dict[str, Any]] = {}

class DownloadReportsRequest(BaseModel):
    years: Optional[List[str]] = Field(
        default=None, 
        description="List of years to download reports for, e.g. ['2020', '2021']. If not provided, all available years will be downloaded."
    )
    max_reports: Optional[int] = Field(
        default=None, 
        description="Maximum number of reports to download."
    )
    force_refresh: Optional[bool] = Field(
        default=False, 
        description="If True, will re-download reports even if they already exist in S3."
    )

class ProcessPDFsRequest(BaseModel):
    parser_type: str = Field(
        description="Parser type to use. Options: 'basic', 'mistral', 'docling'."
    )
    pdf_paths: Optional[List[str]] = Field(
        default=None,
        description="List of S3 paths to PDF files to process. If not provided, all PDFs in the reports directory will be processed."
    )
    year_filter: Optional[str] = Field(
        default=None,
        description="Filter PDFs by year, e.g. '2023'."
    )

class DownloadStatus(BaseModel):
    task_id: str
    status: str  # "pending", "in_progress", "completed", "failed"
    total_reports: Optional[int] = None
    downloaded_reports: Optional[int] = None
    s3_paths: Optional[List[str]] = None
    error: Optional[str] = None

class ProcessingStatus(BaseModel):
    task_id: str
    status: str  # "pending", "in_progress", "completed", "failed"
    total_pdfs: Optional[int] = None
    processed_pdfs: Optional[int] = None
    output_paths: Optional[List[str]] = None
    error: Optional[str] = None

@router.post("/download", response_model=Dict[str, str])
async def download_reports(
    request: DownloadReportsRequest,
    background_tasks: BackgroundTasks
):
    """
    Initiate the downloading of NVIDIA reports.
    
    Returns a task ID that can be used to check the status of the download.
    """
    task_id = str(uuid.uuid4())
    
    # Initialize task status
    download_tasks[task_id] = {
        "status": "pending",
        "total_reports": 0,
        "downloaded_reports": 0,
        "s3_paths": [],
        "error": None
    }
    
    # Start background task
    background_tasks.add_task(
        _download_reports_in_background, 
        task_id, 
        request.years, 
        request.max_reports,
        request.force_refresh
    )
    
    return {"task_id": task_id, "message": "Download task started"}

def _download_reports_in_background(
    task_id: str,
    years: Optional[List[str]] = None,
    max_reports: Optional[int] = None,
    force_refresh: bool = False
):
    """Download NVIDIA reports in the background."""
    try:
        # Update task status
        download_tasks[task_id]["status"] = "in_progress"
        
        # Rest of the implementation is in the async version below
        logger.info(f"Starting download task {task_id} for years: {years}")
    except Exception as e:
        logger.error(f"Error starting background task: {str(e)}")
        download_tasks[task_id]["status"] = "failed"
        download_tasks[task_id]["error"] = str(e)

@router.get("/download/{task_id}", response_model=DownloadStatus)
async def get_download_status(task_id: str):
    """
    Get the status of a download task.
    
    Args:
        task_id: The task ID
        
    Returns:
        DownloadStatus object
    """
    if task_id not in download_tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    task = download_tasks[task_id]
    
    return DownloadStatus(
        task_id=task_id,
        status=task["status"],
        total_reports=task["total_reports"],
        downloaded_reports=task["downloaded_reports"],
        s3_paths=task["s3_paths"],
        error=task["error"]
    )

@router.get("/list", response_model=Dict[str, Any])
async def list_reports():
    """
    List all NVIDIA reports available in S3.
    
    Returns:
        Dict with total reports count and reports by year
    """
    try:
        s3_manager = S3FileManager()
        
        # List all PDF files in the reports folder
        files = s3_manager.list_files(prefix="reports/")
        
        # Filter for PDF files
        pdf_files = [f for f in files if f.endswith(".pdf") and "NVIDIA" in f]
        
        # Also check in downloaded_reports folder
        downloaded_files = s3_manager.list_files(prefix="downloaded_reports/")
        pdf_downloaded_files = [f for f in downloaded_files if f.endswith(".pdf") and "NVIDIA" in f]
        
        # Combine the lists
        all_pdf_files = pdf_files + pdf_downloaded_files
        
        # Organize by year
        reports_by_year = {}
        for file_path in all_pdf_files:
            # Extract year from filename or path
            year_match = re.search(r'NVIDIA_(\d{4})_', file_path)
            
            if year_match:
                year = year_match.group(1)
            else:
                # If year can't be extracted, put in "Other" category
                year = "Other"
            
            if year not in reports_by_year:
                reports_by_year[year] = []
            
            filename = os.path.basename(file_path)
            
            # Add file info without metadata
            reports_by_year[year].append({
                "filename": filename,
                "s3_path": file_path,
                "last_modified": None,  # No metadata available
                "size": 0  # No metadata available
            })
        
        # Return response
        return {
            "total_reports": len(all_pdf_files),
            "reports": all_pdf_files,  # Simple list of paths
            "reports_by_year": reports_by_year
        }
        
    except Exception as e:
        logger.error(f"Error listing reports: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing reports: {str(e)}")

async def _download_reports_in_background(
    task_id: str, 
    years: Optional[List[str]], 
    max_reports: int, 
    force_refresh: bool
):
    """
    Background task for downloading NVIDIA reports.
    
    Args:
        task_id: The task ID
        years: List of years to download or None for all
        max_reports: Maximum number of reports to download
        force_refresh: Whether to force re-download even if files exist
    """
    try:
        # Initialize download task status
        download_tasks[task_id] = {
            "status": "in_progress",
            "total_reports": 0,
            "downloaded_reports": 0,
            "s3_paths": [],
            "error": None
        }
        
        # Initialize S3 file manager and report downloader
        s3_manager = S3FileManager()
        downloader = NvidiaReportDownloader(s3_manager=s3_manager)
        
        # Use the existing method to download all reports directly to S3
        try:
            s3_paths = downloader.download_all_reports_s3_only(max_reports=max_reports or 10)
            
            # Filter by year if specified
            if years:
                filtered_paths = []
                for path in s3_paths:
                    for year in years:
                        if f"NVIDIA_{year}_" in path:
                            filtered_paths.append(path)
                            break
                s3_paths = filtered_paths
            
            # Update task status
            download_tasks[task_id]["status"] = "completed"
            download_tasks[task_id]["total_reports"] = len(s3_paths)
            download_tasks[task_id]["downloaded_reports"] = len(s3_paths)
            download_tasks[task_id]["s3_paths"] = s3_paths
            
            logger.info(f"Successfully downloaded {len(s3_paths)} reports to S3")
            
        except Exception as e:
            logger.error(f"Error downloading reports: {str(e)}")
            download_tasks[task_id]["status"] = "failed"
            download_tasks[task_id]["error"] = f"Error downloading reports: {str(e)}"
        
        # Close the WebDriver
        downloader.quit()
        
    except Exception as e:
        logger.error(f"Background download task error: {str(e)}")
        
        # Update task status
        if task_id in download_tasks:
            download_tasks[task_id]["status"] = "failed"
            download_tasks[task_id]["error"] = f"Background download task error: {str(e)}" 
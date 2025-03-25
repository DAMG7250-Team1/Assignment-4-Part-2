#!/usr/bin/env python
"""
NVIDIA Reports API Client

This script demonstrates how to use the FastAPI endpoints to download NVIDIA reports.
"""

import requests
import json
import time
import argparse
import sys
import os
from typing import List, Optional

# API Base URL - adjust as needed
DEFAULT_API_URL = "http://localhost:8000"

def download_nvidia_reports(
    api_url: str,
    years: Optional[List[str]] = None,
    max_reports: int = 50,
    force_refresh: bool = False
) -> str:
    """
    Start a download job for NVIDIA reports.
    
    Args:
        api_url: Base URL of the API
        years: List of years to download or None for all
        max_reports: Maximum number of reports to download
        force_refresh: Whether to force download even if files exist
        
    Returns:
        task_id: ID to check download status
    """
    print(f"Starting download of NVIDIA reports...")
    
    # Create request data
    request_data = {
        "years": years,
        "max_reports": max_reports,
        "force_refresh": force_refresh
    }
    
    # Call API to start download
    response = requests.post(
        f"{api_url}/api/nvidia-reports/download",
        json=request_data
    )
    
    # Check response
    if response.status_code == 200:
        result = response.json()
        task_id = result["task_id"]
        print(f"Download job started with task ID: {task_id}")
        return task_id
    else:
        print(f"Error starting download: {response.status_code}")
        print(response.text)
        sys.exit(1)

def check_download_status(api_url: str, task_id: str, wait: bool = False) -> dict:
    """
    Check the status of a download job.
    
    Args:
        api_url: Base URL of the API
        task_id: ID of the download task
        wait: Whether to wait for the download to complete
        
    Returns:
        dict: Download status
    """
    print(f"Checking status of download job {task_id}...")
    
    while True:
        # Call API to check status
        response = requests.get(f"{api_url}/api/nvidia-reports/download/{task_id}")
        
        # Check response
        if response.status_code == 200:
            status = response.json()
            
            # Print status
            print(f"Status: {status['status']}")
            print(f"Progress: {status['downloaded_reports']}/{status['total_reports']} reports")
            
            if status['error']:
                print(f"Error: {status['error']}")
            
            # If waiting and not completed, wait and try again
            if wait and status['status'] == 'in_progress':
                print("Waiting for download to complete...")
                time.sleep(5)
                continue
            
            return status
        else:
            print(f"Error checking status: {response.status_code}")
            print(response.text)
            sys.exit(1)
        
        # If not waiting, break out of loop
        if not wait:
            break

def list_available_reports(api_url: str) -> dict:
    """
    List all NVIDIA reports available in S3.
    
    Args:
        api_url: Base URL of the API
        
    Returns:
        dict: Report information
    """
    print(f"Listing available NVIDIA reports...")
    
    # Call API to list reports
    response = requests.get(f"{api_url}/api/nvidia-reports/list")
    
    # Check response
    if response.status_code == 200:
        result = response.json()
        
        # Print summary
        print(f"Found {result['total_reports']} reports")
        
        # Print details by year
        for year, reports in result.get('reports_by_year', {}).items():
            print(f"\nYear {year}: {len(reports)} reports")
            for report in reports:
                print(f"  - {report['filename']}")
        
        return result
    else:
        print(f"Error listing reports: {response.status_code}")
        print(response.text)
        sys.exit(1)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="NVIDIA Reports API Client")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Base URL of the API")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download NVIDIA reports")
    download_parser.add_argument("--years", nargs="+", help="Years to download (e.g. 2023 2022)")
    download_parser.add_argument("--max", type=int, default=50, help="Maximum number of reports to download")
    download_parser.add_argument("--force", action="store_true", help="Force download even if files exist")
    download_parser.add_argument("--wait", action="store_true", help="Wait for the download to complete")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check status of a download job")
    status_parser.add_argument("task_id", help="Task ID from the download command")
    status_parser.add_argument("--wait", action="store_true", help="Wait for the download to complete")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available reports")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the appropriate command
    if args.command == "download":
        task_id = download_nvidia_reports(
            api_url=args.api_url,
            years=args.years,
            max_reports=args.max,
            force_refresh=args.force
        )
        
        # If waiting for completion, check status
        if args.wait:
            check_download_status(args.api_url, task_id, wait=True)
    
    elif args.command == "status":
        check_download_status(args.api_url, args.task_id, wait=args.wait)
    
    elif args.command == "list":
        list_available_reports(args.api_url)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 
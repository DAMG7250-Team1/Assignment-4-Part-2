import streamlit as st
import os
import time
import requests
import pandas as pd
from datetime import datetime

# FastAPI backend URL
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")

def check_api_health():
    """Check if the FastAPI backend is healthy."""
    try:
        response = requests.get(f"{FASTAPI_URL}/api/health", timeout=2)
        return response.status_code == 200
    except Exception as e:
        st.error(f"API health check failed: {str(e)}")
        return False

def get_reports_list():
    """Get list of already downloaded reports from the API."""
    try:
        response = requests.get(f"{FASTAPI_URL}/api/nvidia-reports/list", timeout=5)
        if response.status_code == 200:
            return response.json().get("reports", [])
        else:
            st.error(f"Failed to get reports list: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        st.error(f"Error getting reports list: {str(e)}")
        return []

def check_download_status(task_id):
    """Check the status of a download task."""
    try:
        response = requests.get(f"{FASTAPI_URL}/api/nvidia-reports/download/{task_id}", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "status": "error",
                "error": f"API error: {response.status_code} - {response.text}"
            }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Error checking status: {str(e)}"
        }

def download_nvidia_reports_api(years=None, max_reports=None, force_refresh=False):
    """Download NVIDIA financial reports using the FastAPI endpoint."""
    try:
        # Prepare request payload
        payload = {
            "force_refresh": force_refresh
        }
        
        # Add optional parameters if provided
        if years:
            payload["years"] = years
        if max_reports is not None:
            payload["max_reports"] = max_reports
            
        # Make API request
        response = requests.post(
            f"{FASTAPI_URL}/api/nvidia-reports/download",
            json=payload,
            timeout=10  # Give it 10 seconds to respond with task ID
        )
        
        # Accept both 200 and 202 status codes as success
        if response.status_code in [200, 202]:  # OK or Accepted
            result = response.json()
            if "task_id" in result:
                return {
                    "success": True,
                    "task_id": result.get("task_id")
                }
            else:
                return {
                    "success": False,
                    "error": f"API response missing task_id: {response.text}"
                }
        else:
            return {
                "success": False,
                "error": f"API error: {response.status_code} - {response.text}"
            }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error initiating download: {str(e)}"
        }

def display_reports_table(reports):
    """Display a table of downloaded reports."""
    if not reports:
        st.info("No NVIDIA reports have been downloaded yet.")
        return
        
    # Prepare data for table
    table_data = []
    for report in reports:
        # Extract year and quarter from filename
        filename = os.path.basename(report)
        
        # Parse information from filename (NVIDIA_YYYY_Quarter_N.pdf)
        parts = filename.replace(".pdf", "").split("_")
        
        if len(parts) >= 4:
            year = parts[1]
            quarter = parts[3]
            table_data.append({
                "Year": year,
                "Quarter": f"Q{quarter}",
                "Filename": filename,
                "S3 Path": report
            })
        else:
            # Handle files that don't match the expected pattern
            table_data.append({
                "Year": "Unknown",
                "Quarter": "Unknown",
                "Filename": filename,
                "S3 Path": report
            })
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Sort by Year and Quarter (most recent first)
    df = df.sort_values(by=["Year", "Quarter"], ascending=[False, False])
    
    # Display table
    st.dataframe(df, use_container_width=True)

def download_page():
    """Download NVIDIA financial reports."""
    st.title("Download NVIDIA Financial Reports")
    
    # Check if API is available
    if not check_api_health():
        st.error("FastAPI backend is not available. Make sure the backend is running.")
        return
        
    st.success("FastAPI backend is available")
    
    # Display tabs for different sections
    tab1, tab2 = st.tabs(["Download Reports", "View Downloaded Reports"])
    
    with tab1:
        st.subheader("Download NVIDIA Financial Reports")
        st.write("""
        This tool can download quarterly financial reports from NVIDIA's investor relations website.
        Reports will be saved directly to the S3 bucket.
        """)
        
        # Form for download parameters
        with st.form("download_form"):
            # Year selection
            current_year = datetime.now().year
            available_years = list(range(current_year, current_year - 6, -1))  # Last 5 years
            
            selected_years = st.multiselect(
                "Select Years to Download",
                options=available_years,
                default=[current_year, current_year - 1],  # Default to current and previous year
                help="Select which years of reports to download"
            )
            
            # Maximum reports option
            max_reports = st.number_input(
                "Maximum Number of Reports",
                min_value=1,
                max_value=100,
                value=20,
                help="Limit the number of reports to download"
            )
            
            # Force refresh option
            force_refresh = st.checkbox(
                "Force Refresh",
                value=False,
                help="Re-download reports even if they already exist in S3"
            )
            
            # Submit button
            submitted = st.form_submit_button("Start Download", type="primary")
            
        if submitted:
            with st.spinner("Initiating download..."):
                # Convert selected years to strings
                year_strings = [str(year) for year in selected_years]
                
                # Call API to start download
                result = download_nvidia_reports_api(
                    years=year_strings,
                    max_reports=max_reports,
                    force_refresh=force_refresh
                )
                
                if result["success"]:
                    task_id = result["task_id"]
                    st.success(f"Download task initiated with ID: {task_id}")
                    
                    # Create a progress display
                    st.subheader("Download Progress")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    report_count = st.empty()
                    error_container = st.empty()
                    
                    # Poll for status updates
                    complete = False
                    start_time = time.time()
                    
                    while not complete and time.time() - start_time < 300:  # Timeout after 5 minutes
                        # Check status
                        status = check_download_status(task_id)
                        
                        if status["status"] == "completed":
                            progress_bar.progress(1.0)
                            status_text.success("Download completed!")
                            report_count.info(f"Downloaded {status.get('total_reports', 0)} reports")
                            
                            # Display the list of downloaded paths
                            if status.get('s3_paths'):
                                st.write("Downloaded reports:")
                                for path in status.get('s3_paths'):
                                    st.write(f"- {path}")
                            
                            complete = True
                            break
                        elif status["status"] == "processing":
                            # Update progress
                            total = status.get("total_reports", 0)
                            downloaded = status.get("downloaded_reports", 0)
                            
                            if total > 0:
                                progress = downloaded / total
                                progress_bar.progress(progress)
                                status_text.info(f"Downloading reports... {downloaded}/{total}")
                            else:
                                status_text.info("Preparing to download reports...")
                        elif status["status"] == "error":
                            progress_bar.progress(1.0)
                            status_text.error("Download failed!")
                            error_container.error(status.get("error", "Unknown error occurred"))
                            complete = True
                            break
                        
                        # Wait before checking again
                        time.sleep(3)
                    
                    # Handle timeout
                    if not complete:
                        status_text.warning("Monitoring timed out, but download may still be in progress")
                        st.info("You can check the 'View Downloaded Reports' tab to see if reports were downloaded")
                else:
                    st.error(f"Failed to start download: {result['error']}")
    
    with tab2:
        st.subheader("Downloaded NVIDIA Reports")
        
        # Add refresh button
        if st.button("Refresh List"):
            st.rerun()
        
        # Get list of reports from API
        reports = get_reports_list()
        
        # Display reports in a table
        display_reports_table(reports)
        
        if reports:
            # Additional statistics
            years_available = set()
            quarters_available = set()
            
            for report in reports:
                filename = os.path.basename(report)
                parts = filename.replace(".pdf", "").split("_")
                
                if len(parts) >= 4:
                    years_available.add(parts[1])
                    quarters_available.add(f"{parts[1]}_Q{parts[3]}")
            
            st.info(f"Reports available for {len(years_available)} years and {len(quarters_available)} quarters")
        else:
            st.warning("No reports have been downloaded yet. Go to the 'Download Reports' tab to download reports.")

if __name__ == "__main__":
    download_page() 
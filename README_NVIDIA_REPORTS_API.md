# NVIDIA Reports API Documentation

This document provides information on how to use the FastAPI endpoints for downloading and managing NVIDIA financial reports.

## API Endpoints

The NVIDIA Reports API is accessible via the following endpoints:

### Download Reports

```
POST /api/nvidia-reports/download
```

Initiates a background task to download NVIDIA reports to S3.

**Request Body:**
```json
{
  "years": ["2023", "2022"],  // optional - limit to specific years
  "max_reports": 50,          // optional - max number of reports to download
  "force_refresh": false      // optional - re-download even if file exists
}
```

**Response:**
```json
{
  "task_id": "12345678-1234-5678-1234-567812345678",
  "message": "Download task started"
}
```

### Check Download Status

```
GET /api/nvidia-reports/download/{task_id}
```

Checks the status of a download task.

**Response:**
```json
{
  "task_id": "12345678-1234-5678-1234-567812345678",
  "status": "in_progress", // "in_progress", "completed", "failed"
  "total_reports": 10,
  "downloaded_reports": 5,
  "s3_paths": ["reports/NVIDIA_2023_Quarter_1.pdf", "..."],
  "error": null
}
```

### List Available Reports

```
GET /api/nvidia-reports/list
```

Lists all NVIDIA reports available in S3.

**Response:**
```json
{
  "total_reports": 20,
  "reports_by_year": {
    "2023": [
      {
        "filename": "NVIDIA_2023_Quarter_1.pdf",
        "s3_path": "reports/NVIDIA_2023_Quarter_1.pdf",
        "last_modified": "2023-05-01T12:00:00Z",
        "size": 1234567
      },
      // More reports...
    ],
    "2022": [
      // Reports for 2022...
    ]
    // More years...
  }
}
```

## Using the Python Client Script

A Python client script is provided to interact with the API. The script is located at `scripts/download_nvidia_api.py`.

### Prerequisites

The script requires the following packages:
```
requests
```

You can install the dependencies with:
```
pip install requests
```

### Usage

#### Download Reports

```bash
python scripts/download_nvidia_api.py download [--years 2023 2022] [--max 50] [--force] [--wait]
```

Options:
- `--years`: Limit to specific years (space-separated)
- `--max`: Maximum number of reports to download (default: 50)
- `--force`: Force download even if files exist
- `--wait`: Wait for the download to complete

#### Check Download Status

```bash
python scripts/download_nvidia_api.py status <task_id> [--wait]
```

Options:
- `--wait`: Wait for the download to complete

#### List Available Reports

```bash
python scripts/download_nvidia_api.py list
```

### Examples

Download reports from 2022 and 2023:
```bash
python scripts/download_nvidia_api.py download --years 2022 2023
```

Download all available reports and wait for completion:
```bash
python scripts/download_nvidia_api.py download --wait
```

Check status of a download task:
```bash
python scripts/download_nvidia_api.py status 12345678-1234-5678-1234-567812345678
```

List all available reports:
```bash
python scripts/download_nvidia_api.py list
```

## Integration with the Streamlit App

The Streamlit application uses these API endpoints to download and process NVIDIA reports. When reports are downloaded, they are stored in the S3 bucket, making them available for RAG processing.

The download process checks if reports already exist in S3 before downloading, avoiding unnecessary downloads and bandwidth usage.

## Benefits of the API Approach

1. **One-time downloading**: Reports are downloaded once and stored in S3, avoiding repeated downloads.
2. **Background processing**: Download tasks run in the background, allowing users to continue using the application.
3. **Status tracking**: Users can track the progress of download tasks.
4. **Selective downloading**: Users can download reports for specific years only.
5. **Force refresh**: When needed, users can force a refresh of the reports, re-downloading them from the source. 
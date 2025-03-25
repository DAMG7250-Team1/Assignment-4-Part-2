#!/bin/bash

# Script to easily download NVIDIA financial reports
# This is a wrapper around the Python PDF downloader

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOWNLOAD_DIR="$REPO_ROOT/downloads"

# Default to download mode with 5 reports
SCRAPE=0
DOWNLOAD=1
S3_UPLOAD=0
MAX_REPORTS=5
YEARS=""
QUARTERS=""
QUIET=0

# Print usage information
function show_help {
    echo "Usage: $0 [options]"
    echo ""
    echo "Download NVIDIA financial reports"
    echo ""
    echo "Options:"
    echo "  --scrape-only       Scrape PDF links without downloading"
    echo "  --download-dir DIR  Directory to save PDFs (default: $DOWNLOAD_DIR)"
    echo "  --s3-upload         Upload PDFs to S3"
    echo "  --years YEARS       Years to download (e.g., '2021 2022 2023')"
    echo "  --quarters QTRS     Quarters to download (e.g., '1 3')"
    echo "  --max NUM           Maximum number of reports to download (default: 5)"
    echo "  --quiet             Reduce output verbosity"
    echo "  --help              Show this help message"
    echo ""
    echo "Example: $0 --years \"2022 2023\" --max 10 --s3-upload"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --scrape-only)
            SCRAPE=1
            DOWNLOAD=0
            shift
            ;;
        --download-dir)
            DOWNLOAD_DIR="$2"
            shift 2
            ;;
        --s3-upload)
            S3_UPLOAD=1
            shift
            ;;
        --years)
            YEARS="$2"
            shift 2
            ;;
        --quarters)
            QUARTERS="$2"
            shift 2
            ;;
        --max)
            MAX_REPORTS="$2"
            shift 2
            ;;
        --quiet)
            QUIET=1
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Construct the Python command
CMD="python -m src.data.ingestion.pdf_downloader"

# Add options
if [ $SCRAPE -eq 1 ]; then
    CMD="$CMD --scrape"
fi

if [ $DOWNLOAD -eq 1 ]; then
    CMD="$CMD --download"
fi

if [ $S3_UPLOAD -eq 1 ]; then
    CMD="$CMD --s3-upload"
fi

if [ -n "$YEARS" ]; then
    CMD="$CMD --years $YEARS"
fi

if [ -n "$QUARTERS" ]; then
    CMD="$CMD --quarters $QUARTERS"
fi

if [ $QUIET -eq 1 ]; then
    CMD="$CMD --quiet"
fi

CMD="$CMD --output \"$DOWNLOAD_DIR\" --max $MAX_REPORTS"

# Print the command if not in quiet mode
if [ $QUIET -eq 0 ]; then
    echo "Running: $CMD"
    echo "Working directory: $REPO_ROOT"
    echo "-------------------------------"
fi

# Run the command
cd "$REPO_ROOT" && eval "$CMD"

# Check exit status
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "Error: Command failed with exit code $EXIT_CODE"
    echo "You may need to install required packages:"
    echo "  pip install requests"
    echo "  pip install selenium (optional, for live scraping)"
    exit $EXIT_CODE
fi

exit 0 
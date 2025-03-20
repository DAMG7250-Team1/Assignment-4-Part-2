@echo off
REM Installation script for NVIDIA RAG Pipeline on Windows

echo Installing NVIDIA RAG Pipeline dependencies...

REM Check if Python is installed
python --version 2>NUL
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python is not installed or not in PATH. Please install Python 3.8+ first.
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%V in ('python -c "import sys; print(sys.version.split()[0])"') do set python_version=%%V
for /f "tokens=1,2 delims=." %%A in ("%python_version%") do (
    set major=%%A
    set minor=%%B
)
if %major% LSS 3 (
    echo Error: Python version must be at least 3.8. Your version is %python_version%.
    exit /b 1
)
if %major%==3 if %minor% LSS 8 (
    echo Error: Python version must be at least 3.8. Your version is %python_version%.
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
pip install --upgrade pip

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Install the package in development mode
echo Installing NVIDIA RAG Pipeline in development mode...
pip install -e .

REM Create necessary directories
if not exist data\reports mkdir data\reports
if not exist data\vectorstore mkdir data\vectorstore

REM Create .env file from example if it doesn't exist
if not exist .env if exist .env.example (
    echo Creating .env file from example...
    copy .env.example .env
    echo Please edit the .env file with your API keys and settings.
)

echo Installation complete!
echo To activate the environment, run: venv\Scripts\activate.bat
echo To test the pipeline, run: python examples\nvidia_rag_example.py --download

REM Keep the window open
pause 
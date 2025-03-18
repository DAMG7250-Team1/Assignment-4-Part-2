FROM apache/airflow:2.7.1

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    unzip \
    libmagic1 \
    chromium \
    chromium-driver \
    python3-dev \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Set environment variables for Chrome/Chromium
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver
ENV PYTHONPATH="${PYTHONPATH}:/opt/airflow"
ENV OMP_NUM_THREADS=4
ENV DISPLAY=:99
ENV SELENIUM_HEADLESS=1

# Copy requirements first to leverage Docker cache
COPY requirements.txt /opt/airflow/requirements.txt

# Install Python packages with pip upgrade
RUN pip install --upgrade pip && \
    pip install --no-cache-dir selenium webdriver-manager && \
    pip install --no-cache-dir -r /opt/airflow/requirements.txt

# Copy the rest of the application
COPY . /opt/airflow/

# Verify installations and setup with explicit ChromeDriver path
RUN python -c "from selenium import webdriver; from selenium.webdriver.chrome.options import Options; from selenium.webdriver.chrome.service import Service; options = Options(); options.add_argument('--headless'); options.add_argument('--no-sandbox'); options.add_argument('--disable-dev-shm-usage'); service = Service(executable_path='/usr/bin/chromedriver'); driver = webdriver.Chrome(service=service, options=options); driver.quit(); print('Selenium setup verified successfully!')" 
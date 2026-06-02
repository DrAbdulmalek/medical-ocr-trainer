# =============================================================
# Medical OCR Trainer — Docker for HF Spaces (Free Tier)
# Lightweight: 3 engines only (PaddleOCR + EasyOCR + Tesseract)
# TrOCR + Surya removed — need ~2.3GB extra RAM
# =============================================================
FROM python:3.11-slim

# System packages (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-ara \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Cache directories
ENV TORCH_HOME=/app/.cache/torch
ENV PADDLE_HOME=/app/.cache/paddleocr

# Create persistent storage dirs + cache dirs
RUN mkdir -p /app/.cache/torch /app/.cache/paddleocr \
    /data/uploads /data/crops /data/db /data/exports

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Verify Tesseract languages
RUN tesseract --list-langs

# Copy project files
COPY . .

# Pre-download models at build time (avoids runtime timeout)
RUN python pre_download_models.py

EXPOSE 7860

# Run Streamlit — XSRF disabled for HF Spaces proxy (fixes 403 upload errors)
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]

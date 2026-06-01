# =============================================================
# Medical OCR Trainer — Docker HF Space
# 5 Engines: PaddleOCR + EasyOCR + Tesseract + TrOCR + Surya
# =============================================================
FROM python:3.11-slim

# تثبيت نظام الحزم المطلوبة
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
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# تعيين متغيرات البيئة
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV TORCH_HOME=/app/.cache/torch
ENV PADDLE_HOME=/app/.cache/paddleocr

# منع تنزيل النماذج من الإنترنت أثناء التشغيل (النماذج مُنزّلة مسبقاً)
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# إنشاء مجلدات التخزين المؤقت
RUN mkdir -p /app/.cache/huggingface /app/.cache/torch /app/.cache/paddleocr /app/data /app/uploads /app/crops /app/exports

WORKDIR /app

# نسخ ملف المتطلبات وتثبيتها
COPY requirements.txt .

# تثبيت torch CPU-only أولاً (يوفر ~10GB مقارنة بنسخة CUDA الكاملة)
# ثم تثبيت باقي المتطلبات مع استخدام نفس فهرس CPU لضمان التوافق
RUN pip install --no-cache-dir \
    torch --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# تنزيل ملفات Tesseract النموذجية
RUN tesseract --list-langs

# نسخ ملفات المشروع
COPY . .

# تنزيل مسبق لنماذج كل المحركات (لتفادي timeout عند التشغيل)
RUN python pre_download_models.py

EXPOSE 7860

# تشغيل Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]

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
    libgl1-mesa-glx \
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

# إنشاء مجلدات التخزين المؤقت
RUN mkdir -p /app/.cache/huggingface /app/.cache/torch /app/.cache/paddleocr /app/data /app/uploads /app/crops /app/exports

WORKDIR /app

# نسخ ملف المتطلبات وتثبيتها
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# تنزيل ملفات Tesseract النموذجية (Ensures tesseract langs are present)
RUN tesseract --list-langs

# نسخ ملفات المشروع
COPY . .

# =============================================================
# PRE-DOWNLOAD MODELS (لتفادي timeout عند التشغيل الأول)
# =============================================================
RUN python -c "
import sys
print('=== Pre-downloading OCR models ===', flush=True)

# 1. PaddleOCR — تنزيل النماذج الأولية
print('[1/5] Downloading PaddleOCR models...', flush=True)
try:
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='ar', show_log=False, use_gpu=False)
    print('  PaddleOCR: OK', flush=True)
except Exception as e:
    print(f'  PaddleOCR: {e}', flush=True)

# 2. EasyOCR — تنزيل النماذج
print('[2/5] Downloading EasyOCR models...', flush=True)
try:
    import easyocr
    reader = easyocr.Reader(['ar', 'en'], gpu=False, download_enabled=True)
    print('  EasyOCR: OK', flush=True)
except Exception as e:
    print(f'  EasyOCR: {e}', flush=True)

# 3. TrOCR — تنزيل النموذج
print('[3/5] Downloading TrOCR models...', flush=True)
try:
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    print('  TrOCR: OK', flush=True)
    del processor, model
    import gc; gc.collect()
except Exception as e:
    print(f'  TrOCR: {e}', flush=True)

# 4. Surya OCR — تنزيل النماذج
print('[4/5] Downloading Surya OCR models...', flush=True)
try:
    from surya.model.detection.model import load_model as load_det
    from surya.model.recognition.model import load_model as load_rec
    det = load_det()
    rec = load_rec()
    print('  Surya OCR: OK', flush=True)
    del det, rec
    import gc; gc.collect()
except Exception as e:
    print(f'  Surya OCR: {e}', flush=True)

print('[5/5] All model downloads complete!', flush=True)
" || echo "Model pre-download completed with warnings"

EXPOSE 7860

# تشغيل Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]

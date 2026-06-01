"""Pre-download all OCR models during Docker build to avoid runtime timeout."""
import sys
import os
import gc

# تعطيل تنزيل النماذج من الإنترنت أثناء التشغيل (النماذج مُنزّلة مسبقاً)
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

print("=== Pre-downloading OCR models ===", flush=True)

# 1. PaddleOCR — تنزيل النماذج + تشغيل على صورة وهمية لضمان تحميل كل النماذج الفرعية
print("[1/5] Downloading PaddleOCR models...", flush=True)
try:
    from paddleocr import PaddleOCR
    import numpy as np
    from PIL import Image as PILImage

    ocr = PaddleOCR(use_textline_orientation=True, lang='ar')

    # إنشاء صورة وهمية وتشغيل OCR لضمان تحميل كل النماذج (det, rec, cls)
    dummy = PILImage.fromarray(np.zeros((100, 300, 3), dtype=np.uint8))
    dummy.save('/tmp/dummy_paddle.png')
    ocr.ocr('/tmp/dummy_paddle.png', cls=True)
    print("  PaddleOCR: OK (all sub-models loaded)", flush=True)
    del ocr
    gc.collect()
except Exception as e:
    print(f"  PaddleOCR: {e}", flush=True)
    # لا نوقف البناء — نسمح بالمتابعة

# 2. EasyOCR — تنزيل النماذج
print("[2/5] Downloading EasyOCR models...", flush=True)
try:
    import easyocr
    reader = easyocr.Reader(['ar', 'en'], gpu=False, download_enabled=True)
    print("  EasyOCR: OK", flush=True)
    del reader
    gc.collect()
except Exception as e:
    print(f"  EasyOCR: {e}", flush=True)

# تفعيل التحميل من الإنترنت لنماذج HF فقط
os.environ.pop('HF_HUB_OFFLINE', None)
os.environ.pop('TRANSFORMERS_OFFLINE', None)

# 3. TrOCR — تنزيل النموذج
print("[3/5] Downloading TrOCR models...", flush=True)
try:
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    print("  TrOCR: OK", flush=True)
    del processor, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
except Exception as e:
    print(f"  TrOCR: {e}", flush=True)

# 4. Surya OCR — تنزيل النماذج
print("[4/5] Downloading Surya OCR models...", flush=True)
try:
    from surya.detection import run_detection
    from surya.model.detection.model import load_model as load_det
    from surya.model.recognition.model import load_model as load_rec
    det = load_det()
    rec = load_rec()
    print("  Surya OCR: OK", flush=True)
    del det, rec
    gc.collect()
except Exception as e:
    print(f"  Surya OCR: {e}", flush=True)

# إعادة تعطيل التحميل من الإنترنت
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

print("[5/5] All model downloads complete!", flush=True)
print("=== Models cached successfully ===", flush=True)

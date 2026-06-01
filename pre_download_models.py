"""Pre-download all OCR models during Docker build to avoid runtime timeout."""
import sys
import gc

print("=== Pre-downloading OCR models ===", flush=True)

# 1. PaddleOCR
print("[1/5] Downloading PaddleOCR models...", flush=True)
try:
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='ar', show_log=False, use_gpu=False)
    print("  PaddleOCR: OK", flush=True)
except Exception as e:
    print(f"  PaddleOCR: {e}", flush=True)

# 2. EasyOCR
print("[2/5] Downloading EasyOCR models...", flush=True)
try:
    import easyocr
    reader = easyocr.Reader(['ar', 'en'], gpu=False, download_enabled=True)
    print("  EasyOCR: OK", flush=True)
except Exception as e:
    print(f"  EasyOCR: {e}", flush=True)

# 3. TrOCR
print("[3/5] Downloading TrOCR models...", flush=True)
try:
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    print("  TrOCR: OK", flush=True)
    del processor, model
    gc.collect()
except Exception as e:
    print(f"  TrOCR: {e}", flush=True)

# 4. Surya OCR
print("[4/5] Downloading Surya OCR models...", flush=True)
try:
    from surya.model.detection.model import load_model as load_det
    from surya.model.recognition.model import load_model as load_rec
    det = load_det()
    rec = load_rec()
    print("  Surya OCR: OK", flush=True)
    del det, rec
    gc.collect()
except Exception as e:
    print(f"  Surya OCR: {e}", flush=True)

print("[5/5] All model downloads complete!", flush=True)

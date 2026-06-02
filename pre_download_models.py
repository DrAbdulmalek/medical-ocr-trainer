"""Pre-download OCR models during Docker build to avoid runtime timeout.

Lightweight version for HF Free Tier — only 3 engines:
    1. PaddleOCR (Arabic + English)
    2. EasyOCR (Arabic + English)
    3. Tesseract (via apt-get, no download needed)
"""
import sys
import os
import gc

print("=== Pre-downloading OCR models (3 engines) ===", flush=True)

# 1. PaddleOCR — download models + warm up all sub-models (det, rec, cls)
print("[1/2] Downloading PaddleOCR models...", flush=True)
try:
    from paddleocr import PaddleOCR
    import numpy as np
    from PIL import Image as PILImage

    ocr = PaddleOCR(use_textline_orientation=True, lang='ar')

    # Create dummy image and run OCR to ensure all sub-models are loaded
    dummy = PILImage.fromarray(np.zeros((100, 300, 3), dtype=np.uint8))
    dummy.save('/tmp/dummy_paddle.png')
    ocr.ocr('/tmp/dummy_paddle.png', cls=True)
    print("  PaddleOCR: OK (all sub-models loaded)", flush=True)
    del ocr
    gc.collect()
except Exception as e:
    print(f"  PaddleOCR: {e}", flush=True)

# 2. EasyOCR — download language models
print("[2/2] Downloading EasyOCR models...", flush=True)
try:
    import easyocr
    reader = easyocr.Reader(['ar', 'en'], gpu=False, download_enabled=True)
    print("  EasyOCR: OK", flush=True)
    del reader
    gc.collect()
except Exception as e:
    print(f"  EasyOCR: {e}", flush=True)

print("=== All model downloads complete! ===", flush=True)

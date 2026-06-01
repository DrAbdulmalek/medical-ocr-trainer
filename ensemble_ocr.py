"""
نظام التجمع المتعدد المحركات للتعرف على النصوص
==================================================
يجمع بين 5 محركات OCR ويدمج نتائجها باستخدام استراتيجيات متقدمة:

المحركات المدعومة:
    1. PaddleOCR — الأفضل للعربية المطبوعة واليدوية
    2. EasyOCR — جيد للغة اللاتينية والمختلطة
    3. Tesseract — سريع وموثوق للمطبوع
    4. TrOCR — Transformer-based، ممتاز للخط اليدوي
    5. Surya OCR — محرك حديث عالي الدقة

استراتيجيات الدمج:
    - majority_voting: تصويت الأغلبية (3+ محركات)
    - confidence_weighted: متوسط مرجح بالثقة
    - levenshtein_consensus: أقرب نص بإجماع Levenshtein
    - best_single: أفضل نتيجة واحدة حسب الثقة

الاستخدام:
    from ensemble_ocr import EnsembleOCR

    ocr = EnsembleOCR(engines=['paddleocr', 'easyocr', 'tesseract'])
    results = ocr.process_image("scan.jpg", strategy='majority_voting')

الاستخدام من سطر الأوامر:
    python ensemble_ocr.py --image scan.jpg --engines all --strategy majority_voting
"""

import os
import sys
import json
import time
import logging
import importlib
import argparse
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

# ============================================================
# إعدادات التسجيل
# ============================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("EnsembleOCR")

# ============================================================
# هياكل البيانات
# ============================================================

@dataclass
class OcrWord:
    """نتيجة كلمة واحدة من محرك OCR"""
    text: str
    confidence: float
    bbox: List[List[float]] = field(default_factory=list)  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    engine: str = ""
    engine_idx: int = -1  # ترتيب الكلمة داخل المحرك

    def to_dict(self):
        return {
            "text": self.text,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "engine": self.engine,
        }


@dataclass
class EngineResult:
    """نتائج محرك واحد على صورة كاملة"""
    engine_name: str
    words: List[OcrWord] = field(default_factory=list)
    processing_time: float = 0.0
    available: bool = True
    error: str = ""

    @property
    def word_count(self):
        return len(self.words)

    def to_dict(self):
        return {
            "engine_name": self.engine_name,
            "word_count": self.word_count,
            "processing_time": round(self.processing_time, 2),
            "available": self.available,
            "error": self.error,
            "words": [w.to_dict() for w in self.words],
        }


@dataclass
class EnsembleWord:
    """نتيجة كلمة واحدة بعد دمج المحركات"""
    text: str
    confidence: float
    bbox: List[List[float]]
    engines_used: List[str] = field(default_factory=list)
    engine_votes: Dict[str, str] = field(default_factory=dict)  # engine -> text
    agreement_count: int = 0
    strategy: str = ""

    def to_dict(self):
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox,
            "engines_used": self.engines_used,
            "agreement_count": self.agreement_count,
            "strategy": self.strategy,
            "engine_votes": self.engine_votes,
        }


@dataclass
class EnsembleResult:
    """نتائج التجمع الكاملة"""
    words: List[EnsembleWord] = field(default_factory=list)
    engine_results: Dict[str, EngineResult] = field(default_factory=dict)
    total_time: float = 0.0
    strategy: str = ""
    engines_active: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "word_count": len(self.words),
            "total_time": round(self.total_time, 2),
            "strategy": self.strategy,
            "engines_active": self.engines_active,
            "words": [w.to_dict() for w in self.words],
            "per_engine": {k: v.to_dict() for k, v in self.engine_results.items()},
        }


# ============================================================
# واجهة المحرك الأساسية
# ============================================================

class BaseOcrEngine(ABC):
    """واجهة أساسية لكل محرك OCR"""

    name: str = "base"
    description: str = ""
    supports_arabic: bool = False
    supports_latin: bool = False
    supports_handwriting: bool = False
    memory_mb: int = 0

    @abstractmethod
    def is_available(self) -> bool:
        """التحقق من توفر المحرك"""
        pass

    @abstractmethod
    def recognize(self, image_path: str) -> List[OcrWord]:
        """تشغيل OCR والعودة بقائمة الكلمات"""
        pass

    def warmup(self):
        """تحميل مسبق (اختياري)"""
        pass


# ============================================================
# محرك PaddleOCR
# ============================================================

class PaddleOcrEngine(BaseOcrEngine):
    """
    PaddleOCR — الأفضل للنصوص العربية المطبوعة واليدوية
    ======================================================
    - lang='ar': يدعم العربية واللاتينية المختلطة
    - use_angle_cls=True: كشف اتجاه النص
    - استخدام الذاكرة: ~300MB
    """

    name = "paddleocr"
    description = "PaddleOCR (Arabic + English) — Best for mixed Arabic/Latin documents"
    supports_arabic = True
    supports_latin = True
    supports_handwriting = True
    memory_mb = 300

    def __init__(self, lang='ar'):
        self.lang = lang
        self._model = None
        self._available = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import paddleocr
            self._available = True
        except ImportError:
            self._available = False
            logger.warning("PaddleOCR not installed. Install with: pip install paddleocr paddlepaddle")
        return self._available

    def _get_model(self):
        if self._model is None:
            from paddleocr import PaddleOCR
            import logging as _logging
            # Suppress PaddleOCR verbose logs in production
            _logging.getLogger("ppocr").setLevel(_logging.WARNING)
            self._model = PaddleOCR(
                use_textline_orientation=True,
                lang=self.lang,
            )
        return self._model

    def recognize(self, image_path: str) -> List[OcrWord]:
        model = self._get_model()
        t0 = time.time()

        try:
            results = model.ocr(image_path, cls=True)
            words = []
            if results and results[0]:
                for idx, line in enumerate(results[0]):
                    bbox, (text, conf) = line[0], line[1]
                    words.append(OcrWord(
                        text=str(text),
                        confidence=float(conf),
                        bbox=bbox,
                        engine=self.name,
                        engine_idx=idx,
                    ))
            logger.info(f"PaddleOCR: {len(words)} words in {time.time()-t0:.2f}s")
            return words
        except Exception as e:
            logger.error(f"PaddleOCR error: {e}")
            return []


# ============================================================
# محرك EasyOCR
# ============================================================

class EasyOcrEngine(BaseOcrEngine):
    """
    EasyOCR — جيد للنصوص اللاتينية والمختلطة
    ===========================================
    - يدعم +80 لغة بما فيها العربية والإنجليزية
    - GPU optional
    - استخدام الذاكرة: ~500MB
    """

    name = "easyocr"
    description = "EasyOCR (80+ languages) — Good for Latin and mixed text"
    supports_arabic = True
    supports_latin = True
    supports_handwriting = True
    memory_mb = 500

    def __init__(self, langs=['ar', 'en']):
        self.langs = langs
        self._reader = None
        self._available = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import easyocr
            self._available = True
        except ImportError:
            self._available = False
            logger.warning("EasyOCR not installed. Install with: pip install easyocr")
        return self._available

    def _get_reader(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(self.langs, gpu=False)
        return self._reader

    def recognize(self, image_path: str) -> List[OcrWord]:
        reader = self._get_reader()
        t0 = time.time()

        try:
            results = reader.readtext(image_path)
            words = []
            for idx, (bbox, text, conf) in enumerate(results):
                # EasyOCR bbox is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] (points, not rect)
                bbox_list = [list(p) for p in bbox]
                words.append(OcrWord(
                    text=str(text),
                    confidence=float(conf),
                    bbox=bbox_list,
                    engine=self.name,
                    engine_idx=idx,
                ))
            logger.info(f"EasyOCR: {len(words)} words in {time.time()-t0:.2f}s")
            return words
        except Exception as e:
            logger.error(f"EasyOCR error: {e}")
            return []


# ============================================================
# محرك Tesseract
# ============================================================

class TesseractEngine(BaseOcrEngine):
    """
    Tesseract OCR — سريع وموثوق للمطبوع
    ======================================
    - الأسرع بين المحركات (بدون GPU)
    - الأفضل للنصوص المطبوعة الواضحة
    - usage: ~50MB
    """

    name = "tesseract"
    description = "Tesseract OCR — Fast and reliable for printed text"
    supports_arabic = True
    supports_latin = True
    supports_handwriting = False
    memory_mb = 50

    def __init__(self, langs='ara+eng'):
        self.langs = langs
        self._available = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import pytesseract
            # تحقق من تثبيت tesseract binary
            import shutil
            if not shutil.which('tesseract'):
                self._available = False
                logger.warning("tesseract binary not found. Install: apt install tesseract-ocr")
            else:
                self._available = True
        except ImportError:
            self._available = False
            logger.warning("pytesseract not installed. Install with: pip install pytesseract")
        return self._available

    def recognize(self, image_path: str) -> List[OcrWord]:
        import pytesseract
        from PIL import Image as PILImage
        t0 = time.time()

        try:
            img = PILImage.open(image_path)
            data = pytesseract.image_to_data(
                img, lang=self.langs,
                output_type=pytesseract.Output.DICT,
                config='--psm 6 --oem 3'
            )

            words = []
            n_boxes = len(data['text'])
            for idx in range(n_boxes):
                text = data['text'][idx].strip()
                conf = int(data['conf'][idx])
                if text and conf > 0:
                    # بناء bbox من الإحداثيات
                    x, y, w, h = data['left'][idx], data['top'][idx], data['width'][idx], data['height'][idx]
                    bbox = [
                        [x, y], [x + w, y],
                        [x + w, y + h], [x, y + h]
                    ]
                    words.append(OcrWord(
                        text=text,
                        confidence=conf / 100.0,
                        bbox=bbox,
                        engine=self.name,
                        engine_idx=idx,
                    ))

            logger.info(f"Tesseract: {len(words)} words in {time.time()-t0:.2f}s")
            return words
        except Exception as e:
            logger.error(f"Tesseract error: {e}")
            return []


# ============================================================
# محرك TrOCR
# ============================================================

class TrocrEngine(BaseOcrEngine):
    """
    TrOCR — Transformer-based OCR للخط اليدوي
    ==========================================
    - يعتمد على Vision Encoder-Decoder
    - الأفضل للخط اليدوي (الأفضل مع القصاصات)
    - استخدام الذاكرة: ~1.5GB
    - الاستراتيجية: يستقبل الصورة الكاملة أو القصاصات
    """

    name = "trocr"
    description = "TrOCR (Transformer) — Best for handwriting recognition"
    supports_arabic = True
    supports_latin = True
    supports_handwriting = True
    memory_mb = 1500

    def __init__(self, model_name="microsoft/trocr-base-printed"):
        self.model_name = model_name
        self._processor = None
        self._model = None
        self._available = None
        # النموذج المطبوع — أفضل للوصفات والتقارير الطبية
        self._ar_model_name = "microsoft/trocr-base-printed"

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            self._available = True
        except ImportError:
            self._available = False
            logger.warning("TrOCR dependencies not installed. Install with: pip install transformers torch sentencepiece")
        return self._available

    def _get_model(self):
        if self._model is None:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            # نموذج للنص المطبوع — مناسب للوصفات والتقارير الطبية
            self._processor = TrOCRProcessor.from_pretrained(
                "microsoft/trocr-base-printed"
            )
            self._model = VisionEncoderDecoderModel.from_pretrained(
                "microsoft/trocr-base-printed"
            )
            self._model.eval()

        return self._processor, self._model

    def recognize(self, image_path: str) -> List[OcrWord]:
        """
        TrOCR يعمل على مستوى السطر/القصاصة.
        نقسم الصورة إلى مناطق نصية ثم نتعرف عليها.
        """
        try:
            processor, model = self._get_model()
        except Exception as e:
            logger.error(f"TrOCR model loading failed: {e}")
            return []

        t0 = time.time()

        try:
            import torch
            from PIL import Image as PILImage

            img = PILImage.open(image_path).convert("RGB")

            # محاولة 1: الصورة الكاملة
            try:
                pixel_values = processor(img, return_tensors="pt").pixel_values
                with torch.no_grad():
                    generated_ids = model.generate(pixel_values)
                full_text = processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0].strip()

                if full_text:
                    w, h = img.size
                    # bbox كامل الصورة
                    bbox = [[0, 0], [w, 0], [w, h], [0, h]]
                    words_list = full_text.split()

                    # تقسيم ثقة تذكرية (TrOCR لا يعطي ثقة لكل كلمة)
                    base_conf = 0.75

                    # حساب عرض تقريبي لكل كلمة لتوزيع الـ bbox
                    result_words = []
                    if words_list:
                        total_chars = sum(len(w) for w in words_list)
                        x_offset = 0
                        for idx, word in enumerate(words_list):
                            word_width = int((len(word) / max(total_chars, 1)) * w)
                            bbox_w = [
                                [x_offset, 0],
                                [min(x_offset + word_width, w), 0],
                                [min(x_offset + word_width, w), h],
                                [x_offset, h],
                            ]
                            result_words.append(OcrWord(
                                text=word,
                                confidence=base_conf,
                                bbox=bbox_w,
                                engine=self.name,
                                engine_idx=idx,
                            ))
                            x_offset += word_width

                    logger.info(f"TrOCR: {len(result_words)} words in {time.time()-t0:.2f}s")
                    return result_words
            except Exception as e:
                logger.warning(f"TrOCR full image failed: {e}")

            logger.info(f"TrOCR: 0 words in {time.time()-t0:.2f}s")
            return []

        except Exception as e:
            logger.error(f"TrOCR error: {e}")
            return []


# ============================================================
# محرك Surya OCR
# ============================================================

class SuryaOcrEngine(BaseOcrEngine):
    """
    Surya OCR — محرك حديث عالي الدقة
    ==================================
    - من VikParuchuri (صاحب marker)
    - يدعم +90 لغة
    - كشف نص متقدم (text line detection)
    - استخدام الذاكرة: ~800MB
    """

    name = "surya"
    description = "Surya OCR — Modern high-accuracy engine (90+ languages)"
    supports_arabic = True
    supports_latin = True
    supports_handwriting = True
    memory_mb = 800

    def __init__(self, langs=['ar', 'en']):
        self.langs = langs
        self._available = None
        self._det_model = None
        self._rec_model = None

    def is_available(self) -> bool:
        """فحص عميق لتوفر Surya — يتأكد من إمكانية استيراد جميع الوحدات المطلوبة"""
        if self._available is not None:
            return self._available
        try:
            import surya.detection
            import surya.recognition
            from surya.model.detection.model import load_model as load_det
            from surya.model.recognition.model import load_model as load_rec
            self._available = True
            logger.info("Surya OCR: all modules available")
        except ImportError as e:
            self._available = False
            logger.warning(f"Surya OCR not fully installed: {e}")
        except Exception as e:
            self._available = False
            logger.warning(f"Surya OCR availability check failed: {e}")
        return self._available

    def _get_models(self):
        """تحميل نماذج Surya (lazy loading مع تخزين مؤقت)"""
        if self._det_model is not None and self._rec_model is not None:
            return self._det_model, self._rec_model

        try:
            from surya.model.detection.model import load_model as load_det
            from surya.model.recognition.model import load_model as load_rec
            self._det_model = load_det()
            self._rec_model = load_rec()
            logger.info("Surya OCR: models loaded successfully")
            return self._det_model, self._rec_model
        except Exception as e:
            logger.error(f"Surya OCR model loading failed: {e}")
            raise

    def recognize(self, image_path: str) -> List[OcrWord]:
        t0 = time.time()
        try:
            from PIL import Image as PILImage
            import torch

            det_model, rec_model = self._get_models()
            img_pil = PILImage.open(image_path).convert("RGB")

            # كشف خطوط النص
            from surya.detection import run_detection
            det_results = run_detection([img_pil], det_model)

            if not det_results or not det_results[0]:
                logger.info(f"Surya OCR: 0 words (no detections) in {time.time()-t0:.2f}s")
                return []

            lines = det_results[0] if isinstance(det_results[0], list) else det_results
            if not lines:
                logger.info(f"Surya OCR: 0 words (empty lines) in {time.time()-t0:.2f}s")
                return []

            # إعداد القصاصات
            crops = []
            for line in lines:
                b = line.bbox
                crops.append(img_pil.crop((b[0], b[1], b[2], b[3])))

            # التعرف على النص — v0.6.4 API
            words = []
            try:
                from surya.recognition import run_recognition
                rec_results = run_recognition(crops, [lines], rec_model, self.langs)

                for idx, result in enumerate(rec_results):
                    b = lines[idx].bbox
                    bbox = [
                        [float(b[0]), float(b[1])],
                        [float(b[2]), float(b[1])],
                        [float(b[2]), float(b[3])],
                        [float(b[0]), float(b[3])],
                    ]
                    text = getattr(result, 'text', '') or ''
                    conf = getattr(result, 'confidence', 0.7) or 0.7
                    if text.strip():
                        words.append(OcrWord(
                            text=text.strip(),
                            confidence=float(conf),
                            bbox=bbox,
                            engine=self.name,
                            engine_idx=idx,
                        ))
            except (ImportError, AttributeError, TypeError) as e:
                # Fallback: محاولة استخدام النموذج مباشرة مع المعالج
                logger.info(f"Surya: run_recognition failed ({e}), trying direct inference")
                try:
                    from surya.model.recognition.processor import load_processor as load_rec_proc
                    rec_processor = load_rec_proc()
                    for idx, crop in enumerate(crops):
                        b = lines[idx].bbox
                        bbox = [
                            [float(b[0]), float(b[1])],
                            [float(b[2]), float(b[1])],
                            [float(b[2]), float(b[3])],
                            [float(b[0]), float(b[3])],
                        ]
                        try:
                            pixel_values = rec_processor(crop, return_tensors="pt").pixel_values
                            with torch.no_grad():
                                generated = rec_model.generate(pixel_values)
                            text = rec_processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
                            if text:
                                words.append(OcrWord(
                                    text=text,
                                    confidence=0.7,
                                    bbox=bbox,
                                    engine=self.name,
                                    engine_idx=idx,
                                ))
                        except Exception as e2:
                            logger.warning(f"Surya line {idx} recognition failed: {e2}")
                except Exception as e3:
                    logger.error(f"Surya direct inference setup failed: {e3}")

            logger.info(f"Surya OCR: {len(words)} words in {time.time()-t0:.2f}s")
            return words

        except Exception as e:
            logger.error(f"Surya OCR error: {e}")
            return []


# ============================================================
# نظام التجمع (Ensemble)
# ============================================================

class EnsembleOCR:
    """
    نظام تجمع محركات OCR المتعددة
    ===============================

    يجمع نتائج عدة محركات ويدمجها باستخدام استراتيجيات ذكية.

    الاستخدام:
        ocr = EnsembleOCR(engines=['paddleocr', 'easyocr', 'tesseract'])
        result = ocr.process_image("document.jpg", strategy='majority_voting')

    الاستراتيجيات:
        - 'majority_voting': تصويت الأغلبية
        - 'confidence_weighted': متوسط مرجح بالثقة
        - 'levenshtein_consensus': أقرب نص بإجماع
        - 'best_single': أفضل نتيجة واحدة
    """

    ENGINE_MAP = {
        'paddleocr': PaddleOcrEngine,
        'easyocr': EasyOcrEngine,
        'tesseract': TesseractEngine,
        'trocr': TrocrEngine,
        'surya': SuryaOcrEngine,
    }

    ENGINE_DESCRIPTIONS = {
        'paddleocr': {
            'name': 'PaddleOCR',
            'icon': '🔷',
            'strengths': 'Arabic/English mixed text, handwriting',
            'memory': '300MB',
        },
        'easyocr': {
            'name': 'EasyOCR',
            'icon': '🟢',
            'strengths': '80+ languages, Latin text, mixed',
            'memory': '500MB',
        },
        'tesseract': {
            'name': 'Tesseract',
            'icon': '🔵',
            'strengths': 'Fast, printed text, reliable',
            'memory': '50MB',
        },
        'trocr': {
            'name': 'TrOCR',
            'icon': '🟠',
            'strengths': 'Handwriting, Transformer-based',
            'memory': '1500MB',
        },
        'surya': {
            'name': 'Surya OCR',
            'icon': '🟣',
            'strengths': 'Modern, 90+ languages, high accuracy',
            'memory': '800MB',
        },
    }

    def __init__(self, engines=None, strategy='majority_voting', confidence_threshold=0.3):
        """
        Args:
            engines: قائمة المحركات ('all' أو قائمة محددة)
            strategy: استراتيجية الدمج
            confidence_threshold: الحد الأدنى للثقة لقبول كلمة
        """
        if engines == 'all' or engines is None:
            self.engine_names = list(self.ENGINE_MAP.keys())
        else:
            self.engine_names = [e for e in engines if e in self.ENGINE_MAP]

        self.strategy = strategy
        self.confidence_threshold = confidence_threshold
        self._engines: Dict[str, BaseOcrEngine] = {}
        self._initialized = False

    def _init_engines(self):
        """تحميل المحركات المطلوبة (lazy loading)"""
        if self._initialized:
            return

        for name in self.engine_names:
            engine_class = self.ENGINE_MAP[name]
            self._engines[name] = engine_class()

        self._initialized = True

    def get_available_engines(self) -> Dict[str, bool]:
        """التحقق من توفر كل محرك"""
        self._init_engines()
        return {
            name: engine.is_available()
            for name, engine in self._engines.items()
        }

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """حساب مسافة Levenshtein"""
        s1, s2 = s1.lower().strip(), s2.lower().strip()
        if s1 == s2:
            return 0
        if not s1:
            return len(s2)
        if not s2:
            return len(s1)

        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + cost
                )
        return dp[m][n]

    def _compute_iou(self, bbox1, bbox2) -> float:
        """حساب تقاطع BBox"""
        def to_rect(bbox):
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            return [min(xs), min(ys), max(xs), max(ys)]

        r1 = to_rect(bbox1)
        r2 = to_rect(bbox2)

        x_left = max(r1[0], r2[0])
        y_top = max(r1[1], r2[1])
        x_right = min(r1[2], r2[2])
        y_bottom = min(r1[3], r2[3])

        if x_right <= x_left or y_bottom <= y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = (r1[2] - r1[0]) * (r1[3] - r1[1])
        area2 = (r2[2] - r2[0]) * (r2[3] - r2[1])
        union = area1 + area2 - intersection

        return intersection / max(union, 1e-6)

    def _align_words(self, engine_results: Dict[str, EngineResult]) -> List[List[OcrWord]]:
        """
        محاذاة الكلمات من محركات مختلفة بناءً على تقاطع BBox.
        يعيد قائمة مجموعات: كل مجموعة تحتوي كلمات متقاربة مكانياً.
        """
        all_words = []
        for name, result in engine_results.items():
            for word in result.words:
                all_words.append(word)

        if not all_words:
            return []

        # فرز حسب موقع y ثم x (قراءة من أعلى لأسفل، من يسار ليمين)
        all_words.sort(key=lambda w: (
            sum(p[1] for p in w.bbox) / 4,
            sum(p[0] for p in w.bbox) / 4,
        ))

        groups = []
        used = [False] * len(all_words)

        for i, word in enumerate(all_words):
            if used[i]:
                continue

            group = [word]
            used[i] = True

            for j in range(i + 1, len(all_words)):
                if used[j]:
                    continue

                # تحقق من التقاطع مع أي كلمة في المجموعة
                matches = False
                for g_word in group:
                    iou = self._compute_iou(g_word.bbox, all_words[j].bbox)
                    # عتبة宽松 — نقبل حتى تقاطع ضعيف
                    if iou > 0.05:
                        matches = True
                        break
                    # تحقق أيضاً من المسافة
                    cx1 = sum(p[0] for p in g_word.bbox) / 4
                    cy1 = sum(p[1] for p in g_word.bbox) / 4
                    cx2 = sum(p[0] for p in all_words[j].bbox) / 4
                    cy2 = sum(p[1] for p in all_words[j].bbox) / 4

                    # نفس السطر تقريباً + قريب أفقياً
                    if abs(cy1 - cy2) < max(
                        (g_word.bbox[2][1] - g_word.bbox[0][1]) * 0.5,
                        (all_words[j].bbox[2][1] - all_words[j].bbox[0][1]) * 0.5,
                        20,
                    ):
                        dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
                        if dist < max(100, max(
                            g_word.bbox[2][0] - g_word.bbox[0][0],
                            all_words[j].bbox[2][0] - all_words[j].bbox[0][0],
                        )):
                            matches = True
                            break

                if matches:
                    group.append(all_words[j])
                    used[j] = True

            groups.append(group)

        return groups

    def _merge_majority_voting(self, group: List[OcrWord]) -> Optional[EnsembleWord]:
        """
        استراتيجية تصويت الأغلبية:
        - نص حصل على أكبر عدد من الأصوات يفوز
        - تعادل → النص ذو أعلى ثقة مجمعة يفوز
        """
        if not group:
            return None

        if len(group) == 1:
            w = group[0]
            return EnsembleWord(
                text=w.text,
                confidence=w.confidence,
                bbox=w.bbox,
                engines_used=[w.engine],
                engine_votes={w.engine: w.text},
                agreement_count=1,
                strategy='single_engine',
            )

        # تصويت
        text_votes: Dict[str, List[Tuple[str, float]]] = {}  # text -> [(engine, conf)]
        for w in group:
            # تطبيع النص للمقارنة
            normalized = w.text.lower().strip()
            # البحث عن نص مشابه موجود
            matched_key = None
            for key in text_votes:
                if self._levenshtein_distance(normalized, key) <= max(1, len(normalized) * 0.2):
                    matched_key = key
                    break

            if matched_key:
                text_votes[matched_key].append((w.engine, w.confidence))
            else:
                text_votes[normalized] = [(w.engine, w.confidence)]

        # اختيار الفائز
        best_text = max(text_votes, key=lambda k: len(text_votes[k]))
        votes = text_votes[best_text]

        # حساب الثقة المجمعة
        avg_conf = sum(c for _, c in votes) / len(votes)
        bonus = (len(votes) - 1) * 0.05  # مكافأة للإجماع
        final_conf = min(0.99, avg_conf + bonus)

        # اختيار أفضل bbox
        best_word = max((w for w in group if w.text.lower().strip() == best_text or
                         self._levenshtein_distance(w.text.lower().strip(), best_text) <= 1),
                        key=lambda w: w.confidence, default=group[0])

        return EnsembleWord(
            text=best_word.text,
            confidence=final_conf,
            bbox=best_word.bbox,
            engines_used=[e for e, _ in votes],
            engine_votes={w.engine: w.text for w in group},
            agreement_count=len(votes),
            strategy='majority_voting',
        )

    def _merge_confidence_weighted(self, group: List[OcrWord]) -> Optional[EnsembleWord]:
        """
        استراتيجية المتوسط المرجح بالثقة:
        - كل نص يحصل على وزن حسب ثقة محركه
        - النص ذو أعلى وزن يفوز
        """
        if not group:
            return None

        if len(group) == 1:
            w = group[0]
            return EnsembleWord(
                text=w.text, confidence=w.confidence, bbox=w.bbox,
                engines_used=[w.engine], engine_votes={w.engine: w.text},
                agreement_count=1, strategy='single_engine',
            )

        # تجميع النصوص المتشابهة
        clusters: List[Dict] = []  # [{text, engines, total_weight, best_bbox}]

        for w in group:
            matched = False
            for cluster in clusters:
                if self._levenshtein_distance(w.text.lower().strip(), cluster['text'].lower().strip()) <= max(1, len(w.text) * 0.2):
                    cluster['engines'].append(w.engine)
                    cluster['total_weight'] += w.confidence
                    if w.confidence > cluster.get('max_conf', 0):
                        cluster['best_bbox'] = w.bbox
                        cluster['max_conf'] = w.confidence
                    matched = True
                    break

            if not matched:
                clusters.append({
                    'text': w.text,
                    'engines': [w.engine],
                    'total_weight': w.confidence,
                    'best_bbox': w.bbox,
                    'max_conf': w.confidence,
                })

        # اختيار الكتلة الأعلى وزناً
        best = max(clusters, key=lambda c: c['total_weight'])

        return EnsembleWord(
            text=best['text'],
            confidence=best['total_weight'] / len(best['engines']),
            bbox=best['best_bbox'],
            engines_used=best['engines'],
            engine_votes={w.engine: w.text for w in group},
            agreement_count=len(best['engines']),
            strategy='confidence_weighted',
        )

    def _merge_levenshtein_consensus(self, group: List[OcrWord]) -> Optional[EnsembleWord]:
        """
        استراتيجية إجماع Levenshtein:
        - النص الذي يقل مسافته عن جميع النصوص الأخرى يفوز
        - يعمل جيداً مع الأخطاء الصغيرة
        """
        if not group:
            return None

        if len(group) == 1:
            w = group[0]
            return EnsembleWord(
                text=w.text, confidence=w.confidence, bbox=w.bbox,
                engines_used=[w.engine], engine_votes={w.engine: w.text},
                agreement_count=1, strategy='single_engine',
            )

        texts = [w.text for w in group]
        confs = [w.confidence for w in group]

        # حساب إجمالي المسافة لكل نص
        best_idx = 0
        min_total_dist = float('inf')

        for i, text_i in enumerate(texts):
            total_dist = 0
            for j, text_j in enumerate(texts):
                if i != j:
                    total_dist += self._levenshtein_distance(text_i, text_j)

            # تعديل بالثقة (نص عالي الثقة + مسافة منخفضة = الأفضل)
            adjusted = total_dist / max(confs[i], 0.1)

            if adjusted < min_total_dist:
                min_total_dist = adjusted
                best_idx = i

        best_word = group[best_idx]

        # حساب الثقة النهائية (متوسط + مكافأة الاتفاق)
        avg_conf = sum(confs) / len(confs)
        agreement_bonus = sum(1 for t in texts if self._levenshtein_distance(t, best_word.text) <= 1) * 0.03
        final_conf = min(0.99, avg_conf + agreement_bonus)

        return EnsembleWord(
            text=best_word.text,
            confidence=final_conf,
            bbox=best_word.bbox,
            engines_used=[w.engine for w in group],
            engine_votes={w.engine: w.text for w in group},
            agreement_count=sum(1 for t in texts if self._levenshtein_distance(t, best_word.text) <= 1),
            strategy='levenshtein_consensus',
        )

    def _merge_best_single(self, group: List[OcrWord]) -> Optional[EnsembleWord]:
        """
        استراتيجية أفضل نتيجة واحدة:
        - نختار النص الأعلى ثقة فقط
        """
        if not group:
            return None

        best = max(group, key=lambda w: w.confidence)

        return EnsembleWord(
            text=best.text,
            confidence=best.confidence,
            bbox=best.bbox,
            engines_used=[best.engine],
            engine_votes={w.engine: w.text for w in group},
            agreement_count=sum(1 for w in group if w.text.lower().strip() == best.text.lower().strip()),
            strategy='best_single',
        )

    def _merge_group(self, group: List[OcrWord]) -> Optional[EnsembleWord]:
        """دمج مجموعة كلمات باستخدام الاستراتيجية المحددة"""
        if self.strategy == 'majority_voting':
            return self._merge_majority_voting(group)
        elif self.strategy == 'confidence_weighted':
            return self._merge_confidence_weighted(group)
        elif self.strategy == 'levenshtein_consensus':
            return self._merge_levenshtein_consensus(group)
        elif self.strategy == 'best_single':
            return self._merge_best_single(group)
        else:
            return self._merge_majority_voting(group)

    def process_image(self, image_path: str, strategy=None) -> EnsembleResult:
        """
        معالجة صورة بكل المحركات ودمج النتائج.

        Args:
            image_path: مسار الصورة
            strategy: استراتيجية الدمج (اختياري، يُستخدم الإعداد الافتراضي)

        Returns:
            EnsembleResult مع النتائج المدمجة
        """
        self._init_engines()

        if strategy:
            self.strategy = strategy

        t_start = time.time()
        engine_results: Dict[str, EngineResult] = {}
        active_engines = []

        # تشغيل كل محرك
        for name in self.engine_names:
            engine = self._engines.get(name)

            if engine is None or not engine.is_available():
                engine_results[name] = EngineResult(
                    engine_name=name,
                    available=False,
                    error="Not available" if engine else "Unknown engine",
                )
                logger.warning(f"Engine '{name}' not available, skipping")
                continue

            t0 = time.time()
            try:
                words = engine.recognize(image_path)
                elapsed = time.time() - t0

                engine_results[name] = EngineResult(
                    engine_name=name,
                    words=words,
                    processing_time=elapsed,
                    available=True,
                )
                active_engines.append(name)
                logger.info(f"{name}: {len(words)} words in {elapsed:.2f}s")

            except Exception as e:
                elapsed = time.time() - t0
                engine_results[name] = EngineResult(
                    engine_name=name,
                    available=False,
                    processing_time=elapsed,
                    error=str(e),
                )
                logger.error(f"{name} failed: {e}")

        total_time = time.time() - t_start

        # محاذاة ودمج الكلمات
        ensemble_result = EnsembleResult(
            engine_results=engine_results,
            total_time=total_time,
            strategy=self.strategy,
            engines_active=active_engines,
        )

        if active_engines:
            # محاذاة الكلمات
            groups = self._align_words(engine_results)

            logger.info(f"Aligned into {len(groups)} groups")

            for group in groups:
                merged = self._merge_group(group)
                if merged and merged.confidence >= self.confidence_threshold:
                    ensemble_result.words.append(merged)

            # فرز حسب الموقع (أعلى→أسفل، يسار→يمين)
            ensemble_result.words.sort(key=lambda w: (
                sum(p[1] for p in w.bbox) / 4,
                sum(p[0] for p in w.bbox) / 4,
            ))

        return ensemble_result

    def process_and_compare(self, image_path: str) -> EnsembleResult:
        """
        معالجة مع مقارنة تفصيلية لكل محرك.
        يعرض نص كل محرك لكل منطقة.
        """
        return self.process_image(image_path, strategy=self.strategy)


# ============================================================
# تشغيل من سطر الأوامر
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="نظام تجمع OCR متعدد المحركات",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
مثال:
  python ensemble_ocr.py --image scan.jpg --engines all --strategy majority_voting
  python ensemble_ocr.py --image doc.png --engines paddleocr easyocr tesseract
  python ensemble_ocr.py --image note.jpg --strategy confidence_weighted --json
        """
    )
    parser.add_argument('--image', required=True, help='مسار الصورة')
    parser.add_argument(
        '--engines', nargs='+',
        default=['paddleocr', 'easyocr', 'tesseract'],
        choices=['paddleocr', 'easyocr', 'tesseract', 'trocr', 'surya', 'all'],
        help='المحركات المطلوبة'
    )
    parser.add_argument(
        '--strategy',
        choices=['majority_voting', 'confidence_weighted', 'levenshtein_consensus', 'best_single'],
        default='majority_voting',
        help='استراتيجية الدمج'
    )
    parser.add_argument('--json', action='store_true', help='إخراج بصيغة JSON')
    parser.add_argument('--threshold', type=float, default=0.3, help='حد الثقة الأدنى')

    args = parser.parse_args()

    engines = 'all' if 'all' in args.engines else args.engines
    ocr = EnsembleOCR(engines=engines, strategy=args.strategy, confidence_threshold=args.threshold)

    # عرض المحركات المتوفرة
    available = ocr.get_available_engines()
    print("\n" + "=" * 60)
    print("  نظام تجمع OCR متعدد المحركات")
    print("=" * 60)
    print(f"  الاستراتيجية: {args.strategy}")
    print(f"  المحركات المطلوبة: {engines if engines != 'all' else 'all'}")
    print("\n  حالة المحركات:")
    for name, avail in available.items():
        info = EnsembleOCR.ENGINE_DESCRIPTIONS.get(name, {})
        status = "✅ متاح" if avail else "❌ غير متاح"
        print(f"    {info.get('icon', '?')} {info.get('name', name):20s} {status}")
    print("=" * 60 + "\n")

    # معالجة الصورة
    result = ocr.process_image(args.image)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(f"\n📊 النتائج:")
        print(f"   إجمالي الوقت: {result.total_time:.2f}s")
        print(f"   المحركات النشطة: {result.engines_active}")
        print(f"   عدد الكلمات المدمجة: {len(result.words)}")

        print(f"\n📋 نتائج كل محرك:")
        for name, er in result.engine_results.items():
            info = EnsembleOCR.ENGINE_DESCRIPTIONS.get(name, {})
            status = f"{er.word_count} كلمة" if er.available else f"غير متاح ({er.error})"
            print(f"   {info.get('icon', '?')} {info.get('name', name):20s} {status} ({er.processing_time:.2f}s)")

        print(f"\n📝 الكلمات المدمجة ({result.strategy}):")
        print("-" * 80)
        for i, word in enumerate(result.words):
            engines_str = ', '.join(word.engines_used)
            votes_str = ' | '.join(f"{e}: {t}" for e, t in word.engine_votes.items())
            print(f"   {i+1:3d}. [{word.confidence:.0%}] {word.text}")
            print(f"        محركات: {engines_str} | اتفاق: {word.agreement_count}")
            if len(word.engines_used) > 1:
                print(f"        تصويت: {votes_str}")
            print()

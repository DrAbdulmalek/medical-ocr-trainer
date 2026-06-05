"""
evaluation — metrics and benchmark integration for medical-ocr-trainer
========================================================================

This package provides:

- **Local metrics** (always available, via ``evaluation.metrics``):
    - ``OCRMetrics.character_error_rate()`` — Character Error Rate
    - ``OCRMetrics.word_error_rate()`` — Word Error Rate
    - ``OCRMetrics.medical_term_accuracy()`` — Medical term accuracy
    - ``OCRMetrics.evaluate()`` — Comprehensive evaluation report

- **Convenience functions** (thin wrappers, always available):
    - ``get_cer(ocr_text, gt_text)`` — Character Error Rate
    - ``get_wer(ocr_text, gt_text)`` — Word Error Rate
    - ``get_medical_accuracy(ocr_text, gt_text)`` — Medical term accuracy

- **Benchmark Bridge** (graceful fallback when benchmarks package is missing):
    - ``BenchmarkBridge`` — wraps the ``medical-ocr-benchmarks`` suite

- **Dataset management** (via ``evaluation.dataset_manager``):
    - ``DatasetManager`` — load, validate, split, and manage golden datasets

- **Benchmark runner** (via ``evaluation.benchmark``):
    - ``BenchmarkRunner`` — run evaluations against golden datasets
    - ``BenchmarkResult`` — container for results with markdown/JSON export

Usage::

    # Local metrics
    from evaluation.metrics import OCRMetrics
    metrics = OCRMetrics()
    report = metrics.evaluate("ground truth", "ocr hypothesis")

    # Convenience functions
    from evaluation import get_cer, get_wer, get_medical_accuracy

    # Full benchmark bridge (requires medical-ocr-benchmarks package)
    from evaluation import BenchmarkBridge
    bridge = BenchmarkBridge()
    results = bridge.run_benchmark(ocr_results, ground_truth)
"""

from evaluation.benchmark_bridge import (
    BenchmarkBridge,
    get_cer,
    get_wer,
    get_medical_accuracy,
    BENCHMARKS_AVAILABLE,
)
from evaluation.metrics import OCRMetrics
from evaluation.dataset_manager import DatasetManager
from evaluation.benchmark import BenchmarkRunner, BenchmarkResult

__all__ = [
    # Benchmark bridge
    "BenchmarkBridge",
    "get_cer",
    "get_wer",
    "get_medical_accuracy",
    "BENCHMARKS_AVAILABLE",
    # Local evaluation
    "OCRMetrics",
    "DatasetManager",
    "BenchmarkRunner",
    "BenchmarkResult",
]

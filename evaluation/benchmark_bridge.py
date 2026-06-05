"""
Benchmark Bridge — connects medical-ocr-trainer to medical-ocr-benchmarks
=========================================================================
Provides a unified interface that wraps the external ``medical-ocr-benchmarks``
package and falls back to the local ``evaluation.metrics.OCRMetrics`` when
the package is not installed.

Classes:
    BenchmarkBridge — main wrapper around the benchmark suite API

Standalone functions (always available, no external dependency):
    get_cer(ocr_text, gt_text)       — Character Error Rate
    get_wer(ocr_text, gt_text)       — Word Error Rate
    get_medical_accuracy(ocr_text, gt_text) — Medical term accuracy

Usage:
    from evaluation.benchmark_bridge import BenchmarkBridge, get_cer, get_wer

    # Always available (pure local computation)
    cer = get_cer("paracetamol 500mg", "paracetamol 500 mg")

    # With benchmarks package installed
    bridge = BenchmarkBridge()
    results = bridge.run_benchmark(ocr_results, ground_truth)
"""

import os
import re
import json
import logging
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

# Import local evaluation metrics as the fallback
from evaluation.metrics import OCRMetrics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attempt to import the external benchmarks package — graceful fallback
# ---------------------------------------------------------------------------
BENCHMARKS_AVAILABLE = False
ExtBenchmarkRunner = None
ExtDatasetManager = None
ExtReportGenerator = None
ExtThresholdChecker = None

try:
    from medical_ocr_benchmarks import (
        BenchmarkRunner as ExtBenchmarkRunner,
        DatasetManager as ExtDatasetManager,
        ReportGenerator as ExtReportGenerator,
        ThresholdChecker as ExtThresholdChecker,
    )
    BENCHMARKS_AVAILABLE = True
    logger.info("medical-ocr-benchmarks package loaded successfully")
except ImportError:
    logger.info(
        "medical-ocr-benchmarks not installed. "
        "BenchmarkBridge will use local-only metrics. "
        "Install with: pip install 'medical-ocr-benchmarks "
        "@ git+https://github.com/DrAbdulmalek/medical-ocr-benchmarks.git'"
    )


# ---------------------------------------------------------------------------
# Medical term dictionaries (supplementary to evaluation.metrics)
# ---------------------------------------------------------------------------
LOCAL_CLINICAL_TERMS = {
    # English
    "paracetamol", "ibuprofen", "amoxicillin", "metformin", "insulin",
    "aspirin", "prednisone", "prednisolone", "dexamethasone",
    "ciprofloxacin", "azithromycin", "omeprazole", "enalapril",
    "digoxin", "warfarin", "heparin", "morphine", "fentanyl",
    "diabetes", "hypertension", "fracture", "pneumonia", "asthma",
    "osteoporosis", "arthritis", "appendicitis",
    "ORIF", "biopsy", "MRI", "CT", "ECG", "IV", "IM", "SC",
    "AVN", "LMWH", "NG", "NPO", "PRN", "BID", "TID", "QID",
    "PO", "SL", "PR", "Topical",
    # Arabic
    "باراسيتامول", "إيبوبروفين", "أموكسيسيلين", "ميتفورمين",
    "أنسولين", "أسبرين",
    "السكري", "ارتفاع ضغط الدم", "كسر", "التهاب رئوي", "ربو",
}

# Default OCRMetrics instance used by standalone functions
_default_metrics = OCRMetrics(medical_terms=list(LOCAL_CLINICAL_TERMS))


# ---------------------------------------------------------------------------
# Standalone metric functions (always available)
# ---------------------------------------------------------------------------

def get_cer(ocr_text: str, gt_text: str) -> float:
    """
    Compute Character Error Rate (CER) between OCR output and ground truth.

    CER = (S + D + I) / N
    where S=substitutions, D=deletions, I=insertions, N=ground truth length

    Args:
        ocr_text: The text produced by OCR
        gt_text: The ground truth reference text

    Returns:
        CER as a float between 0.0 (perfect) and 1.0+ (total failure)
    """
    return _default_metrics.character_error_rate(gt_text, ocr_text)


def get_wer(ocr_text: str, gt_text: str) -> float:
    """
    Compute Word Error Rate (WER) between OCR output and ground truth.

    WER = (S + D + I) / N
    where S=substitutions, D=deletions, I=insertions, N=number of GT words

    Args:
        ocr_text: The text produced by OCR
        gt_text: The ground truth reference text

    Returns:
        WER as a float between 0.0 (perfect) and 1.0+ (total failure)
    """
    return _default_metrics.word_error_rate(gt_text, ocr_text)


def get_medical_accuracy(ocr_text: str, gt_text: str) -> float:
    """
    Compute medical term accuracy — the fraction of medical terms in the
    ground truth that are correctly recognised in the OCR output.

    Uses the built-in medical term dictionary from ``evaluation.metrics``
    supplemented with Arabic clinical terms.

    Args:
        ocr_text: The text produced by OCR
        gt_text: The ground truth reference text

    Returns:
        Accuracy as a float between 0.0 and 1.0
    """
    result = _default_metrics.medical_term_accuracy(gt_text, ocr_text)
    return result.get("accuracy", 1.0)


# ---------------------------------------------------------------------------
# BenchmarkBridge class
# ---------------------------------------------------------------------------

class BenchmarkBridge:
    """
    Unified bridge between medical-ocr-trainer and medical-ocr-benchmarks.

    When the ``medical-ocr-benchmarks`` package is installed, this class
    delegates to its ``BenchmarkRunner``, ``DatasetManager``,
    ``ReportGenerator``, and ``ThresholdChecker`` for standardised
    evaluation against golden benchmark datasets.

    When the package is **not** installed, the bridge falls back to the
    local ``evaluation.metrics.OCRMetrics``, logging a one-time warning.

    Args:
        cache_dir: Directory for caching benchmark datasets
            (default: ``data/benchmarks``)
        threshold_cer: Maximum acceptable CER threshold (default: 0.15)
        threshold_wer: Maximum acceptable WER threshold (default: 0.25)
        threshold_medical: Minimum acceptable medical term accuracy
            (default: 0.90)
        medical_terms: Optional list of medical terms for accuracy
            evaluation. If not provided, the default clinical dictionary
            is used.
    """

    def __init__(
        self,
        cache_dir: str = "data/benchmarks",
        threshold_cer: float = 0.15,
        threshold_wer: float = 0.25,
        threshold_medical: float = 0.90,
        medical_terms: Optional[list[str]] = None,
    ):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.threshold_cer = threshold_cer
        self.threshold_wer = threshold_wer
        self.threshold_medical = threshold_medical

        terms = medical_terms or list(LOCAL_CLINICAL_TERMS)
        self._local_metrics = OCRMetrics(medical_terms=terms)

        # Internal references (populated when benchmarks package is available)
        self._runner: Optional[Any] = None
        self._dataset_manager: Optional[Any] = None
        self._report_generator: Optional[Any] = None
        self._threshold_checker: Optional[Any] = None

        if BENCHMARKS_AVAILABLE:
            self._init_benchmark_components()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_benchmark_components(self):
        """Initialise benchmark suite components when the package is available."""
        try:
            self._dataset_manager = ExtDatasetManager(cache_dir=self.cache_dir)
            self._report_generator = ExtReportGenerator(output_dir=self.cache_dir)
            self._threshold_checker = ExtThresholdChecker(
                cer_threshold=self.threshold_cer,
                wer_threshold=self.threshold_wer,
                medical_accuracy_threshold=self.threshold_medical,
            )
            self._runner = ExtBenchmarkRunner(
                dataset_manager=self._dataset_manager,
                report_generator=self._report_generator,
                threshold_checker=self._threshold_checker,
            )
            logger.info("BenchmarkBridge: all benchmark components initialised")
        except Exception as exc:
            logger.warning(
                f"BenchmarkBridge: failed to initialise benchmark components "
                f"({exc}). Falling back to local-only metrics."
            )
            self._runner = None

    @property
    def is_benchmarks_available(self) -> bool:
        """Whether the external benchmarks package is available and initialised."""
        return self._runner is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_benchmark(
        self,
        ocr_results: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
        dataset_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run a full benchmark evaluation comparing OCR results to ground truth.

        When the benchmarks package is installed, this delegates to
        ``BenchmarkRunner.run()`` which uses the standardised golden datasets
        and produces a comprehensive report.

        Without the benchmarks package, it computes local-only CER, WER,
        and medical term accuracy and returns them in the same dict format.

        Args:
            ocr_results: List of dicts, each containing at least ``"text"``.
                         May also include ``"image_path"``, ``"confidence"``,
                         ``"engine"``, etc.
            ground_truth: List of dicts, each containing at least ``"text"``.
                          May also include ``"image_path"``, ``"source"``, etc.
            dataset_name: Optional name to tag the evaluation run.

        Returns:
            Dict with keys:
                - ``cer``: float — aggregate Character Error Rate
                - ``wer``: float — aggregate Word Error Rate
                - ``medical_accuracy``: float — medical term accuracy
                - ``per_sample``: list of per-sample scores
                - ``thresholds_passed``: dict of bools
                - ``report_path``: str or None — path to generated report
                - ``benchmark_available``: bool
        """
        # Match OCR results to ground truth by index
        per_sample = []
        all_cer, all_wer, all_med = [], [], []

        n_pairs = min(len(ocr_results), len(ground_truth))
        for i in range(n_pairs):
            ocr_text = str(ocr_results[i].get("text", ""))
            gt_text = str(ground_truth[i].get("text", ""))

            sample = {
                "index": i,
                "ocr_text": ocr_text,
                "gt_text": gt_text,
                "image_path": ground_truth[i].get(
                    "image_path",
                    ocr_results[i].get("image_path", ""),
                ),
            }
            sample["cer"] = get_cer(ocr_text, gt_text)
            sample["wer"] = get_wer(ocr_text, gt_text)
            sample["medical_accuracy"] = get_medical_accuracy(ocr_text, gt_text)
            sample["confidence"] = ocr_results[i].get("confidence", None)
            sample["engine"] = ocr_results[i].get("engine", "")

            all_cer.append(sample["cer"])
            all_wer.append(sample["wer"])
            all_med.append(sample["medical_accuracy"])
            per_sample.append(sample)

        # Aggregate
        aggregate = {
            "cer": sum(all_cer) / len(all_cer) if all_cer else 0.0,
            "wer": sum(all_wer) / len(all_wer) if all_wer else 0.0,
            "medical_accuracy": sum(all_med) / len(all_med) if all_med else 1.0,
            "sample_count": n_pairs,
            "per_sample": per_sample,
            "thresholds_passed": {
                "cer": (aggregate["cer"] if all_cer else 0.0) <= self.threshold_cer,
                "wer": (aggregate["wer"] if all_wer else 0.0) <= self.threshold_wer,
                "medical_accuracy": (
                    aggregate["medical_accuracy"] if all_med else 1.0
                ) >= self.threshold_medical,
            },
            "benchmark_available": self.is_benchmarks_available,
            "dataset_name": dataset_name,
            "timestamp": datetime.now().isoformat(),
        }

        # If benchmarks package is available, enhance with remote dataset eval
        if self.is_benchmarks_available and self._runner:
            try:
                enhanced = self._runner.run(
                    predictions=ocr_results,
                    references=ground_truth,
                    dataset_name=dataset_name,
                )
                if isinstance(enhanced, dict):
                    for key, value in enhanced.items():
                        if key not in aggregate:
                            aggregate[key] = value
                    aggregate["benchmark_available"] = True
            except Exception as exc:
                logger.warning(
                    f"BenchmarkBridge: remote benchmark run failed ({exc}), "
                    "using local-only results"
                )
                aggregate["benchmark_run_error"] = str(exc)

        # Save a local report
        report_path = os.path.join(
            self.cache_dir,
            f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )
        os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(aggregate, f, ensure_ascii=False, indent=2)
            aggregate["report_path"] = report_path
        except Exception as exc:
            logger.warning(f"BenchmarkBridge: could not save report ({exc})")

        return aggregate

    def list_available_datasets(self) -> List[Dict[str, str]]:
        """
        List benchmark datasets available from the benchmarks package.

        Returns an empty list when the package is not installed.
        """
        if not self.is_benchmarks_available or not self._dataset_manager:
            return []

        try:
            datasets = self._dataset_manager.list_datasets()
            return datasets if isinstance(datasets, list) else []
        except Exception as exc:
            logger.warning(
                f"BenchmarkBridge: failed to list datasets ({exc})"
            )
            return []

    def download_dataset(self, dataset_name: str) -> Optional[str]:
        """
        Download a benchmark dataset from the benchmarks package.

        Returns the local path to the downloaded dataset, or None on failure.
        """
        if not self.is_benchmarks_available or not self._dataset_manager:
            logger.warning(
                "BenchmarkBridge: cannot download — "
                "benchmarks package not available"
            )
            return None

        try:
            path = self._dataset_manager.download(
                dataset_name, output_dir=self.cache_dir
            )
            logger.info(
                f"BenchmarkBridge: dataset '{dataset_name}' "
                f"downloaded to {path}"
            )
            return str(path)
        except Exception as exc:
            logger.warning(
                f"BenchmarkBridge: failed to download dataset "
                f"'{dataset_name}' ({exc})"
            )
            return None

    def check_thresholds(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check evaluation results against configured thresholds.

        Args:
            results: Dict returned by ``run_benchmark()``.

        Returns:
            Dict with ``passed`` (bool) and ``details`` (dict per metric).
        """
        details = {
            "cer": {
                "value": results.get("cer", 0.0),
                "threshold": self.threshold_cer,
                "passed": results.get("cer", 0.0) <= self.threshold_cer,
            },
            "wer": {
                "value": results.get("wer", 0.0),
                "threshold": self.threshold_wer,
                "passed": results.get("wer", 0.0) <= self.threshold_wer,
            },
            "medical_accuracy": {
                "value": results.get("medical_accuracy", 0.0),
                "threshold": self.threshold_medical,
                "passed": (
                    results.get("medical_accuracy", 0.0)
                    >= self.threshold_medical
                ),
            },
        }

        all_passed = all(d["passed"] for d in details.values())
        return {"passed": all_passed, "details": details}

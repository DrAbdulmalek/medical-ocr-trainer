#!/usr/bin/env python3
"""
run_eval_with_benchmarks — CLI evaluation script using the benchmark suite
=============================================================================

Evaluates OCR output against ground truth using both local metrics (CER, WER,
medical term accuracy) and, when installed, the ``medical-ocr-benchmarks``
package for standardised golden-dataset evaluation.

Supports three report formats:
    - json     — machine-readable JSON
    - markdown — human-readable Markdown table
    - html     — standalone HTML report

Usage examples:
    # Evaluate local directories (text files)
    python evaluation/run_eval_with_benchmarks.py --gt-dir data/gold/ --ocr-dir exports/

    # Custom thresholds and Markdown output
    python evaluation/run_eval_with_benchmarks.py \
        --gt-dir data/gold/ --ocr-dir exports/ \
        --report-format markdown --threshold-cer 0.10

    # HTML report saved to file
    python evaluation/run_eval_with_benchmarks.py \
        --gt-dir data/gold/ --ocr-dir exports/ \
        --report-format html --output reports/benchmark.html

    # Use a specific benchmark dataset (requires medical-ocr-benchmarks)
    python evaluation/run_eval_with_benchmarks.py \
        --benchmark-dataset medical_prescriptions_v1 --gt-dir data/gold/

    # List available benchmark datasets
    python evaluation/run_eval_with_benchmarks.py --list-datasets

    # Use JSONL input format (from export_training.py)
    python evaluation/run_eval_with_benchmarks.py \
        --gt-dir data/gold/ --ocr-dir exports/ --format jsonl

    # Evaluate a golden JSON dataset
    python evaluation/run_eval_with_benchmarks.py \
        --golden-dataset data/golden/sample_eval_set.json --engine paddleocr
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path so the evaluation package can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.benchmark_bridge import (
    BenchmarkBridge,
    get_cer,
    get_wer,
    get_medical_accuracy,
    BENCHMARKS_AVAILABLE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_eval_with_benchmarks")


# ---------------------------------------------------------------------------
# File discovery helpers
# ---------------------------------------------------------------------------

def _find_text_files(directory: str) -> List[Path]:
    """Find all .txt files in a directory (recursive)."""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    return sorted(dir_path.rglob("*.txt"))


def _pair_files(
    gt_dir: str,
    ocr_dir: str,
) -> List[Dict[str, str]]:
    """
    Pair ground-truth and OCR text files by matching filename.

    Each file in gt_dir should have a corresponding file in ocr_dir
    with the same basename. Missing pairs are skipped with a warning.
    """
    gt_files = _find_text_files(gt_dir)
    ocr_files = {p.stem: p for p in _find_text_files(ocr_dir)}

    pairs = []
    for gt_path in gt_files:
        if gt_path.stem in ocr_files:
            pairs.append({
                "gt_path": str(gt_path),
                "ocr_path": str(ocr_files[gt_path.stem]),
                "basename": gt_path.stem,
            })
        else:
            logger.warning(f"No OCR file found for ground truth: {gt_path.name}")

    return pairs


def _load_jsonl_pairs(gt_dir: str, ocr_dir: str) -> List[Dict[str, Any]]:
    """
    Load paired JSONL files — each line is a JSON record with at least a
    ``text`` field.
    """
    gt_files = sorted(Path(gt_dir).glob("*.jsonl"))
    ocr_files = sorted(Path(ocr_dir).glob("*.jsonl"))

    pairs_map: Dict[str, Dict[str, list]] = {}

    for f in gt_files:
        records = []
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        pairs_map.setdefault(f.stem, {})["gt"] = records

    for f in ocr_files:
        records = []
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        pairs_map.setdefault(f.stem, {})["ocr"] = records

    result = []
    for stem, data in pairs_map.items():
        if "gt" in data and "ocr" in data:
            for gt_rec, ocr_rec in zip(data["gt"], data["ocr"]):
                result.append({
                    "basename": stem,
                    "gt": gt_rec,
                    "ocr": ocr_rec,
                })
    return result


# ---------------------------------------------------------------------------
# Report formatters
# ---------------------------------------------------------------------------

def _format_json(results: Dict[str, Any]) -> str:
    return json.dumps(results, ensure_ascii=False, indent=2)


def _format_markdown(results: Dict[str, Any]) -> str:
    lines = [
        "# Benchmark Evaluation Report",
        "",
        f"**Timestamp:** {results.get('timestamp', 'N/A')}",
        f"**Dataset:** {results.get('dataset_name', 'local')}",
        f"**Samples evaluated:** {results.get('sample_count', 0)}",
        "**Benchmarks package:** "
        f"{'✅ Available' if results.get('benchmark_available') else '❌ Not installed (local metrics only)'}",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Value | Threshold | Passed |",
        "|--------|-------|-----------|--------|",
        f"| CER | {results['cer']:.4f} | ≤ 0.15 | "
        f"{'✅' if results.get('thresholds_passed', {}).get('cer') else '❌'} |",
        f"| WER | {results['wer']:.4f} | ≤ 0.25 | "
        f"{'✅' if results.get('thresholds_passed', {}).get('wer') else '❌'} |",
        f"| Medical Accuracy | {results['medical_accuracy']:.4f} | ≥ 0.90 | "
        f"{'✅' if results.get('thresholds_passed', {}).get('medical_accuracy') else '❌'} |",
        "",
    ]

    per_sample = results.get("per_sample", [])
    if per_sample:
        lines.append("## Per-Sample Details")
        lines.append("")
        lines.append(
            "| # | File | CER | WER | Med Acc | Confidence | Engine |"
        )
        lines.append("|---|------|-----|-----|---------|------------|--------|")
        for s in per_sample[:50]:
            conf = (
                f"{s.get('confidence', 0):.2f}"
                if s.get("confidence") is not None
                else "—"
            )
            lines.append(
                f"| {s['index']} "
                f"| {s.get('basename', s.get('image_path', '—'))} "
                f"| {s['cer']:.4f} | {s['wer']:.4f} "
                f"| {s['medical_accuracy']:.4f} "
                f"| {conf} | {s.get('engine', '—')} |"
            )
        if len(per_sample) > 50:
            lines.append(
                f"| ... | ({len(per_sample) - 50} more samples) "
                f"| | | | | |"
            )

    report_path = results.get("report_path")
    if report_path:
        lines.append(f"\n**Full JSON report:** `{report_path}`")

    return "\n".join(lines) + "\n"


def _format_html(results: Dict[str, Any]) -> str:
    ts = results.get("timestamp", "N/A")
    ds = results.get("dataset_name", "local")
    sc = results.get("sample_count", 0)
    bench_status = (
        "Available"
        if results.get("benchmark_available")
        else "Not installed (local metrics only)"
    )

    cer_pass = results.get("thresholds_passed", {}).get("cer", False)
    wer_pass = results.get("thresholds_passed", {}).get("wer", False)
    med_pass = results.get("thresholds_passed", {}).get(
        "medical_accuracy", False
    )

    status_color = (
        "green" if (cer_pass and wer_pass and med_pass) else "red"
    )
    status_text = (
        "All Thresholds Passed"
        if (cer_pass and wer_pass and med_pass)
        else "Some Thresholds Failed"
    )

    rows_html = ""
    for s in results.get("per_sample", [])[:100]:
        conf = (
            f"{s.get('confidence', 0):.2f}"
            if s.get("confidence") is not None
            else "—"
        )
        name = s.get(
            "basename", s.get("image_path", f"Sample {s['index']}")
        )
        rows_html += (
            f"<tr>"
            f"<td>{s['index']}</td>"
            f"<td>{name}</td>"
            f"<td>{s['cer']:.4f}</td>"
            f"<td>{s['wer']:.4f}</td>"
            f"<td>{s['medical_accuracy']:.4f}</td>"
            f"<td>{conf}</td>"
            f"<td>{s.get('engine', '—')}</td>"
            f"</tr>\n"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Benchmark Evaluation Report — {ds}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 2rem; color: #1a1a1a; }}
  h1 {{ color: #111; }} h2 {{ color: #333; margin-top: 2rem; }}
  .meta {{ color: #666; font-size: 0.9rem; margin-bottom: 1rem; }}
  .status-green {{ padding: 0.5rem 1rem; border-radius: 4px; font-weight: bold; display: inline-block; margin-bottom: 1.5rem; background: #d4edda; color: #155724; }}
  .status-red {{ padding: 0.5rem 1rem; border-radius: 4px; font-weight: bold; display: inline-block; margin-bottom: 1.5rem; background: #f8d7da; color: #721c24; }}
  .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1rem 0 2rem; }}
  .metric-card {{ padding: 1rem; border-radius: 8px; background: #f8f9fa; border-left: 4px solid #dee2e6; }}
  .metric-card.pass {{ border-left-color: #28a745; }}
  .metric-card.fail {{ border-left-color: #dc3545; }}
  .metric-label {{ font-size: 0.85rem; color: #666; text-transform: uppercase; }}
  .metric-value {{ font-size: 1.8rem; font-weight: bold; margin-top: 0.25rem; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
  th, td {{ padding: 0.5rem 0.75rem; text-align: left; border-bottom: 1px solid #dee2e6; font-size: 0.9rem; }}
  th {{ background: #f1f3f5; font-weight: 600; position: sticky; top: 0; }}
  tr:hover {{ background: #f8f9fa; }}
</style>
</head>
<body>
<h1>🏥 Benchmark Evaluation Report</h1>
<div class="meta">
  <strong>Timestamp:</strong> {ts} &nbsp;|&nbsp;
  <strong>Dataset:</strong> {ds} &nbsp;|&nbsp;
  <strong>Samples:</strong> {sc} &nbsp;|&nbsp;
  <strong>Benchmarks Package:</strong> {bench_status}
</div>
<div class="status-{status_color}">{status_text}</div>

<h2>Aggregate Metrics</h2>
<div class="metrics">
  <div class="metric-card {'pass' if cer_pass else 'fail'}">
    <div class="metric-label">Character Error Rate (CER)</div>
    <div class="metric-value">{results['cer']:.4f}</div>
    <div class="metric-label">threshold ≤ 0.15</div>
  </div>
  <div class="metric-card {'pass' if wer_pass else 'fail'}">
    <div class="metric-label">Word Error Rate (WER)</div>
    <div class="metric-value">{results['wer']:.4f}</div>
    <div class="metric-label">threshold ≤ 0.25</div>
  </div>
  <div class="metric-card {'pass' if med_pass else 'fail'}">
    <div class="metric-label">Medical Term Accuracy</div>
    <div class="metric-value">{results['medical_accuracy']:.4f}</div>
    <div class="metric-label">threshold ≥ 0.90</div>
  </div>
</div>

<h2>Per-Sample Details</h2>
<table>
<thead><tr><th>#</th><th>File</th><th>CER</th><th>WER</th><th>Med Acc</th><th>Confidence</th><th>Engine</th></tr></thead>
<tbody>
{rows_html}
</tbody>
</table>

</body>
</html>"""


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate OCR output against ground truth using the benchmark suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--gt-dir",
        type=str,
        help="Directory containing ground truth text/JSONL files",
    )
    parser.add_argument(
        "--ocr-dir",
        type=str,
        help="Directory containing OCR output text/JSONL files",
    )
    parser.add_argument(
        "--benchmark-dataset",
        type=str,
        default=None,
        help="Name of a benchmark dataset to download and use "
        "(requires medical-ocr-benchmarks)",
    )
    parser.add_argument(
        "--golden-dataset",
        type=str,
        default=None,
        help="Path to a golden evaluation JSON/CSV dataset "
        "(uses local BenchmarkRunner)",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="evaluated",
        help="OCR engine name for reporting (default: evaluated)",
    )
    parser.add_argument(
        "--report-format",
        type=str,
        choices=["json", "markdown", "html"],
        default="json",
        help="Output report format (default: json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--threshold-cer",
        type=float,
        default=0.15,
        help="Maximum acceptable CER (default: 0.15)",
    )
    parser.add_argument(
        "--threshold-wer",
        type=float,
        default=0.25,
        help="Maximum acceptable WER (default: 0.25)",
    )
    parser.add_argument(
        "--threshold-medical",
        type=float,
        default=0.90,
        help="Minimum acceptable medical accuracy (default: 0.90)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/benchmarks",
        help="Cache directory for benchmark datasets "
        "(default: data/benchmarks)",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available benchmark datasets and exit",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "jsonl"],
        default="text",
        help="Input file format: text (.txt) or jsonl (.jsonl) "
        "(default: text)",
    )

    args = parser.parse_args()

    # --- Initialise bridge ---
    bridge = BenchmarkBridge(
        cache_dir=args.cache_dir,
        threshold_cer=args.threshold_cer,
        threshold_wer=args.threshold_wer,
        threshold_medical=args.threshold_medical,
    )

    # --- Golden dataset mode (use local BenchmarkRunner) ---
    if args.golden_dataset:
        from evaluation.benchmark import BenchmarkRunner

        runner = BenchmarkRunner()
        result = runner.run(
            args.golden_dataset, engine_name=args.engine
        )

        if args.report_format == "json":
            output_str = result.to_json()
        else:
            output_str = result.to_markdown()

        if args.output:
            os.makedirs(
                os.path.dirname(os.path.abspath(args.output)), exist_ok=True
            )
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output_str)
            print(f"\nReport written to: {args.output}")
        else:
            print(output_str)
        return

    # --- List datasets mode ---
    if args.list_datasets:
        datasets = bridge.list_available_datasets()
        if not datasets:
            print("No benchmark datasets available.")
            if not BENCHMARKS_AVAILABLE:
                print(
                    "Install medical-ocr-benchmarks to access remote datasets:"
                )
                print(
                    "  pip install 'medical-ocr-benchmarks "
                    "@ git+https://github.com/DrAbdulmalek/"
                    "medical-ocr-benchmarks.git'"
                )
            sys.exit(0)
        print(f"Available datasets ({len(datasets)}):")
        for ds in datasets:
            if isinstance(ds, dict):
                print(
                    f"  • {ds.get('name', ds)} "
                    f"— {ds.get('description', '')}"
                )
            else:
                print(f"  • {ds}")
        sys.exit(0)

    # --- Download remote dataset if requested ---
    if args.benchmark_dataset:
        print(f"Downloading benchmark dataset: {args.benchmark_dataset}")
        dataset_path = bridge.download_dataset(args.benchmark_dataset)
        if dataset_path:
            print(f"  Downloaded to: {dataset_path}")
            if not args.gt_dir:
                args.gt_dir = dataset_path
        else:
            print(
                "  ⚠️  Failed to download dataset. "
                "Falling back to local directories."
            )
            if not args.gt_dir:
                print(
                    "ERROR: --gt-dir is required when dataset download fails."
                )
                sys.exit(1)

    # --- Validate inputs ---
    if not args.gt_dir:
        parser.error(
            "--gt-dir is required (or use --benchmark-dataset "
            "to download one)"
        )

    gt_path = Path(args.gt_dir)
    if not gt_path.exists():
        print(f"ERROR: Ground truth directory not found: {args.gt_dir}")
        sys.exit(1)

    # --- Load data ---
    ocr_results: List[Dict[str, Any]] = []
    ground_truth: List[Dict[str, Any]] = []

    if args.format == "jsonl":
        pairs = _load_jsonl_pairs(args.gt_dir, args.ocr_dir or "")
        if not pairs:
            print(
                "ERROR: No paired JSONL records found. "
                "Ensure matching filenames in --gt-dir and --ocr-dir."
            )
            sys.exit(1)
        for pair in pairs:
            ground_truth.append(pair["gt"])
            ocr_results.append(pair["ocr"])
    else:
        # Text file mode
        if not args.ocr_dir:
            parser.error("--ocr-dir is required for text file mode")

        file_pairs = _pair_files(args.gt_dir, args.ocr_dir)
        if not file_pairs:
            print(
                "ERROR: No matching file pairs found between "
                "--gt-dir and --ocr-dir."
            )
            sys.exit(1)

        for pair in file_pairs:
            with open(pair["gt_path"], "r", encoding="utf-8") as f:
                gt_text = f.read().strip()
            with open(pair["ocr_path"], "r", encoding="utf-8") as f:
                ocr_text = f.read().strip()
            ground_truth.append({
                "text": gt_text,
                "image_path": pair["gt_path"],
            })
            ocr_results.append({
                "text": ocr_text,
                "image_path": pair["ocr_path"],
            })

    print(f"Evaluating {len(ground_truth)} samples...")

    # --- Run benchmark ---
    results = bridge.run_benchmark(
        ocr_results=ocr_results,
        ground_truth=ground_truth,
        dataset_name=args.benchmark_dataset or os.path.basename(args.gt_dir),
    )

    # --- Format output ---
    formatters = {
        "json": _format_json,
        "markdown": _format_markdown,
        "html": _format_html,
    }
    output_str = formatters[args.report_format](results)

    # --- Output ---
    if args.output:
        os.makedirs(
            os.path.dirname(os.path.abspath(args.output)), exist_ok=True
        )
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_str)
        print(f"\nReport written to: {args.output}")
    else:
        print(output_str)

    # --- Summary to stderr ---
    tp = results.get("thresholds_passed", {})
    all_ok = all(tp.values())
    if all_ok:
        print(
            f"\n✅ All thresholds passed! "
            f"CER={results['cer']:.4f}, "
            f"WER={results['wer']:.4f}, "
            f"MedAcc={results['medical_accuracy']:.4f}"
        )
    else:
        print(f"\n❌ Some thresholds failed:")
        print(
            f"   CER  = {results['cer']:.4f} "
            f"(≤{args.threshold_cer})  "
            f"{'✅' if tp.get('cer') else '❌'}"
        )
        print(
            f"   WER  = {results['wer']:.4f} "
            f"(≤{args.threshold_wer})  "
            f"{'✅' if tp.get('wer') else '❌'}"
        )
        print(
            f"   MedAcc = {results['medical_accuracy']:.4f} "
            f"(≥{args.threshold_medical})  "
            f"{'✅' if tp.get('medical_accuracy') else '❌'}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

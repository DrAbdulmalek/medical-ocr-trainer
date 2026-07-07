> ## Repository Archived

This repository has been **archived**. Its code and functionality have been consolidated into the
[omni-medical-suite](https://github.com/DrAbdulmalek/omni-medical-suite) monorepo.

| Detail | Value |
|--------|-------|
| **Suite location** | `apps/trainer-ui/` |
| **Original purpose** | Human-in-the-Loop data collection with 5 OCR engines |
| **Archived on** | 2026-07-07 |
| **Status** | Read-only - no further updates will be made here |

Please use [omni-medical-suite](https://github.com/DrAbdulmalek/omni-medical-suite) for all development.

---

---
title: Medical OCR Trainer
emoji: 🏥
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
tags:
  - ocr
  - medical
  - handwriting
  - arabic
  - paddleocr
  - easyocr
  - tesseract
  - trocr
  - surya
  - ensemble
  - streamlit
---

# Medical OCR Trainer

**Interactive tool for training and correcting medical handwriting OCR**

5 OCR engines (PaddleOCR, EasyOCR, Tesseract, TrOCR, Surya) with smart ensemble merging and a human-in-the-loop correction pipeline.

---

## ⚡ Quick Install

Choose your installation level based on available resources:

```bash
# Clone
git clone https://github.com/DrAbdulmalek/medical-ocr-trainer.git
cd medical-ocr-trainer

# Option 1: Lite (~350MB) — PaddleOCR + Tesseract only
pip install -e ".[lite]"

# Option 2: Medium (~850MB) — Add EasyOCR
pip install -e ".[medium]"

# Option 3: TrOCR (~2.3GB) — Add TrOCR (needs PyTorch)
pip install -e ".[trocr]"

# Option 4: Surya (~1.6GB) — Add Surya OCR (GPU recommended)
pip install -e ".[surya]"

# Option 5: Full (~3.1GB+) — All engines
pip install -e ".[full]"

# System dependency (Tesseract)
# Ubuntu/Debian: sudo apt install tesseract-ocr
# macOS: brew install tesseract

# Run
streamlit run app.py
```

| Mode | Engines | RAM Needed | Disk | Best For |
|------|---------|-----------|------|----------|
| **Lite** | PaddleOCR + Tesseract | 2GB | ~350MB | Quick testing, CI/CD |
| **Medium** | + EasyOCR | 4GB | ~850MB | Standard training |
| **TrOCR** | + TrOCR | 8GB | ~2.3GB | Advanced handwriting |
| **Surya** | + Surya OCR | 8GB | ~1.6GB | Multi-language layout |
| **Full** | All 5 engines | 16GB+ | ~3.1GB+ | Benchmarking, production |

## Features

- **5-Engine Ensemble OCR**: PaddleOCR + EasyOCR + Tesseract + TrOCR + Surya OCR running simultaneously with smart result merging
- **4 Merging Strategies**: Majority voting, confidence-weighted averaging, Levenshtein consensus, and best-single selection
- **Upload & OCR**: Upload scanned medical notes (JPG/PNG) with automatic image preprocessing (contrast enhancement + sharpening)
- **Interactive Correction**: Edit recognized words in a Streamlit data editor, sorted by confidence (lowest first), with per-engine vote visibility
- **Engine Comparison**: Detailed per-engine performance comparison with word counts, processing times, and visual charts
- **Auto Crop Generation**: Word crops are automatically generated with padding when corrections are saved
- **Multi-layer Data Filtering**: Classify corrections as gold/pending/rejected based on confidence, error frequency, clinical importance, and medical dictionary matching
- **Multi-format Export**: Export training data as JSONL (HuggingFace), CSV, or HuggingFace image folder format with ensemble metadata
- **Real-time Metrics**: Track CER, WER, confidence distribution, inter-engine agreement, and correction progress
- **Arabic Support**: Full RTL support with script detection (Arabic/Latin/Numeric/Mixed)

## Architecture

```
medical_ocr_trainer/
├── app.py                 # Main Streamlit application (upload, ensemble OCR, correct, compare)
├── ensemble_ocr.py         # 5-engine ensemble system with 4 merging strategies
├── data_filter.py         # Automated correction quality filter (5-layer classification)
├── export_training.py     # Multi-format training data exporter (JSONL/CSV/HuggingFace)
├── evaluation/              # Evaluation metrics and benchmark integration
│   ├── __init__.py          # Package exports
│   ├── metrics.py            # CER, WER, medical term accuracy (OCRMetrics)
│   ├── benchmark.py          # BenchmarkRunner against golden datasets
│   ├── dataset_manager.py     # Dataset loading, validation, splitting
│   └── benchmark_bridge.py   # Bridge to medical-ocr-benchmarks (graceful fallback)
│   └── run_eval_with_benchmarks.py  # CLI evaluation script
├── requirements.txt       # Python dependencies (all 5 OCR engines)
├── uploads/                # Uploaded medical note images (gitignored)
├── crops/                  # Auto-generated word crops for training (gitignored)
├── data/                   # SQLite database (corrections.db) (gitignored)
└── exports/                # Exported training datasets (gitignored)
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### 3. Use the Tool

1. **Select engines** — Choose which OCR engines to enable in the sidebar
2. **Choose strategy** — Pick a merging strategy (majority voting, weighted, etc.)
3. **Upload** a scanned medical note (JPG/PNG)
4. **Review** ensemble results — words sorted by confidence with per-engine votes visible
5. **Correct** wrong words in the interactive editor
6. **Save** — corrections stored, crops auto-generated, engine logs recorded
7. **Compare** — View per-engine performance in the comparison tab
8. **Export** training data with full ensemble metadata

## Data Filtering Pipeline

The `data_filter.py` module classifies corrections through 5 layers:

| Layer | Criterion | Outcome |
|-------|-----------|---------|
| 1. Formal Check | Empty/non-text | Rejected |
| 2. Medical Dictionary | Full match | Gold |
| 3. Confidence Logic | Low confidence + correction | Gold |
| 4. Error Consensus | Multiple identical corrections | Gold |
| 5. Clinical Priority | Drug/diagnosis terms | Gold |

### Run Filters

```bash
# Classify all corrections
python data_filter.py

# Apply filters to database
python data_filter.py --apply

# Export gold samples
python data_filter.py --export

# Custom thresholds
python data_filter.py --threshold 0.7 --min-agree 3
```

## Export Training Data

```bash
# JSONL (for HuggingFace Datasets)
python export_training.py --format jsonl

# CSV (for Excel/Pandas)
python export_training.py --format csv

# HuggingFace image folder format
python export_training.py --format huggingface

# Gold samples only
python export_training.py --format jsonl --gold-only

# View statistics
python export_training.py --stats
```

## Database Schema

### `images` — Uploaded document metadata
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| filename | TEXT | Original file name |
| path | TEXT | Storage path |
| width/height | INTEGER | Image dimensions |
| created_at | TIMESTAMP | Upload time |

### `words` — Extracted words and corrections
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| image_id | INTEGER FK | Reference to image |
| bbox | TEXT | JSON bounding box coordinates |
| predicted_text | TEXT | OCR output |
| confidence | REAL | OCR confidence score |
| corrected_text | TEXT | Human-corrected text |
| crop_path | TEXT | Path to word crop image |
| is_corrected | BOOLEAN | Has been corrected |
| review_status | TEXT | pending/approved/rejected/gold |
| is_gold_standard | BOOLEAN | High-quality training sample |
| script_class | TEXT | arabic/latin/numeric/mixed |
| correction_count | INTEGER | Number of corrections |

### `correction_history` — Full audit trail
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| word_id | INTEGER FK | Reference to word |
| old_text | TEXT | Previous text |
| new_text | TEXT | New corrected text |
| confidence_at_correction | REAL | Confidence when corrected |
| created_at | TIMESTAMP | Correction time |

## Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | >= 1.28.0 | Web UI framework |
| paddleocr | >= 2.7.0 | Arabic/English OCR engine |
| paddlepaddle | >= 2.5.0 | Deep learning framework |
| easyocr | >= 1.7.0 | Multi-language OCR (80+ langs) |
| pytesseract | >= 0.3.10 | Fast printed text OCR |
| transformers | >= 4.35.0 | TrOCR (Transformer OCR) |
| torch | >= 2.0.0 | PyTorch for TrOCR |
| sentencepiece | >= 0.1.99 | Tokenizer for TrOCR |
| surya-ocr | >= 0.5.0 | Modern high-accuracy OCR |
| Pillow | >= 10.0.0 | Image processing |
| pandas | >= 2.0.0 | Data manipulation |
| numpy | >= 1.24.0 | Numerical operations |

### Minimal Installation (skip heavy engines)

```bash
# Core only (PaddleOCR + Streamlit)
pip install streamlit paddleocr paddlepaddle Pillow pandas numpy

# Add Tesseract (also needs: apt install tesseract-ocr)
pip install pytesseract

# Add EasyOCR (~500MB extra)
pip install easyocr

# Add TrOCR (~1.5GB extra)
pip install transformers torch sentencepiece

# Add Surya OCR (~800MB extra)
pip install surya-ocr
```

## Ensemble System

The `ensemble_ocr.py` module provides a unified interface for running multiple OCR engines and merging their results:

### Engines

| Engine | Strengths | Memory | Languages |
|--------|-----------|--------|----------|
| PaddleOCR | Arabic/English mixed, handwriting | ~300MB | 80+ |
| EasyOCR | Latin text, mixed documents | ~500MB | 80+ |
| Tesseract | Fast, printed text | ~50MB | 100+ |
| TrOCR | Handwriting recognition | ~1.5GB | Latin |
| Surya OCR | Modern, high accuracy | ~800MB | 90+ |

### Merging Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `majority_voting` | Text with most engine votes wins | General use, high accuracy |
| `confidence_weighted` | Weighted average by confidence | Mixed quality engines |
| `levenshtein_consensus` | Most similar text to all engines | Small OCR errors |
| `best_single` | Highest confidence result only | One strong engine |

### Command Line Usage

```bash
# Run all engines with majority voting
python ensemble_ocr.py --image scan.jpg --engines all --strategy majority_voting

# Run specific engines
python ensemble_ocr.py --image doc.png --engines paddleocr easyocr tesseract --strategy confidence_weighted

# JSON output
python ensemble_ocr.py --image note.jpg --engines all --strategy majority_voting --json
```

## Benchmark Integration

The trainer integrates with [medical-ocr-benchmarks](https://github.com/DrAbdulmalek/medical-ocr-benchmarks) for standardised evaluation against golden datasets. The integration is **optional** — the trainer works fully without it, using built-in local metrics with a graceful fallback when the benchmarks package is not installed.

### How It Works

| Component | Description |
|-----------|-------------|
| `BenchmarkBridge` | Unified wrapper around the benchmark suite; delegates to `BenchmarkRunner`, `DatasetManager`, `ReportGenerator`, `ThresholdChecker` when available |
| `get_cer()` | Character Error Rate (always available locally) |
| `get_wer()` | Word Error Rate (always available locally) |
| `get_medical_accuracy()` | Medical term accuracy using a clinical dictionary |

### CLI Evaluation

The `evaluation/run_eval_with_benchmarks.py` script provides a command-line interface for running evaluations:

```bash
# Evaluate local text directories
python evaluation/run_eval_with_benchmarks.py --gt-dir data/gold/ --ocr-dir exports/

# Markdown report with custom thresholds
python evaluation/run_eval_with_benchmarks.py \
    --gt-dir data/gold/ --ocr-dir exports/ \
    --report-format markdown --threshold-cer 0.10

# HTML report saved to file
python evaluation/run_eval_with_benchmarks.py \
    --gt-dir data/gold/ --ocr-dir exports/ \
    --report-format html --output reports/benchmark.html

# Evaluate a golden JSON dataset
python evaluation/run_eval_with_benchmarks.py \
    --golden-dataset data/golden/sample_eval_set.json --engine paddleocr

# Evaluate against a remote benchmark dataset (requires benchmarks package)
python evaluation/run_eval_with_benchmarks.py \
    --benchmark-dataset medical_prescriptions_v1 --gt-dir data/gold/

# List available benchmark datasets
python evaluation/run_eval_with_benchmarks.py --list-datasets

# Use JSONL input format (from export_training.py)
python evaluation/run_eval_with_benchmarks.py \
    --gt-dir data/gold/ --ocr-dir exports/ --format jsonl
```

### Programmatic Usage

```python
from evaluation import BenchmarkBridge, get_cer, get_wer, get_medical_accuracy

# Quick local metrics (no benchmarks package needed)
cer = get_cer("paracetamol 500mg", "paracetamol 500 mg")
wer = get_wer("the patient has", "the patient has diabetes")

# Full benchmark evaluation
bridge = BenchmarkBridge(threshold_cer=0.15, threshold_wer=0.25)
results = bridge.run_benchmark(
    ocr_results=[{"text": "paracetamol 500mg", "confidence": 0.85}],
    ground_truth=[{"text": "paracetamol 500 mg"}],
    dataset_name="prescriptions_v1",
)
print(f"CER: {results['cer']:.4f}")
print(f"Thresholds passed: {results['thresholds_passed']}")
```

### Report Formats

| Format | Use Case | Output |
|--------|----------|--------|
| `json` | CI/CD pipelines, automation | Machine-readable JSON to stdout or file |
| `markdown` | Documentation, PRs | Human-readable Markdown table |
| `html` | Sharing, archiving | Standalone HTML report with styled cards |

### Default Thresholds

| Metric | Default | Meaning |
|--------|---------|---------|
| CER | ≤ 0.15 | At most 15% character errors |
| WER | ≤ 0.25 | At most 25% word errors |
| Medical Accuracy | ≥ 0.90 | At least 90% of medical terms correct |

## Related Projects

- [medical-ocr-benchmarks](https://github.com/DrAbdulmalek/medical-ocr-benchmarks) — Standardised golden datasets and evaluation suite for medical OCR
- [medical-handwriting-ocr](https://github.com/DrAbdulmalek/medical-handwriting-ocr) — Full production OCR platform (FastAPI + React + K8s)

## License

MIT License

---

<p align="center">
  <strong>Dr. Abdulmalek</strong><br>
  Medical Handwriting Recognition
</p>


---

## Repository Status

| Field | Value |
|-------|-------|
| **Role** | Data Collection & Training Tool |
| **Status** | Active Development |
| **Layer** | Applications (Product) |
| **Priority** | Medium |
| **Relation** | Feeds training data to medical-handwriting-ocr and omni-medical-suite |

## Who Should Use This

- ML engineers building **medical handwriting training datasets**
- Clinical teams needing **human-in-the-loop correction** workflows
- Researchers comparing **multiple OCR engines** on medical documents
- Projects needing **active learning** data pipelines

## What This Produces

```
┌─────────────────────────┐
│   medical-ocr-trainer    │
│                         │
│   Input: Scanned images │
│   Output:               │
│   ├── JSONL (HF format) │──▶ medical-handwriting-ocr (fine-tuning)
│   ├── CSV datasets      │──▶ omni-medical-suite (evaluation)
│   ├── HF Image Folders  │──▶ HuggingFace Hub
│   └── Gold/Pending/      │
│       Rejected labels   │
└─────────────────────────┘
```

## When to Use This vs Other Repos

| Need | Repository |
|------|-----------|
| Collect & correct training data | **This repo** (medical-ocr-trainer) |
| Production OCR deployment | [medical-handwriting-ocr](https://github.com/DrAbdulmalek/medical-handwriting-ocr) |
| OCR correction engine | [medical-ocr-postprocessor](https://github.com/DrAbdulmalek/medical-ocr-postprocessor) |
| Unified platform | [omni-medical-suite](https://github.com/DrAbdulmalek/omni-medical-suite) |
| HF demo deployment | [medical-ocr-trainer-hf](https://github.com/DrAbdulmalek/medical-ocr-trainer-hf) |

## Related Repositories

| Repo | Role | Status |
|------|------|--------|
| [medical-handwriting-ocr](https://github.com/DrAbdulmalek/medical-handwriting-ocr) | Production OCR | Active |
| [medical-ocr-trainer-hf](https://github.com/DrAbdulmalek/medical-ocr-trainer-hf) | HF Deployment | Deployment |
| [omni-medical-suite](https://github.com/DrAbdulmalek/omni-medical-suite) | Main Platform | Active |
| [medical-ocr-postprocessor](https://github.com/DrAbdulmalek/medical-ocr-postprocessor) | Core Correction Engine | Active |

**License: MIT** — Dr. Abdulmalek Tamer Al-husseini

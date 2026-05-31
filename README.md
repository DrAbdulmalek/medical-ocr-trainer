# Medical OCR Trainer

**Interactive tool for training and correcting medical handwriting OCR**

5 OCR engines (PaddleOCR, EasyOCR, Tesseract, TrOCR, Surya) with smart ensemble merging and a human-in-the-loop correction pipeline.

---

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

## Related Projects

- [medical-handwriting-ocr](https://github.com/DrAbdulmalek/medical-handwriting-ocr) — Full production OCR platform (FastAPI + React + K8s)

## License

MIT License

---

<p align="center">
  <strong>Dr. Abdulmalek</strong><br>
  Medical Handwriting Recognition
</p>

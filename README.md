# Medical OCR Trainer

**Interactive tool for training and correcting medical handwriting OCR**

Arabic & English medical note recognition powered by PaddleOCR with a human-in-the-loop correction pipeline.

---

## Features

- **Upload & OCR**: Upload scanned medical notes (JPG/PNG) and run PaddleOCR with automatic image preprocessing (contrast enhancement + sharpening)
- **Interactive Correction**: Edit recognized words in a Streamlit data editor, sorted by confidence (lowest first)
- **Auto Crop Generation**: Word crops are automatically generated with padding when corrections are saved
- **Multi-layer Data Filtering**: Classify corrections as gold/pending/rejected based on confidence, error frequency, clinical importance, and medical dictionary matching
- **Multi-format Export**: Export training data as JSONL (HuggingFace), CSV, or HuggingFace image folder format
- **Real-time Metrics**: Track CER, WER, confidence distribution, and correction progress
- **Arabic Support**: Full RTL support with script detection (Arabic/Latin/Numeric/Mixed)

## Architecture

```
medical_ocr_trainer/
├── app.py                 # Main Streamlit application (upload, OCR, correct, stats)
├── data_filter.py         # Automated correction quality filter (5-layer classification)
├── export_training.py     # Multi-format training data exporter (JSONL/CSV/HuggingFace)
├── requirements.txt       # Python dependencies
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

1. **Upload** a scanned medical note (JPG/PNG)
2. **Review** OCR results — words are sorted by confidence (lowest first)
3. **Correct** wrong words in the interactive editor
4. **Save** — corrections are stored and word crops are auto-generated
5. **Export** training data when ready

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
| Pillow | >= 10.0.0 | Image processing |
| pandas | >= 2.0.0 | Data manipulation |

## Related Projects

- [medical-handwriting-ocr](https://github.com/DrAbdulmalek/medical-handwriting-ocr) — Full production OCR platform (FastAPI + React + K8s)

## License

MIT License

---

<p align="center">
  <strong>Dr. Abdulmalek</strong><br>
  Medical Handwriting Recognition
</p>

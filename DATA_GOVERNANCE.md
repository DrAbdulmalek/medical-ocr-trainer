# Data Governance Policy — سياسة حوكمة البيانات

> Applies to all data in `medical-ocr-trainer`: uploads, crops, exports, and database.

---

## 1. Data Directories

| Directory | Purpose | Retention | Backup |
|-----------|---------|-----------|--------|
| `uploads/` | Raw uploaded medical document images | 90 days, then auto-delete | Manual export before expiry |
| `crops/` | Auto-cropped word images for training | 90 days, then auto-delete | Export to training dataset before expiry |
| `exports/` | Exported training data (JSONL, CSV, Parquet) | Indefinite (versioned) | Git-tracked or cloud backup |
| `data/corrections.db` | SQLite database with all corrections | Indefinite | Weekly backup recommended |
| `data/golden/` | Golden evaluation datasets | Indefinite | Git-tracked |
| `data/raw/` | Raw imported data | 90 days, then auto-delete | Export before expiry |

## 2. Data Retention Rules

### Auto-Delete Schedule (Recommended)

```
uploads/    → Delete files older than 90 days
crops/      → Delete files older than 90 days
data/raw/   → Delete files older than 90 days
exports/    → KEEP indefinitely (these are your training assets)
data/golden/ → KEEP indefinitely (evaluation baseline)
data/corrections.db → KEEP indefinitely (correction history)
```

### Implementation

Create a cron job or scheduled task:

```bash
# Run weekly
#!/bin/bash
# cleanup_old_data.sh

UPLOAD_DIR="./uploads"
CROPS_DIR="./crops"
RAW_DIR="./data/raw"
DAYS=90

echo "Cleaning files older than $DAYS days..."

find "$UPLOAD_DIR" -type f -mtime +$DAYS -delete 2>/dev/null
find "$CROPS_DIR" -type f -mtime +$DAYS -delete 2>/dev/null
find "$RAW_DIR" -type f -mtime +$DAYS -delete 2>/dev/null

echo "Cleanup complete."
```

```bash
# Add to crontab (weekly at 3 AM Sunday)
crontab -e
# 0 3 * * 0 cd /path/to/medical-ocr-trainer && bash scripts/cleanup_old_data.sh
```

## 3. Backup Policy

### Critical Data (Must Backup)

| Data | Method | Frequency | Destination |
|------|--------|-----------|-------------|
| `data/corrections.db` | File copy | Weekly | External drive / cloud |
| `exports/` | Git commit / cloud sync | On each export | GitHub / S3 |
| `data/golden/` | Git commit | On each change | GitHub |

### Backup Script

```bash
#!/bin/bash
# scripts/backup.sh
BACKUP_DIR="./backups/$(date +%Y-%m-%d)"
mkdir -p "$BACKUP_DIR"

# Backup SQLite database
cp data/corrections.db "$BACKUP_DIR/corrections_$(date +%H%M%S).db" 2>/dev/null

# Backup exports
cp -r exports/ "$BACKUP_DIR/exports/" 2>/dev/null

# Keep only last 30 backups
ls -dt ./backups/*/ 2>/dev/null | tail -n +31 | xargs rm -rf

echo "Backup saved to: $BACKUP_DIR"
```

## 4. Data Privacy

- **No real PHI/PII** should be committed to git
- Upload images should be anonymized before use in training
- `data/corrections.db` contains OCR text — treat as potentially sensitive
- Export files (`exports/`) should be reviewed for PHI before sharing

## 5. Data Quality

### Correction Quality Gates

Before exporting training data, verify:
- [ ] At least 50 corrected samples
- [ ] Correction acceptance rate > 80%
- [ ] No duplicate images in the dataset
- [ ] Mixed specialties represented (not just one type)

### Dataset Versioning

When exporting a training dataset:
1. Use date-stamped filenames: `training_2026-06-14.jsonl`
2. Record the source correction count in exports metadata
3. Track CER/WER improvement over time using `medical-ocr-benchmarks`

## 6. Storage Estimates

| Data Type | Size per 1000 pages | 10K pages | 100K pages |
|-----------|---------------------|-----------|------------|
| Uploads (images) | ~500 MB | ~5 GB | ~50 GB |
| Crops (word images) | ~50 MB | ~500 MB | ~5 GB |
| Corrections DB | ~5 MB | ~50 MB | ~500 MB |
| Exports (JSONL) | ~10 MB | ~100 MB | ~1 GB |
| Exports (Parquet) | ~2 MB | ~20 MB | ~200 MB |

---

> Part of the [Medical OCR Ecosystem](https://github.com/DrAbdulmalek/omni-medical-suite/blob/main/PORTFOLIO.md)

## 6. Parquet Export Integration

Training data can now be exported in Parquet format for efficient downstream processing:

| Format | Use Case | Compression | Best For |
|--------|----------|-------------|----------|
| JSONL | HuggingFace training | None | Direct model training |
| CSV | Excel/analysis | None | Human review |
| Parquet | Analytics/warehousing | zstd | Polars, DuckDB, large-scale |
| HF Image Folders | TrOCR fine-tuning | None | Image-to-text models |

### Parquet Export Command
```bash
# Export training pairs as Parquet
python export_training.py --format parquet --output exports/
```

### Parquet Schema
- `training_pairs.parquet`: columns = [image_id, crop_path, predicted_text, corrected_text, confidence, script_class, specialty]
- `benchmark_results.parquet`: columns = [engine, language, specialty, cer, wer, medical_accuracy, timestamp]

## 7. Automated Governance (CI/CD)

### Weekly Governance Workflow
Runs every Sunday 03:00 UTC via GitHub Actions:

| Step | What It Does | Frequency |
|------|-------------|-----------|
| Cleanup Check | Verifies cleanup scripts exist and are syntactically valid | Weekly |
| Backup Verification | Validates backup script integrity | Weekly |
| DB Schema Validation | Creates test DB, verifies all 5 tables + 4 indexes | Weekly |

### Manual Governance Actions
```bash
# Trigger via GitHub Actions UI
# Actions → Data Governance → Run workflow → Select action
# Options: all, cleanup, backup-check, db-stats
```

## 8. Privacy & Compliance

### PHI Handling
- **No real PHI in git**: All medical images are gitignored (uploads/, crops/)
- **DB treated as sensitive**: `corrections.db` may contain document text fragments
- **Export review**: All exports should be reviewed for accidental PHI before sharing
- **Golden datasets**: `data/golden/` contains only synthetic/test data

### Data Deletion Request
To request deletion of all stored data:
1. Delete `data/corrections.db`
2. Clear `uploads/`, `crops/`, `exports/`, `data/raw/`
3. Re-initialize: `python -c "import sqlite3; sqlite3.connect('data/corrections.db').close()"`

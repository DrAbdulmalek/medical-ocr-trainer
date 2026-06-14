#!/bin/bash
# backup.sh — Backup critical medical-ocr-trainer data
# Usage: bash scripts/backup.sh

set -euo pipefail

BACKUP_ROOT="./backups"
TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)
BACKUP_DIR="$BACKUP_ROOT/$TIMESTAMP"

mkdir -p "$BACKUP_DIR"

echo "=== Medical OCR Trainer — Backup $TIMESTAMP ==="

# 1. Backup SQLite database
if [ -f "data/corrections.db" ]; then
    cp "data/corrections.db" "$BACKUP_DIR/corrections.db"
    SIZE=$(du -sh "$BACKUP_DIR/corrections.db" | cut -f1)
    echo "[OK] corrections.db backed up ($SIZE)"
else
    echo "[SKIP] corrections.db not found"
fi

# 2. Backup exports
if [ -d "exports" ] && [ "$(ls -A exports/ 2>/dev/null)" ]; then
    cp -r exports/ "$BACKUP_DIR/exports/"
    SIZE=$(du -sh "$BACKUP_DIR/exports" | cut -f1)
    echo "[OK] exports/ backed up ($SIZE)"
else
    echo "[SKIP] exports/ is empty"
fi

# 3. Backup golden datasets
if [ -d "data/golden" ]; then
    cp -r data/golden/ "$BACKUP_DIR/golden/"
    echo "[OK] data/golden/ backed up"
fi

# 4. Cleanup old backups (keep last 30)
BACKUP_COUNT=$(ls -d "$BACKUP_ROOT"/*/ 2>/dev/null | wc -l)
if [ "$BACKUP_COUNT" -gt 30 ]; then
    ls -dt "$BACKUP_ROOT"/*/ | tail -n +31 | xargs rm -rf
    echo "[CLEANUP] Removed old backups (kept 30 most recent)"
fi

TOTAL_SIZE=$(du -sh "$BACKUP_DIR" | cut -f1)
echo ""
echo "=== Backup complete: $BACKUP_DIR ($TOTAL_SIZE) ==="

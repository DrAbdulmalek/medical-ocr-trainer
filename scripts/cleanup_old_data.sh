#!/bin/bash
# cleanup_old_data.sh — Remove files older than retention period
# Usage: bash scripts/cleanup_old_data.sh [DAYS]
# Default: 90 days

set -euo pipefail

DAYS=${1:-90}
DELETED=0

echo "=== Cleaning files older than $DAYS days ==="

for DIR in "uploads" "crops" "data/raw"; do
    if [ -d "$DIR" ]; then
        COUNT_BEFORE=$(find "$DIR" -type f 2>/dev/null | wc -l)
        find "$DIR" -type f -mtime +"$DAYS" -delete 2>/dev/null || true
        COUNT_AFTER=$(find "$DIR" -type f 2>/dev/null | wc -l)
        REMOVED=$((COUNT_BEFORE - COUNT_AFTER))
        DELETED=$((DELETED + REMOVED))
        echo "[OK] $DIR: removed $REMOVED files"
    fi
done

echo ""
echo "=== Cleanup complete: $DELETED files removed ==="

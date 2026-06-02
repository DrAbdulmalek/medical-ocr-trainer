"""
أداة تصدير بيانات التدريب بصيغ متعددة
==========================================
يدعم:
  - JSONL: للاستخدام مع HuggingFace Datasets
  - CSV: للاستخدام مع Pandas / Excel
  - HuggingFace format: images/ + metadata.jsonl

الاستخدام:
    python export_training.py [--format jsonl|csv|huggingface] [--output dir]
    python export_training.py --stats          # عرض إحصائيات فقط
"""

import os
import json
import csv
import sqlite3
import argparse
from datetime import datetime


DB_PATH = "data/corrections.db"
DIR_CROPS = "crops"


def get_corrections(db_path=DB_PATH, status_filter=None):
    """جلب التصحيحات من قاعدة البيانات"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    if status_filter:
        c.execute("""
            SELECT w.id, w.predicted_text, w.corrected_text, w.confidence,
                   w.bbox, w.script_class, w.is_gold_standard,
                   w.correction_count, w.corrected_at, w.review_status,
                   i.filename, i.path as image_path
            FROM words w
            JOIN images i ON w.image_id = i.id
            WHERE w.is_corrected = 1
              AND w.corrected_text IS NOT NULL
              AND w.corrected_text != w.predicted_text
              AND (? IS NULL OR w.review_status = ?)
            ORDER BY w.confidence ASC
        """, (status_filter, status_filter))
    else:
        c.execute("""
            SELECT w.id, w.predicted_text, w.corrected_text, w.confidence,
                   w.bbox, w.script_class, w.is_gold_standard,
                   w.correction_count, w.corrected_at, w.review_status,
                   i.filename, i.path as image_path
            FROM words w
            JOIN images i ON w.image_id = i.id
            WHERE w.is_corrected = 1
              AND w.corrected_text IS NOT NULL
              AND w.corrected_text != w.predicted_text
            ORDER BY w.confidence ASC
        """)

    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def export_jsonl(corrections, output_path):
    """تصدير بصيغة JSONL"""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for row in corrections:
            record = {
                "word_id": row["id"],
                "predicted_text": row["predicted_text"],
                "corrected_text": row["corrected_text"],
                "confidence": row["confidence"],
                "bbox": json.loads(row["bbox"]),
                "script_class": row["script_class"],
                "is_gold_standard": bool(row["is_gold_standard"]),
                "correction_count": row["correction_count"],
                "document": row["filename"],
                "crop_path": os.path.join(DIR_CROPS, f"{row['id']}.png"),
                "review_status": row["review_status"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"✅ تم تصدير {len(corrections)} سجل إلى {output_path}")
    return output_path


def export_csv(corrections, output_path):
    """تصدير بصيغة CSV"""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'word_id', 'predicted_text', 'corrected_text', 'confidence',
            'script_class', 'is_gold_standard', 'correction_count',
            'document', 'review_status'
        ])

        for row in corrections:
            writer.writerow([
                row["id"],
                row["predicted_text"],
                row["corrected_text"],
                row["confidence"],
                row["script_class"],
                row["is_gold_standard"],
                row["correction_count"],
                row["filename"],
                row["review_status"],
            ])

    print(f"✅ تم تصدير {len(corrections)} سجل إلى {output_path}")
    return output_path


def export_huggingface(corrections, output_dir):
    """
    تصدير بصيغة HuggingFace Datasets:
      output_dir/
        train/
          images/       ← نسخ القصاصات
          metadata.jsonl
    """
    images_dir = os.path.join(output_dir, 'train', 'images')
    os.makedirs(images_dir, exist_ok=True)

    metadata_path = os.path.join(output_dir, 'train', 'metadata.jsonl')
    copied = 0

    with open(metadata_path, 'w', encoding='utf-8') as f:
        for row in corrections:
            crop_src = os.path.join(DIR_CROPS, f"{row['id']}.png")
            crop_dst = os.path.join(images_dir, f"{row['id']}.png")

            file_name = f"{row['id']}.png"
            if os.path.exists(crop_src):
                import shutil
                shutil.copy2(crop_src, crop_dst)
                copied += 1

            record = {
                "file_name": file_name,
                "text": row["corrected_text"],
                "predicted_text": row["predicted_text"],
                "confidence": row["confidence"],
                "script_class": row["script_class"],
                "is_gold_standard": bool(row["is_gold_standard"]),
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    # إنشاء dataset_dict.py
    dict_path = os.path.join(output_dir, 'dataset_dict.py')
    with open(dict_path, 'w', encoding='utf-8') as f:
        f.write('''"""كود تحميل مجموعة البيانات في HuggingFace"""
from datasets import load_dataset

dataset = load_dataset("imagefolder", data_dir=".")
print(dataset)
print(dataset["train"][0])
''')

    print(f"✅ تم تصدير {len(corrections)} سجل إلى {output_dir}")
    print(f"   صور منسوخة: {copied}/{len(corrections)}")
    return output_dir


def show_stats(db_path=DB_PATH):
    """عرض إحصائيات قاعدة البيانات"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    total_images = c.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    total_words = c.execute("SELECT COUNT(*) FROM words").fetchone()[0]
    corrected = c.execute("SELECT COUNT(*) FROM words WHERE is_corrected=1").fetchone()[0]
    gold = c.execute("SELECT COUNT(*) FROM words WHERE is_gold_standard=1").fetchone()[0]
    changed = c.execute(
        "SELECT COUNT(*) FROM words WHERE is_corrected=1 AND corrected_text != predicted_text"
    ).fetchone()[0]

    # توزيع حالات المراجعة
    c.execute("SELECT review_status, COUNT(*) FROM words GROUP BY review_status")
    status_dist = dict(c.fetchall())

    # توزيع اللغات
    c.execute("SELECT script_class, COUNT(*) FROM words GROUP BY script_class")
    script_dist = dict(c.fetchall())

    # توزيع الثقة
    c.execute(
        "SELECT COUNT(*) FROM words WHERE confidence < 0.5"
    )
    low_conf = c.fetchone()[0]
    c.execute(
        "SELECT COUNT(*) FROM words WHERE confidence BETWEEN 0.5 AND 0.7"
    )
    mid_conf = c.fetchone()[0]
    c.execute(
        "SELECT COUNT(*) FROM words WHERE confidence > 0.7"
    )
    high_conf = c.fetchone()[0]

    conn.close()

    print("=" * 50)
    print("📊 إحصائيات قاعدة بيانات التدريب")
    print("=" * 50)
    print(f"  المستندات:       {total_images}")
    print(f"  الكلمات الكلية:  {total_words}")
    print(f"  التصحيحات:       {corrected}")
    print(f"  تغييرات فعلية:  {changed}")
    print(f"  عينات ذهبية:    {gold}")
    print()
    print("  توزيع الثقة:")
    print(f"    منخفضة (<50%):  {low_conf}")
    print(f"    متوسطة (50-70%): {mid_conf}")
    print(f"    عالية (>70%):   {high_conf}")
    print()
    print("  حالات المراجعة:")
    for status, count in sorted(status_dist.items()):
        print(f"    {status}: {count}")
    print()
    print("  توزيع اللغات:")
    for script, count in sorted(script_dist.items()):
        print(f"    {script}: {count}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="تصدير بيانات التدريب")
    parser.add_argument('--format', choices=['jsonl', 'csv', 'huggingface'], default='jsonl',
                        help='صيغة التصدير')
    parser.add_argument('--output', default='exports', help='مجلد الإخراج')
    parser.add_argument('--db', default=DB_PATH, help='مسار قاعدة البيانات')
    parser.add_argument('--gold-only', action='store_true', help='تصدير العينات الذهبية فقط')
    parser.add_argument('--stats', action='store_true', help='عرض الإحصائيات فقط')
    args = parser.parse_args()

    if args.stats:
        show_stats(args.db)
    else:
        status_filter = 'gold' if args.gold_only else None
        corrections = get_corrections(args.db, status_filter)

        if not corrections:
            print("⚠️ لا توجد تصحيحات لتصديرها.")
            show_stats(args.db)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = "_gold" if args.gold_only else ""

            if args.format == 'jsonl':
                output_path = os.path.join(args.output, f"training_data{suffix}_{timestamp}.jsonl")
                export_jsonl(corrections, output_path)
            elif args.format == 'csv':
                output_path = os.path.join(args.output, f"training_data{suffix}_{timestamp}.csv")
                export_csv(corrections, output_path)
            elif args.format == 'huggingface':
                output_dir = os.path.join(args.output, f"hf_dataset{suffix}_{timestamp}")
                export_huggingface(corrections, output_dir)

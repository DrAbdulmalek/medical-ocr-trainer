"""
طبقة التصفية الآلية لتصحيحات OCR
=====================================
تُصنّف التصحيحات تلقائياً إلى:
  - gold: عينة ذهبية (جاهزة للتدريب مباشرة)
  - pending_review: تحتاج مراجعة بشرية إضافية
  - rejected: مرفوضة (ربما خطأ بشري)

المعايير المطبقة:
  1. عتبة الثقة: تصحيحات الثقة المنخفضة = عينة ذهبية عالية القيمة
  2. تكرار الخطأ: كلمة صححها عدة مستخدمين بنفس الطريقة = قاعدة ذهبية
  3. الأهمية السريرية: الجرعات والأدوية تُعطى وزناً أعلى
  4. فحص شكلي: استبعاد القيم الرقمية الخالصة والرموز غير النصية

الاستخدام:
    python data_filter.py [--threshold 0.65] [--min-agree 2] [--db data/corrections.db]

كوحدة مستوردة:
    from data_filter import DataFilter
    f = DataFilter("data/corrections.db")
    results = f.classify_all()
    f.apply_filters()
"""

import os
import re
import json
import sqlite3
import argparse


# ============================================================
# قوامات طبية ثابتة (مرحلة أولى — يمكن توسيعها لاحقاً)
# ============================================================

# تعابير نمطية للأرقام والجرعات
DOSAGE_PATTERNS = [
    r'^\d+\s*mg$',               # 500 mg
    r'^\d+\s*mcg$',              # 50 mcg
    r'^\d+\s*ml$',               # 10 ml
    r'^\d+\s* IU$',              # 1000 IU
    r'^\d+/\d+\s*(mg|ml|g)$',   # 5/250 mg
    r'^\d+\s*[×x]\s*\d+$',       # 2x3
    r'^\d+\s*-%$',               # 80-%
    r'^\d+\s*/\s*\d+$',          # 10/5
    r'^\d+\s*م[لجر]$',           # عربي
]

# تعابير للرموز غير النصية
NON_TEXT_PATTERNS = [
    r'^[\d\s\.\-+/]+$',           # أرقام ورموز فقط
    r'^[=\-+]{2,}$',               # خطوط فقط
    r'^[\•\-\*]+$',                # نقاط فقط
    r'^\s*$',                      # فارغ
    r'^\d{1,2}[\-/]\d{1,2}[\-/]\d{2,4}$',  # تاريخ
]

# كلمات سريرية عالية الأهمية (وزن أعلى عند التصفية)
CLINICAL_KEYWORDS = {
    # أدوية شائعة
    'paracetamol', 'ibuprofen', 'amoxicillin', 'metformin', 'insulin',
    'aspirin', 'prednisone', 'prednisolone', 'dexamethasone',
    'ciprofloxacin', 'azithromycin', 'omeprazole', 'enalapril',
    'digoxin', 'warfarin', 'heparin', 'morphine', 'fentanyl',
    # أدوية عربية
    'باراسيتامول', 'إيبوبروفين', 'أموكسيسيلين', 'ميتفورمين',
    'أنسولين', 'أسبرين',

    # تشخيصات شائعة
    'diabetes', 'hypertension', 'fracture', 'pneumonia', 'asthma',
    'osteoporosis', 'arthritis', 'appendicitis',
    'السكري', 'ارتفاع ضغط الدم', 'كسر', 'التهاب رئوي', 'ربو',

    # مصطلحات إجراءات
    'ORIF', 'biopsy', 'MRI', 'CT', 'ECG', 'IV', 'IM', 'SC',
    'AVN', 'LMWH', 'NG', 'NPO', 'PRN', 'BID', 'TID', 'QID',
    'AP', 'Lateral', 'PA', 'Anterior', 'Posterior',

    # اختصارات جرعات
    'PO', 'IV', 'IM', 'SC', 'SL', 'PR', 'Topical',
}

# تعابير التعرف على الكلمات الطبية (نمط مطور)
MEDICAL_PATTERNS = [
    r'^(?:[A-Z]{2,5}\d*|\d+\s*[a-zA-Z]+)$',   # اختصارات مثل AVN, ORIF
    r'[aeiou]{2,}.*[aeiou]{2,}',                # كلمات لاتينية طويلة
    r'itis$', r'osis$', r'oma$', r'itis$', r'algia$',  # لواحق طبية
]


class DataFilter:
    """
    مصنّف التصحيحات الآلي.
    
    يأخذ مسار قاعدة البيانات ويُصنّف كل التصحيحات
    حسب مجموعة معايير متعددة.
    """

    def __init__(self, db_path="data/corrections.db"):
        self.db_path = db_path
        self.confidence_threshold = 0.65
        self.min_agreement_count = 2
        self.results = []

    def classify_correction(self, row):
        """
        تصنيف تصحيح واحد حسب المعايير الثلاثة.
        
        المعايير:
        --------
        1. فحص شكلي (Formal Check):
           - هل النص يحتوي على رموز غير حرفية فقط؟ → rejected (dosage/number)
           - هل هو فارغ أو رموز؟ → rejected
           
        2. قاموس طبي (Medical Dictionary):
           - مطابقة مع قائمة المصطلحات السريرية → gold
           - تطابق جزئي → pending_review
           - غير معروف → يعتمد على معيار الثقة
           
        3. منطق الثقة (Confidence Logic):
           - ثقة منخفضة + تصحيح → gold (عينة عالية القيمة)
           - ثقة عالية + تصحيح → pending_review (نادر، ربما خطأ بشري)
           - تكرار الخطأ → gold (إجماع المستخدمين)
           
        4. أهمية سريرية (Clinical Priority):
           - جرعات وأدوية → تُقبل حتى لو ثقتها عالية
        
        العائد:
        -------
        dict مع:
            - readiness: 'gold' | 'pending_review' | 'rejected' | 'dosage_flag'
            - reasons: list[str] أسباب التصنيف
            - priority: int 1-5 (1 = أعلى أولوية)
        """
        predicted = row.get('predicted_text', '')
        corrected = row.get('corrected_text', '')
        confidence = row.get('confidence', 1.0)
        correction_count = row.get('correction_count', 1)
        script_class = row.get('script_class', 'auto')

        reasons = []
        readiness = 'pending_review'
        priority = 3  # افتراضي

        # === طبقة 1: فحص شكلي ===
        if not corrected or not corrected.strip():
            return {
                'readiness': 'rejected',
                'reasons': ['نص فارغ'],
                'priority': 5
            }

        # هل هي جرعة/رقم؟
        is_dosage = any(re.match(p, corrected.strip(), re.IGNORECASE) for p in DOSAGE_PATTERNS)
        if is_dosage:
            return {
                'readiness': 'dosage_flag',
                'reasons': ['جرعة أو رقم — لا تدخل التدريب النصي'],
                'priority': 4
            }

        # هل هي رموز غير نصية؟
        is_non_text = any(re.match(p, corrected.strip(), re.IGNORECASE) for p in NON_TEXT_PATTERNS)
        if is_non_text:
            return {
                'readiness': 'rejected',
                'reasons': ['رموز غير نصية — مستبعدة من التدريب'],
                'priority': 5
            }

        # === طبقة 2: قاموس طبي ===
        corrected_lower = corrected.strip().lower()
        dict_match = None  # 'full' | 'partial' | 'partial_arabic' | None

        for keyword in CLINICAL_KEYWORDS:
            if corrected_lower == keyword.lower():
                dict_match = 'full'
                break
            elif keyword.lower() in corrected_lower or corrected_lower in keyword.lower():
                dict_match = 'partial'
                break

        # تحقق من اللواحق الطبية
        if not dict_match:
            for pattern in MEDICAL_PATTERNS:
                if re.search(pattern, corrected, re.IGNORECASE):
                    dict_match = 'partial'
                    break

        if dict_match == 'full':
            reasons.append('مطابقة كاملة في القاموس الطبي')
            priority = 1  # أعلى أولوية
            readiness = 'gold'
        elif dict_match == 'partial':
            reasons.append('تطابق جزئي في القاموس الطبي')
            priority = 2

        # === طبقة 3: منطق الثقة ===
        if confidence < self.confidence_threshold:
            if corrected != predicted:
                reasons.append(f'ثقة منخفضة ({confidence:.2f}) + تصحيح = عينة ذهبية عالية القيمة')
                readiness = 'gold'
                priority = min(priority, 1)
            else:
                reasons.append(f'ثقة منخفضة ({confidence:.2f}) لكن بدون تصحيح')
        else:
            # ثقة عالية + تصحيح = غريب، ربما خطأ بشري
            if corrected != predicted and confidence > 0.85:
                reasons.append(f'ثقة عالية ({confidence:.2f}) مع تصحيح — يحتاج مراجعة')
                readiness = 'pending_review'
                priority = max(priority, 3)

        # === طبقة 4: تكرار الخطأ (إجماع) ===
        if correction_count >= self.min_agreement_count:
            reasons.append(f'تصحيح متكرر ({correction_count} مرة) = إجماع مستخدمين')
            readiness = 'gold'
            priority = 1
        elif correction_count > 1:
            reasons.append(f'تصحيح مكرر ({correction_count} مرات)')
            priority = min(priority, 2)

        # === طبقة 5: أهمية سريرية ===
        is_clinical = (
            dict_match == 'full'
            or any(kw.lower() in corrected_lower for kw in CLINICAL_KEYWORDS)
        )
        if is_clinical and readiness == 'pending_review':
            reasons.append('مصطلح سريري — يُقبل للتدريب رغم عدم اليقين')
            readiness = 'gold'
            priority = 2

        return {
            'readiness': readiness,
            'reasons': reasons,
            'priority': priority
        }

    def classify_all(self):
        """تصنيف جميع التصحيحات في قاعدة البيانات"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        c.execute("""
            SELECT id, predicted_text, corrected_text, confidence,
                   correction_count, script_class
            FROM words
            WHERE is_corrected = 1 AND corrected_text IS NOT NULL
        """)

        rows = c.fetchall()
        self.results = []

        for row in rows:
            row_dict = dict(row)
            classification = self.classify_correction(row_dict)
            row_dict['classification'] = classification
            self.results.append(row_dict)

        conn.close()
        return self.results

    def apply_filters(self):
        """
        تطبيق التصنيف على قاعدة البيانات.
        يُحدّث حقل review_status و is_gold_standard.
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        updated = 0
        for item in self.results:
            cl = item['classification']
            word_id = item['id']
            new_status = cl['readiness']
            is_gold = 1 if new_status == 'gold' else 0

            c.execute(
                "UPDATE words SET review_status=?, is_gold_standard=? WHERE id=?",
                (new_status, is_gold, word_id)
            )
            updated += 1

        conn.commit()
        conn.close()
        return updated

    def get_summary(self):
        """ملخص نتائج التصنيف"""
        if not self.results:
            return {}

        summary = {}
        for item in self.results:
            r = item['classification']['readiness']
            summary[r] = summary.get(r, 0) + 1

        return summary

    def get_gold_samples(self):
        """جلب العينات الذهبية فقط (جاهزة للتدريب)"""
        if not self.results:
            self.classify_all()

        return [item for item in self.results if item['classification']['readiness'] == 'gold']

    def export_gold_jsonl(self, output_path="exports/gold_standard.jsonl"):
        """تصدير العينات الذهبية بصيغة JSONL"""
        gold = self.get_gold_samples()

        if not gold:
            print("لا توجد عينات ذهبية لتصديرها.")
            return

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for item in gold:
                record = {
                    'word_id': item['id'],
                    'predicted_text': item['predicted_text'],
                    'corrected_text': item['corrected_text'],
                    'confidence': item['confidence'],
                    'script_class': item['script_class'],
                    'correction_count': item['correction_count'],
                    'crop_path': os.path.join('crops', f"{item['id']}.png"),
                    'classification': item['classification'],
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

        print(f"تم تصدير {len(gold)} عينة ذهبية إلى {output_path}")
        return output_path


# ============================================================
# تشغيل مباشر من سطر الأوامر
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="طبقة التصفية الآلية لتصحيحات OCR")
    parser.add_argument('--db', default='data/corrections.db', help='مسار قاعدة البيانات')
    parser.add_argument('--threshold', type=float, default=0.65, help='عتبة الثقة الافتراضية')
    parser.add_argument('--min-agree', type=int, default=2, help='الحد الأدنى لتكرار التصحيح')
    parser.add_argument('--export', action='store_true', help='تصدير العينات الذهبية')
    parser.add_argument('--apply', action='store_true', help='تطبيق الفلاتر على قاعدة البيانات')
    args = parser.parse_args()

    f = DataFilter(args.db)
    f.confidence_threshold = args.threshold
    f.min_agreement_count = args.min_agree

    print(f"🔍 تصنيف التصحيحات (عتبة الثقة: {args.threshold}, حد التكرار: {args.min_agree})...")
    results = f.classify_all()
    print(f"   إجمالي التصحيحات: {len(results)}")

    summary = f.get_summary()
    print("\n📊 ملخص التصنيف:")
    for status, count in sorted(summary.items()):
        label = {
            'gold': '✅ عينات ذهبية',
            'pending_review': '⏳ قيد المراجعة',
            'rejected': '❌ مرفوضة',
            'dosage_flag': '💊 جرعات/أرقام',
        }.get(status, f'❓ {status}')
        print(f"   {label}: {count}")

    # عرض أمثلة
    print("\n📝 أمثلة على التصنيف:")
    for item in results[:5]:
        cl = item['classification']
        icon = {'gold': '✅', 'pending_review': '⏳', 'rejected': '❌', 'dosage_flag': '💊'}.get(cl['readiness'], '❓')
        print(f"   {icon} [{item['predicted_text']!r} → {item['corrected_text']!r}] (conf={item['confidence']:.2f})")
        for reason in cl['reasons']:
            print(f"      ← {reason}")

    if args.apply:
        updated = f.apply_filters()
        print(f"\n✅ تم تحديث {updated} سجل في قاعدة البيانات")

    if args.export:
        f.export_gold_jsonl()

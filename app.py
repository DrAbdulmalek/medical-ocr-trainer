"""
Medical OCR Trainer — مُدرّب التعرف على الملاحظات الطبية اليدوية
================================================================
واجهة Streamlit تفاعلية لرفع الملاحظات الطبية الممسوحة ضوئياً،
تشغيل 5 محركات OCR مع تجمع ذكي، تصحيح الكلمات،
وحفظ بيانات التدريب تلقائياً.

الاستخدام:
    pip install -r requirements.txt
    streamlit run app.py

المحركات:
    - PaddleOCR: محرك التعرف على النصوص (عربي + إنجليزي)
    - EasyOCR: دعم +80 لغة
    - Tesseract: سريع للمطبوع
    - TrOCR: Transformer للخط اليدوي
    - Surya OCR: محرك حديث عالي الدقة

استراتيجيات الدمج:
    - majority_voting: تصويت الأغلبية
    - confidence_weighted: متوسط مرجح بالثقة
    - levenshtein_consensus: إجماع المسافة
    - best_single: أفضل نتيجة واحدة
"""

import os
import sys
import json
import sqlite3
import uuid
import time
import streamlit as st
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from datetime import datetime

# استيراد نظام التجمع
from ensemble_ocr import EnsembleOCR, EnsembleResult


# ============================================================
# إعدادات المسارات — دعم Hugging Face Spaces
# ============================================================
# على HF Spaces، /data/ هو المجلد الوحيد الدائم (يبقى بعد إعادة تشغيل الحاوية)
IS_HF_SPACE = os.environ.get("SPACE_ID") is not None

if IS_HF_SPACE:
    BASE_DIR = "/data"
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DIR_UPLOADS = os.path.join(BASE_DIR, "uploads")
DIR_CROPS = os.path.join(BASE_DIR, "crops")
DIR_DB = os.path.join(BASE_DIR, "db")
DIR_EXPORTS = os.path.join(BASE_DIR, "exports")
DB_PATH = os.path.join(DIR_DB, "corrections.db")

for d in [DIR_UPLOADS, DIR_CROPS, DIR_DB, DIR_EXPORTS]:
    os.makedirs(d, exist_ok=True)


# ============================================================
# تهيئة نظام التجمع (cached)
# ============================================================
def get_ensemble(
    engines=None,
    strategy='majority_voting',
    confidence_threshold=0.3,
):
    """إنشاء أو إعادة استخدام نظام التجمع"""
    key = f"ensemble_{strategy}_{'_'.join(sorted(engines or []))}_{confidence_threshold}"
    if key not in st.session_state:
        st.session_state[key] = EnsembleOCR(
            engines=engines or ['paddleocr', 'easyocr', 'tesseract', 'trocr', 'surya'],
            strategy=strategy,
            confidence_threshold=confidence_threshold,
        )
    return st.session_state[key]


# ============================================================
# قاعدة البيانات
# ============================================================
def init_db():
    """تهيئة قاعدة البيانات وإنشاء الجداول إذا لم تكن موجودة"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # جدول الصور الأصلية
    c.execute("""CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        path TEXT NOT NULL,
        width INTEGER,
        height INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    # جدول الكلمات المستخرجة والتصحيحات (محدث لدعم التجمع)
    c.execute("""CREATE TABLE IF NOT EXISTS words (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_id INTEGER NOT NULL,
        bbox TEXT NOT NULL,
        predicted_text TEXT NOT NULL,
        confidence REAL NOT NULL,
        corrected_text TEXT,
        crop_path TEXT,
        is_corrected BOOLEAN DEFAULT 0,
        corrected_at TIMESTAMP,
        review_status TEXT DEFAULT 'pending',
        is_gold_standard BOOLEAN DEFAULT 0,
        script_class TEXT DEFAULT 'auto',
        correction_count INTEGER DEFAULT 0,
        ensemble_strategy TEXT,
        engines_used TEXT,
        agreement_count INTEGER DEFAULT 1,
        engine_votes TEXT,
        FOREIGN KEY(image_id) REFERENCES images(id)
    )""")

    # جدول نتائج المحركات الفردية
    c.execute("""CREATE TABLE IF NOT EXISTS engine_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_id INTEGER NOT NULL,
        engine_name TEXT NOT NULL,
        word_count INTEGER DEFAULT 0,
        processing_time REAL DEFAULT 0,
        available BOOLEAN DEFAULT 1,
        error TEXT,
        raw_results TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(image_id) REFERENCES images(id)
    )""")

    # جدول إصدارات النماذج
    c.execute("""CREATE TABLE IF NOT EXISTS model_versions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        version TEXT UNIQUE NOT NULL,
        trained_on_count INTEGER DEFAULT 0,
        cer_score REAL,
        wer_score REAL,
        medical_term_accuracy REAL,
        deployed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    # جدول سجل التصحيحات (لتتبع التكرار)
    c.execute("""CREATE TABLE IF NOT EXISTS correction_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        word_id INTEGER NOT NULL,
        old_text TEXT NOT NULL,
        new_text TEXT NOT NULL,
        corrected_by TEXT DEFAULT 'user',
        confidence_at_correction REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(word_id) REFERENCES words(id)
    )""")

    conn.commit()
    conn.close()


def save_image_meta(filename, path, width=None, height=None):
    """حفظ بيانات الصورة في قاعدة البيانات"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO images (filename, path, width, height) VALUES (?, ?, ?, ?)",
        (filename, path, width, height)
    )
    conn.commit()
    img_id = c.lastrowid
    conn.close()
    return img_id


def save_words_meta(image_id, ensemble_result):
    """حفظ نتائج التجمع في قاعدة البيانات"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    ids = []

    for word in ensemble_result.words:
        script = detect_script(word.text)
        c.execute(
            """INSERT INTO words
            (image_id, bbox, predicted_text, confidence, corrected_text, script_class,
             ensemble_strategy, engines_used, agreement_count, engine_votes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                image_id, json.dumps(word.bbox), word.text, word.confidence,
                word.text, script, word.strategy,
                json.dumps(word.engines_used), word.agreement_count,
                json.dumps(word.engine_votes),
            )
        )
        ids.append(c.lastrowid)

    # حفظ سجلات المحركات
    for name, er in ensemble_result.engine_results.items():
        c.execute(
            """INSERT INTO engine_logs
            (image_id, engine_name, word_count, processing_time, available, error, raw_results)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                image_id, name, len(er.words), er.processing_time,
                1 if er.available else 0, er.error,
                json.dumps(er.to_dict(), ensure_ascii=False),
            )
        )

    conn.commit()
    conn.close()
    return ids


def update_word_correction(word_id, corrected_text, crop_path, is_gold=False):
    """تحديث تصحيح كلمة في قاعدة البيانات"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # سجل التصحيح القديم
    c.execute("SELECT predicted_text, confidence FROM words WHERE id=?", (word_id,))
    row = c.fetchone()
    if row:
        old_text, conf = row[0], row[1]
        c.execute(
            "INSERT INTO correction_history (word_id, old_text, new_text, confidence_at_correction) VALUES (?, ?, ?, ?)",
            (word_id, old_text, corrected_text, conf)
        )

    # تحديث الكلمة
    c.execute(
        "UPDATE words SET corrected_text=?, crop_path=?, is_corrected=1, corrected_at=CURRENT_TIMESTAMP, review_status='approved', is_gold_standard=?, correction_count=correction_count+1 WHERE id=?",
        (corrected_text, crop_path, 1 if is_gold else 0, word_id)
    )
    conn.commit()
    conn.close()


def get_words(image_id):
    """جلب كلمات صورة معينة مرتبة حسب الثقة (الأقل أولاً)"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute(
        """SELECT id, bbox, predicted_text, confidence, corrected_text,
                  is_corrected, script_class, correction_count,
                  ensemble_strategy, engines_used, agreement_count, engine_votes
           FROM words WHERE image_id=? ORDER BY confidence ASC""",
        (image_id,)
    )
    res = [dict(row) for row in c.fetchall()]
    conn.close()
    return res


def get_all_documents():
    """جلب قائمة كل المستندات مع عدد الكلمات"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""
        SELECT i.id, i.filename, i.created_at,
               COUNT(w.id) as word_count,
               SUM(CASE WHEN w.is_corrected=1 THEN 1 ELSE 0 END) as corrected_count
        FROM images i
        LEFT JOIN words w ON w.image_id = i.id
        GROUP BY i.id
        ORDER BY i.created_at DESC
    """)
    res = [dict(row) for row in c.fetchall()]
    conn.close()
    return res


def get_engine_logs(image_id):
    """جلب سجل أداء المحركات لمستند معين"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute(
        "SELECT * FROM engine_logs WHERE image_id=? ORDER BY id",
        (image_id,)
    )
    res = [dict(row) for row in c.fetchall()]
    conn.close()
    return res


def get_stats():
    """جلب إحصائيات عامة"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    total_images = c.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    total_words = c.execute("SELECT COUNT(*) FROM words").fetchone()[0]
    corrected_words = c.execute("SELECT COUNT(*) FROM words WHERE is_corrected=1").fetchone()[0]
    gold_standard = c.execute("SELECT COUNT(*) FROM words WHERE is_gold_standard=1").fetchone()[0]
    pending = c.execute("SELECT COUNT(*) FROM words WHERE review_status='pending' AND is_corrected=0").fetchone()[0]

    avg_conf = c.execute("SELECT AVG(confidence) FROM words").fetchone()[0] or 0
    low_conf = c.execute("SELECT COUNT(*) FROM words WHERE confidence < 0.5").fetchone()[0]

    # إحصائيات التجمع
    ensemble_count = c.execute("SELECT COUNT(*) FROM words WHERE ensemble_strategy IS NOT NULL AND ensemble_strategy != ''").fetchone()[0]
    avg_agreement = c.execute("SELECT AVG(agreement_count) FROM words WHERE agreement_count > 0").fetchone()[0] or 0

    # حساب CER و WER تقريبي
    corrections = c.execute(
        "SELECT predicted_text, corrected_text FROM words WHERE is_corrected=1 AND corrected_text IS NOT NULL"
    ).fetchall()

    total_cer = 0
    total_wer = 0
    count = 0
    for pred, corr in corrections:
        if pred and corr and pred != corr:
            total_cer += _cer(pred, corr)
            total_wer += _wer(pred, corr)
            count += 1

    conn.close()
    return {
        "total_images": total_images,
        "total_words": total_words,
        "corrected_words": corrected_words,
        "gold_standard": gold_standard,
        "pending_review": pending,
        "avg_confidence": avg_conf,
        "low_confidence": low_conf,
        "cer": total_cer / count if count > 0 else 0,
        "wer": total_wer / count if count > 0 else 0,
        "ensemble_count": ensemble_count,
        "avg_agreement": avg_agreement,
    }


def _cer(predicted, actual):
    """معدل خطأ الحروف (Character Error Rate)"""
    p = predicted.replace(" ", "")
    a = actual.replace(" ", "")
    if not a:
        return 1.0
    m, n = len(p), len(a)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + (0 if p[i - 1] == a[j - 1] else 1),
            )
    return dp[m][n] / n


def _wer(predicted, actual):
    """معدل خطأ الكلمات (Word Error Rate)"""
    pw = predicted.split()
    aw = actual.split()
    if not aw:
        return 1.0 if pw else 0.0
    m, n = len(pw), len(aw)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + (0 if pw[i - 1] == aw[j - 1] else 1),
            )
    return dp[m][n] / n


def detect_script(text):
    """كشف نوع الخط (عربي / لاتيني / مختلط / رقمي)"""
    if not text:
        return "auto"
    arabic = sum(1 for c in text if '\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F' or '\uFB50' <= c <= '\uFDFF' or '\uFE70' <= c <= '\uFEFF')
    latin = sum(1 for c in text if 'A' <= c <= 'Z' or 'a' <= c <= 'z')
    digits = sum(1 for c in text if '0' <= c <= '9')
    total = max(len(text), 1)

    if arabic / total > 0.7:
        return "arabic"
    elif latin / total > 0.7:
        return "latin"
    elif digits / total > 0.7:
        return "numeric"
    elif arabic > 0 and latin > 0:
        return "mixed"
    return "auto"


# ============================================================
# معالجة الصورة (تحسين التباين والحدة)
# ============================================================
def preprocess_image(img_path):
    """
    معالجة مبدئية للصورة:
    - تحويل إلى تدرج رمادي
    - زيادة التباين (1.6x) للخط اليدوي
    - شحذ الحواف
    """
    img = Image.open(img_path).convert("L")
    img = ImageEnhance.Contrast(img).enhance(1.6)
    img = img.filter(ImageFilter.SHARPEN)
    pre_path = img_path + "_pre.png"
    img.save(pre_path)
    return pre_path


# ============================================================
# قص الصورة وحفظ القصاصة
# ============================================================
def crop_and_save(word_bbox, img_path, word_id, padding=8):
    """
    قص منطقة الكلمة من الصورة الأصلية مع هوامش.
    bbox بصيغة [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    """
    img = Image.open(img_path)
    w, h = img.size
    xs = [p[0] for p in word_bbox]
    ys = [p[1] for p in word_bbox]
    min_x, max_x = max(0, min(xs) - padding), min(w, max(xs) + padding)
    min_y, max_y = max(0, min(ys) - padding), min(h, max(ys) + padding)

    crop = img.crop((min_x, min_y, max_x, max_y))
    crop_path = os.path.join(DIR_CROPS, f"{word_id}.png")
    crop.save(crop_path)
    return crop_path


# ============================================================
# واجهة Streamlit الرئيسية
# ============================================================
def main():
    st.set_page_config(
        page_title="Medical OCR Trainer — Ensemble",
        page_icon="🏥",
        layout="wide"
    )

    init_db()

    # --- Sidebar ---
    with st.sidebar:
        st.title("Medical OCR Trainer")
        st.caption("Ensemble — 5 Engines")

        st.markdown("---")

        # === إعدادات المحركات ===
        st.subheader("⚙️ محركات OCR")

        all_engines = list(EnsembleOCR.ENGINE_MAP.keys())

        # فحص حالة المحركات
        if 'engine_availability' not in st.session_state:
            st.session_state.engine_availability = {}
            test_ocr = EnsembleOCR(engines=all_engines)
            st.session_state.engine_availability = test_ocr.get_available_engines()

        selected_engines = []
        for eng in all_engines:
            info = EnsembleOCR.ENGINE_DESCRIPTIONS.get(eng, {})
            avail = st.session_state.engine_availability.get(eng, False)
            label = f"{info.get('icon', '?')} {info.get('name', eng)}"
            if avail:
                if st.checkbox(label, value=True, key=f"engine_{eng}"):
                    selected_engines.append(eng)
            else:
                st.checkbox(label, value=False, key=f"engine_{eng}", disabled=True)
                st.caption(f"  ⚠️ غير متاح — {info.get('strengths', '')}")

        # === استراتيجية الدمج ===
        st.markdown("---")
        st.subheader("🔄 استراتيجية الدمج")

        strategy_map = {
            'majority_voting': '🗳️ تصويت الأغلبية',
            'confidence_weighted': '⚖️ متوسط مرجح بالثقة',
            'levenshtein_consensus': '📏 إجماع المسافة',
            'best_single': '🏆 أفضل نتيجة واحدة',
        }

        strategy = st.radio(
            "اختر الاستراتيجية:",
            list(strategy_map.keys()),
            format_func=lambda x: strategy_map[x],
            index=0,
            key="strategy_select",
        )

        # === حد الثقة ===
        confidence_threshold = st.slider(
            "حد الثقة الأدنى",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="تجاهل الكلمات أقل من هذا الحد"
        )

        st.markdown("---")

        # === الإحصائيات السريعة ===
        stats = get_stats()
        st.subheader("📊 الإحصائيات")
        st.metric("المستندات", stats["total_images"])
        st.metric("الكلمات", stats["total_words"])
        st.metric("التصحيحات", stats["corrected_words"])
        st.metric("عينات ذهبية", stats["gold_standard"])

        if stats["ensemble_count"] > 0:
            st.metric("نتائج التجمع", stats["ensemble_count"])
            st.metric("متوسط الاتفاق", f"{stats['avg_agreement']:.1f}")

        st.markdown("---")

        if stats["corrected_words"] > 0:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("CER", f"{stats['cer'] * 100:.1f}%")
            with col2:
                st.metric("WER", f"{stats['wer'] * 100:.1f}%")

            st.progress(
                min(1.0, stats["corrected_words"] / max(1, stats["total_words"])),
                text=f"نسبة التصحيح: {stats['corrected_words']}/{stats['total_words']}"
            )

        st.markdown("---")

        if st.button("📥 تصدير بيانات التدريب (JSONL)", use_container_width=True):
            export_training_data()

        if st.button("🔄 إعادة تهيئة قاعدة البيانات", use_container_width=True, type="secondary"):
            if st.session_state.get("confirm_reset"):
                os.remove(DB_PATH) if os.path.exists(DB_PATH) else None
                init_db()
                st.session_state["confirm_reset"] = False
                st.success("تمت إعادة التهيئة")
                st.rerun()
            else:
                st.session_state["confirm_reset"] = True
                st.warning("اضغط مرة أخرى للتأكيد")

    # --- Main Area ---
    st.title("Medical OCR Trainer — Ensemble")
    st.caption("5 Engines + Smart Merging | PaddleOCR + EasyOCR + Tesseract + TrOCR + Surya")

    # عرض حالة المحركات
    avail = st.session_state.get('engine_availability', {})
    engine_cols = st.columns(len(all_engines))
    for i, eng in enumerate(all_engines):
        info = EnsembleOCR.ENGINE_DESCRIPTIONS.get(eng, {})
        is_avail = avail.get(eng, False)
        with engine_cols[i]:
            status_icon = "✅" if is_avail else "❌"
            st.markdown(
                f"**{info.get('icon', '')} {info.get('name', eng)}**\n\n"
                f"{status_icon} {'متاح' if is_avail else 'غير متاح'}\n\n"
                f"*{info.get('strengths', '')}*\n\n"
                f"~{info.get('memory', '?')} RAM"
            )

    st.markdown("---")

    # --- Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 رفع ومعالجة",
        "📝 التصحيحات",
        "📊 الإحصائيات",
        "🔍 مقارنة المحركات",
    ])

    # ========================================
    # تبويب 1: رفع ومعالجة
    # ========================================
    with tab1:
        uploaded = st.file_uploader(
            "📤 اختر مسحاً ضوئياً (JPG/PNG)",
            type=["jpg", "jpeg", "png"],
            key="file_uploader"
        )

        if not uploaded:
            st.markdown(
                """
                ### 📤 ارفع مستندك الطبي
                يدعم التطبيق:
                - **5 محركات OCR** مع تجمع ذكي
                - **الخط اليدوي** العربي والإنجليزي
                - **4 استراتيجيات دمج** مختلفة
                - **مقارنة تفصيلية** لنتائج كل محرك

                يتم تحسين الصورة تلقائياً (زيادة التباين + شحذ الحواف) قبل التشغيل.
                """
            )
        else:
            file_path = os.path.join(DIR_UPLOADS, f"{uuid.uuid4().hex}_{uploaded.name}")
            with open(file_path, "wb") as f:
                f.write(uploaded.getbuffer())

            if not selected_engines:
                st.error("❌ اختر محركاً واحداً على الأقل من الشريط الجانبي")
            else:
                # عرض تقدم المعالجة
                progress_placeholder = st.empty()
                log_placeholder = st.empty()

                progress_placeholder.progress(0, text="جاري تهيئة المحركات...")

                # إنشاء نظام التجمع
                ensemble = EnsembleOCR(
                    engines=selected_engines,
                    strategy=strategy,
                    confidence_threshold=confidence_threshold,
                )

                with st.spinner(f"⚙️ تشغيل {len(selected_engines)} محرك OCR ({strategy})..."):
                    # معالجة الصورة
                    pre_path = preprocess_image(file_path)
                    result = ensemble.process_image(pre_path, strategy=strategy)

                    progress_placeholder.progress(100, text="تم!")

                # عرض ملخص الأداء
                st.markdown("### 📊 ملخص الأداء")
                perf_cols = st.columns(3)
                perf_cols[0].metric("⏱️ الوقت الكلي", f"{result.total_time:.2f}s")
                perf_cols[1].metric("📝 الكلمات المدمجة", len(result.words))
                perf_cols[2].metric("🔧 المحركات النشطة", len(result.engines_active))

                # عرض أداء كل محرك
                with st.expander("📋 أداء كل محرك", expanded=False):
                    perf_data = []
                    for name, er in result.engine_results.items():
                        info = EnsembleOCR.ENGINE_DESCRIPTIONS.get(name, {})
                        perf_data.append({
                            "المحرك": f"{info.get('icon', '')} {info.get('name', name)}",
                            "الحالة": "✅ نشط" if er.available else "❌",
                            "عدد الكلمات": len(er.words),
                            "الوقت (ث)": f"{er.processing_time:.2f}",
                        })
                    st.dataframe(
                        pd.DataFrame(perf_data),
                        hide_index=True,
                        use_container_width=True,
                    )

                # حفظ في قاعدة البيانات
                img = Image.open(file_path)
                img_id = save_image_meta(uploaded.name, file_path, img.width, img.height)
                save_words_meta(img_id, result)

                # عرض النتائج
                st.markdown("---")
                col_img, col_results = st.columns([1, 1.3])

                with col_img:
                    st.image(file_path, caption="الصورة الأصلية", use_container_width=True)

                with col_results:
                    st.subheader("📝 جدول التصحيح التفاعلي")
                    st.markdown("💡 *الكلمات منخفضة الثقة أولاً. عدّل النص ثم اضغط حفظ.*")

                    words = get_words(img_id)
                    if not words:
                        st.warning("لم يُعثر على نصوص. جرّب صورة أخرى أو تفعيل المزيد من المحركات.")
                    else:
                        st.success(f"تم استخراج **{len(words)}** كلمة (من {len(selected_engines)} محرك)")

                        # تجهيز DataFrame
                        df_data = []
                        for w in words:
                            engines_str = "، ".join(
                                json.loads(w['engines_used']) if isinstance(w['engines_used'], str) else w.get('engines_used', [])
                            ) if w.get('engines_used') else ""
                            df_data.append({
                                "ID": w["id"],
                                "النص": w["predicted_text"],
                                "الثقة": w["confidence"],
                                "المحركات": engines_str,
                                "الاتفاق": w.get("agreement_count", 1),
                                "الاستراتيجية": w.get("ensemble_strategy", ""),
                            })
                        df = pd.DataFrame(df_data)

                        edited = st.data_editor(
                            df,
                            column_config={
                                "ID": st.column_config.NumberColumn(disabled=True, width="small"),
                                "النص": st.column_config.TextColumn(width="large"),
                                "الثقة": st.column_config.ProgressColumn(
                                    format="%.0f%%",
                                    min_value=0,
                                    max_value=100,
                                    help="درجة ثقة التجمع"
                                ),
                                "المحركات": st.column_config.TextColumn(width="medium"),
                                "الاتفاق": st.column_config.NumberColumn(width="small"),
                                "الاستراتيجية": st.column_config.TextColumn(width="small"),
                            },
                            hide_index=True,
                            use_container_width=True,
                            num_rows="dynamic",
                        )

                        if st.button("💾 حفظ التصحيحات وتوليد بيانات التدريب", type="primary", key="save_new"):
                            progress = st.progress(0)
                            saved = 0
                            for i, (_, row) in enumerate(edited.iterrows()):
                                wid = int(row["ID"])
                                new_text = row["النص"]

                                conn = sqlite3.connect(DB_PATH)
                                c = conn.cursor()
                                c.execute("SELECT predicted_text, confidence FROM words WHERE id=?", (wid,))
                                orig = c.fetchone()
                                conn.close()

                                if orig and new_text != orig[0]:
                                    conn2 = sqlite3.connect(DB_PATH)
                                    c2 = conn2.cursor()
                                    c2.execute("SELECT bbox FROM words WHERE id=?", (wid,))
                                    bbox = json.loads(c2.fetchone()[0])
                                    conn2.close()

                                    crop_p = crop_and_save(bbox, pre_path, wid)
                                    is_gold = orig[1] < 0.65 and len(str(new_text)) > 0
                                    update_word_correction(wid, str(new_text), crop_p, is_gold)
                                    saved += 1

                                progress.progress((i + 1) / len(edited))

                            st.success(f"✅ تم حفظ **{saved}** تصحيح وتوليد القصصات التدريبية!")

    # ========================================
    # تبويب 2: التصحيحات (عرض المستندات السابقة)
    # ========================================
    with tab2:
        docs = get_all_documents()

        if not docs:
            st.info("لا توجد مستندات بعد. ارفع مستند من تبويب 'رفع ومعالجة'.")
        else:
            st.subheader(f"📂 المستندات المحفوظة ({len(docs)})")

            for doc in docs:
                with st.expander(
                    f"📄 {doc['filename']} — {doc['word_count']} كلمة | "
                    f"✅ {doc['corrected_count']} تصحيح | "
                    f"{doc['created_at'][:16]}"
                ):
                    words = get_words(doc["id"])
                    if not words:
                        st.caption("لا توجد كلمات.")
                        continue

                    df_data = []
                    for w in words:
                        engines_str = "، ".join(
                            json.loads(w['engines_used']) if isinstance(w['engines_used'], str) else w.get('engines_used', [])
                        ) if w.get('engines_used') else ""
                        df_data.append({
                            "ID": w["id"],
                            "النص المتوقع": w["predicted_text"],
                            "الثقة": w["confidence"],
                            "النص المصحح": w.get("corrected_text", ""),
                            "المحركات": engines_str,
                            "الاتفاق": w.get("agreement_count", 1),
                        })

                    df = pd.DataFrame(df_data)

                    edited = st.data_editor(
                        df,
                        column_config={
                            "النص المصحح": st.column_config.TextColumn(width="large"),
                        },
                        hide_index=True,
                        use_container_width=True,
                        num_rows="dynamic",
                        key=f"edit_doc_{doc['id']}",
                    )

                    if st.button(
                        f"💾 حفظ تصحيحات '{doc['filename']}'",
                        key=f"save_doc_{doc['id']}"
                    ):
                        saved = 0
                        for _, row in edited.iterrows():
                            wid = int(row["ID"])
                            new_text = row["النص المصحح"]
                            if new_text and str(new_text).strip():
                                crop_path = os.path.join(DIR_CROPS, f"{wid}.png")
                                if not os.path.exists(crop_path):
                                    conn = sqlite3.connect(DB_PATH)
                                    c = conn.cursor()
                                    c.execute("SELECT bbox, predicted_text, confidence FROM words WHERE id=?", (wid,))
                                    orig = c.fetchone()
                                    conn.close()
                                    if orig:
                                        bbox = json.loads(orig[0])
                                        conn2 = sqlite3.connect(DB_PATH)
                                        c2 = conn2.cursor()
                                        c2.execute("SELECT path FROM images WHERE id=?", (doc["id"],))
                                        img_row = c2.fetchone()
                                        conn2.close()
                                        if img_row:
                                            crop_path = crop_and_save(bbox, img_row[0], wid)

                                is_gold = False
                                conn = sqlite3.connect(DB_PATH)
                                c = conn.cursor()
                                c.execute("SELECT confidence FROM words WHERE id=?", (wid,))
                                conf_row = c.fetchone()
                                conn.close()
                                if conf_row and conf_row[0] < 0.65:
                                    is_gold = True

                                update_word_correction(wid, str(new_text), crop_path, is_gold)
                                saved += 1

                        st.success(f"✅ تم حفظ {saved} تصحيح!")

    # ========================================
    # تبويب 3: الإحصائيات
    # ========================================
    with tab3:
        stats = get_stats()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📄 المستندات", stats["total_images"])
        col2.metric("📝 الكلمات", stats["total_words"])
        col3.metric("✅ التصحيحات", stats["corrected_words"])
        col4.metric("⭐ عينات ذهبية", stats["gold_standard"])

        st.markdown("---")

        col5, col6, col7 = st.columns(3)
        with col5:
            st.metric("📊 CER", f"{stats['cer'] * 100:.1f}%")
        with col6:
            st.metric("📊 WER", f"{stats['wer'] * 100:.1f}%")
        with col7:
            st.metric("🔗 متوسط الاتفاق", f"{stats['avg_agreement']:.1f}")

        st.progress(
            min(1.0, stats["avg_confidence"]),
            text=f"متوسط الثقة: {stats['avg_confidence'] * 100:.1f}%"
        )

        if stats["low_confidence"] > 0:
            st.warning(f"⚠️ {stats['low_confidence']} كلمة منخفضة الثقة (< 50%) تحتاج انتباه")

        # توزيع الكلمات حسب حالة المراجعة
        if stats["total_words"] > 0:
            st.markdown("### 📊 توزيع حالات المراجعة")
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("""
                SELECT review_status, COUNT(*) as cnt
                FROM words
                GROUP BY review_status
            """)
            rows = c.fetchall()
            conn.close()

            if rows:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                import matplotlib.font_manager as fm

                try:
                    fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC[wght].ttf')
                    plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
                except Exception:
                    pass
                plt.rcParams['axes.unicode_minus'] = False

                labels_map = {
                    'pending': 'قيد المراجعة',
                    'approved': 'معتمد',
                    'rejected': 'مرفوض',
                    'gold': 'ذهبي',
                }
                labels = [labels_map.get(r[0], r[0]) for r in rows]
                sizes = [r[1] for r in rows]

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.barh(labels, sizes, color=['#f59e0b', '#10b981', '#ef4444', '#eab308'][:len(labels)])
                ax.set_xlabel('عدد الكلمات')
                ax.set_title('توزيع حالات المراجعة')
                plt.tight_layout()
                st.pyplot(fig)

    # ========================================
    # تبويب 4: مقارنة المحركات
    # ========================================
    with tab4:
        st.subheader("🔍 مقارنة أداء المحركات")

        # إحصائيات عامة من engine_logs
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        # ملخص عام لكل محرك
        c.execute("""
            SELECT engine_name,
                   COUNT(*) as total_runs,
                   SUM(CASE WHEN available=1 THEN 1 ELSE 0 END) as successful_runs,
                   AVG(CASE WHEN available=1 THEN word_count ELSE NULL END) as avg_words,
                   AVG(CASE WHEN available=1 THEN processing_time ELSE NULL END) as avg_time,
                   SUM(word_count) as total_words
            FROM engine_logs
            GROUP BY engine_name
        """)
        summary_rows = c.fetchall()

        if not summary_rows:
            st.info("لا توجد بيانات بعد. قم بمعالجة مستند من تبويب 'رفع ومعالجة' لرؤية المقارنة.")
        else:
            comp_data = []
            for row in summary_rows:
                name, total, success, avg_w, avg_t, total_w = row
                info = EnsembleOCR.ENGINE_DESCRIPTIONS.get(name, {})
                comp_data.append({
                    "المحرك": f"{info.get('icon', '')} {info.get('name', name)}",
                    "المهام": total,
                    "النجاح": success,
                    "متوسط الكلمات": f"{avg_w:.0f}" if avg_w else "—",
                    "متوسط الوقت": f"{avg_t:.2f}s" if avg_t else "—",
                    "إجمالي الكلمات": total_w or 0,
                })

            st.dataframe(
                pd.DataFrame(comp_data),
                hide_index=True,
                use_container_width=True,
            )

            # رسم بياني
            st.markdown("### 📈 أداء المحركات")
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # رسم 1: عدد الكلمات لكل محرك
            names = [EnsembleOCR.ENGINE_DESCRIPTIONS.get(r[0], {}).get('name', r[0]) for r in summary_rows]
            word_counts = [r[5] or 0 for r in summary_rows]
            colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'][:len(names)]

            axes[0].bar(names, word_counts, color=colors)
            axes[0].set_ylabel('عدد الكلمات')
            axes[0].set_title('إجمالي الكلمات المستخرجة')
            axes[0].tick_params(axis='x', rotation=15)

            # رسم 2: متوسط الوقت
            times = [r[4] for r in summary_rows if r[4] is not None]
            time_names = [names[i] for i in range(len(summary_rows)) if summary_rows[i][4] is not None]
            time_colors = [colors[i] for i in range(len(summary_rows)) if summary_rows[i][4] is not None]

            if times:
                axes[1].barh(time_names, times, color=time_colors)
                axes[1].set_xlabel('الوقت (ثانية)')
                axes[1].set_title('متوسط وقت المعالجة')

            plt.tight_layout()
            st.pyplot(fig)

            # تفاصيل آخر مستند
            st.markdown("### 📝 تفاصيل آخر مستند")
            c.execute("SELECT id FROM images ORDER BY created_at DESC LIMIT 1")
            last_img = c.fetchone()
            if last_img:
                logs = get_engine_logs(last_img[0])
                for log in logs:
                    info = EnsembleOCR.ENGINE_DESCRIPTIONS.get(log['engine_name'], {})
                    with st.expander(
                        f"{info.get('icon', '')} {info.get('name', log['engine_name'])} — "
                        f"{log['word_count']} كلمة | {log['processing_time']:.2f}s"
                    ):
                        if log.get('raw_results'):
                            try:
                                raw = json.loads(log['raw_results'])
                                if raw.get('words'):
                                    for w in raw['words'][:20]:
                                        st.text(f"  [{w['confidence']:.0%}] {w['text']}")
                                    if len(raw['words']) > 20:
                                        st.caption(f"  ... و {len(raw['words'])-20} كلمة أخرى")
                            except Exception:
                                pass

        conn.close()


def export_training_data():
    """تصدير بيانات التدريب بصيغة JSONL"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""
        SELECT w.id, w.predicted_text, w.corrected_text, w.confidence, w.bbox,
               w.script_class, w.is_gold_standard, i.filename,
               w.ensemble_strategy, w.engines_used, w.agreement_count, w.engine_votes
        FROM words w
        JOIN images i ON w.image_id = i.id
        WHERE w.is_corrected = 1 AND w.corrected_text IS NOT NULL
        ORDER BY w.confidence ASC
    """)
    rows = c.fetchall()
    conn.close()

    if not rows:
        st.warning("لا توجد تصحيحات لتصديرها بعد.")
        return

    export_dir = "exports"
    os.makedirs(export_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_path = os.path.join(export_dir, f"training_data_{timestamp}.jsonl")

    with open(export_path, "w", encoding="utf-8") as f:
        for row in rows:
            record = {
                "word_id": row["id"],
                "predicted_text": row["predicted_text"],
                "corrected_text": row["corrected_text"],
                "confidence": row["confidence"],
                "bbox": json.loads(row["bbox"]),
                "script_class": row["script_class"],
                "is_gold_standard": bool(row["is_gold_standard"]),
                "document": row["filename"],
                "crop_path": os.path.join(DIR_CROPS, f"{row['id']}.png"),
                "ensemble": {
                    "strategy": row["ensemble_strategy"],
                    "engines_used": json.loads(row["engines_used"]) if row["engines_used"] else [],
                    "agreement_count": row["agreement_count"],
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    st.success(f"تم تصدير {len(rows)} سجل إلى `{export_path}`")

    with open(export_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines[:3]:
        st.code(json.dumps(json.loads(line), indent=2, ensure_ascii=False), language="json")


if __name__ == "__main__":
    main()

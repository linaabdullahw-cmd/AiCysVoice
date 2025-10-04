# --- أعلى الملف ---
import os, uuid, time, mimetypes
from flask import Flask, render_template, request, abort
from werkzeug.utils import secure_filename
import joblib, numpy as np
from ai.features import extract_features

app = Flask(__name__)

# 1) قيود الرفع
ALLOWED_EXTENSIONS = {"wav", "mp3", "m4a"}
MAX_MB = 10
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024

# 2) مجلدات الرفع منظمة حسب اليوم
BASE_UPLOAD = "uploads"
os.makedirs(BASE_UPLOAD, exist_ok=True)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def is_probably_audio(filepath):
    # فحص سريع بالامتداد والـ mimetype (ليس ضمان 100%)
    mt, _ = mimetypes.guess_type(filepath)
    return (mt and mt.startswith("audio")) or filepath.lower().endswith((".wav",".mp3",".m4a"))

def unique_save_path(filename):
    day = time.strftime("%Y-%m-%d")
    folder = os.path.join(BASE_UPLOAD, day)
    os.makedirs(folder, exist_ok=True)
    uid = uuid.uuid4().hex[:8]
    safe = secure_filename(filename)
    return os.path.join(folder, f"{uid}__{safe}")

def cleanup_old_uploads(older_than_hours=6):
    cutoff = time.time() - older_than_hours * 3600
    for root, _, files in os.walk(BASE_UPLOAD):
        for f in files:
            p = os.path.join(root, f)
            try:
                if os.path.getmtime(p) < cutoff:
                    os.remove(p)
            except Exception:
                pass

# 3) حمّل الموديل مرة واحدة
MODEL_PATH = os.path.join("ai", "voice_model.pkl")
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# --- الراوتات ---
@app.route("/")
def index():
    cleanup_old_uploads()  # تنظيف خفيف عند كل زيارة
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        abort(400, "No file part")
    file = request.files["file"]
    if file.filename.strip() == "":
        abort(400, "No selected file")
    if not allowed_file(file.filename):
        abort(400, f"Only {', '.join(sorted(ALLOWED_EXTENSIONS))} are allowed")

    # حفظ باسم فريد
    save_path = unique_save_path(file.filename)
    file.save(save_path)

    # فحص أنه فعلاً صوت
    if not is_probably_audio(save_path):
        try:
            os.remove(save_path)
        except Exception:
            pass
        abort(400, "Invalid audio file")

    # التحليل
    try:
        features = extract_features(save_path)
        if model is None:
            result = "⚠ No model found. Please run train_model.py first."
            css_class = "result-fake"
        else:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features)[0]
                score = float(proba[1])  # احتمال التلاعب
            else:
                pred = model.predict(features)[0]
                score = 1.0 if pred == 1 else 0.0

            threshold = 0.5
            label = 1 if score >= threshold else 0
            confidence = (score if label == 1 else (1 - score)) * 100.0

            if label == 0:
                result = f"Original ✅ (Confidence: {confidence:.2f}%)"
                css_class = "result-safe"
            else:
                result = f"Manipulated ❌ (Confidence: {confidence:.2f}%)"
                css_class = "result-fake"
    except Exception as e:
        result = f"Error analyzing file: {e}"
        css_class = "result-fake"

    return render_template("result.html", filename=os.path.basename(save_path), result=result, css_class=css_class)

# أخطاء ودّية
@app.errorhandler(413)
def too_large(e):
    return f"File too large. Max {MAX_MB}MB", 413

if __name__ == "__main__":
    app.run(debug=True)
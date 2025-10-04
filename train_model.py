import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from ai.features import extract_features

REAL = "dataset/real"
FAKE = "dataset/fake"

X, y = [], []

def load_folder(folder, label):
    if not os.path.isdir(folder):
        return
    for fname in os.listdir(folder):
        if fname.lower().endswith((".wav", ".mp3", ".m4a")):
            path = os.path.join(folder, fname)
            try:
                feat = extract_features(path)
                X.append(feat.ravel())
                y.append(label)
            except Exception as e:
                print(f"⚠ Skipped {fname}: {e}")

# تحميل البيانات
load_folder(REAL, 0)  # أصلي
load_folder(FAKE, 1)  # مزيف

X = np.array(X)
y = np.array(y)

if len(X) == 0:
    raise SystemExit("⚠ No data in dataset/real or dataset/fake")

# تقسيم
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Pipeline: Scaler + RandomForest
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
])

# Calibration لتحسين الاحتمالات
model = CalibratedClassifierCV(pipe, cv=3)
model.fit(X_train, y_train)

# تقييم
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("=== Classification Report ===")
print(classification_report(y_test, y_pred))
try:
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))
except Exception:
    pass

# حفظ
os.makedirs("ai", exist_ok=True)
joblib.dump(model, "ai/voice_model.pkl")
print("✅ Saved model to ai/voice_model.pkl")
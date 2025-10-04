# ai/features.py
import numpy as np
import librosa

def extract_features(filepath, sr=16000, n_mfcc=20):
    """
    يستخرج الخصائص من ملف صوتي (MFCC + مشتقات + خصائص طيفية)
    ويعيدها بشكل متجه (vector) يصلح للتدريب أو التنبؤ.
    """
    y, sr = librosa.load(filepath, sr=sr, mono=True)
    if y.size == 0:
        raise ValueError("⚠ Empty or invalid audio file")

    # قص الصمت
    y, _ = librosa.effects.trim(y, top_db=30)

    # ضمان طول أدنى (0.5 ثانية)
    min_len = sr // 2
    if len(y) < min_len:
        y = np.pad(y, (0, min_len - len(y)), mode="constant")

    # MFCC + المشتقات
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # خصائص إضافية
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=y)

    # تجميع (متوسط + انحراف معياري)
    features = []
    for arr in [mfcc, delta, delta2]:
        features.extend(arr.mean(axis=1))
        features.extend(arr.std(axis=1))

    features.extend(chroma.mean(axis=1))
    features.extend([
        zcr.mean(),
        centroid.mean(),
        rolloff.mean(),
        flatness.mean(),
        len(y) / sr  # مدة الملف بالثواني
    ])

    return np.array(features, dtype=np.float32).reshape(1, -1)
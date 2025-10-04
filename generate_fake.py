import os
import librosa
import soundfile as sf
import numpy as np

REAL_DIR = "dataset/real"
FAKE_DIR = "dataset/fake"

os.makedirs(FAKE_DIR, exist_ok=True)

def add_noise(y, noise_level=0.005):
    noise = np.random.randn(len(y)) * noise_level
    return y + noise

def process_file(file_path, sr=16000):
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    base = os.path.splitext(os.path.basename(file_path))[0]
    created = 0

    # 1) بطيء
    slow = librosa.effects.time_stretch(y, rate=0.8)
    sf.write(os.path.join(FAKE_DIR, f"{base}_slow.wav"), slow, sr); created += 1

    # 2) سريع
    fast = librosa.effects.time_stretch(y, rate=1.2)
    sf.write(os.path.join(FAKE_DIR, f"{base}_fast.wav"), fast, sr); created += 1

    # 3) نغمة أعلى
    high = librosa.effects.pitch_shift(y, sr=sr, n_steps=4)
    sf.write(os.path.join(FAKE_DIR, f"{base}_high.wav"), high, sr); created += 1

    # 4) نغمة أعمق
    low = librosa.effects.pitch_shift(y, sr=sr, n_steps=-4)
    sf.write(os.path.join(FAKE_DIR, f"{base}_low.wav"), low, sr); created += 1

    # 5) ضوضاء بسيطة
    noisy = add_noise(y)
    sf.write(os.path.join(FAKE_DIR, f"{base}_noisy.wav"), noisy, sr); created += 1

    return created

if name == "__main__":
    files = [f for f in os.listdir(REAL_DIR) if f.lower().endswith((".wav", ".mp3", ".m4a"))]
    total_created = 0
    if not files:
        print(f"⚠ No audio files found in {REAL_DIR}")
    else:
        for f in files:
            path = os.path.join(REAL_DIR, f)
            try:
                c = process_file(path)
                total_created += c
                print(f"✅ Processed {f} → {c} fake files created")
            except Exception as e:
                print(f"⚠ Skipped {f}: {e}")
        print(f"\n🎉 Done! Generated {total_created} fake files and saved them in: {FAKE_DIR}")
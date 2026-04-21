"""
Retrain anomaly detection model menggunakan scikit-learn yang terinstall saat ini.
Gunakan script ini setiap kali sklearn diupgrade agar model pkl tetap kompatibel.

Usage:
    python retrain_anomaly.py

Output:
    ai-engine/model_anomaly_detection_v2.pkl
    ai-engine/scaler_anomaly_v2.pkl
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import sklearn

print(f"scikit-learn version: {sklearn.__version__}")

DATA_PATH  = "ai-engine/sensor_cleaned.csv"
OUTPUT_DIR = "ai-engine"

CONTAMINATION  = 0.02
RANDOM_STATE   = 42
N_ESTIMATORS   = 100

FEATURES = [
    "temperature", "humidity", "ammonia",
    "hour_of_day", "temp_delta_1h", "humid_delta_1h",
    "nh3_delta_1h", "comfort_index",
]

# ── 1. Load data ──────────────────────────────────────────────
print(f"\n[1] Loading {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH)
print(f"    Rows: {len(df)}")

df.columns = [c.strip().lower() for c in df.columns]

for col in ["temperature_c", "humidity_rh", "nh3_ppm"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["temperature_c", "humidity_rh", "nh3_ppm"])

# ── 2. Feature engineering ────────────────────────────────────
print("[2] Engineering features ...")

# Parse timestamps untuk hour_of_day
if "recorded_at" in df.columns:
    df["recorded_at"] = pd.to_datetime(df["recorded_at"], utc=True, errors="coerce")
    df = df.sort_values("recorded_at").reset_index(drop=True)
    df["hour_of_day"] = df["recorded_at"].dt.hour.astype(float)
else:
    df["hour_of_day"] = 12.0  # fallback jika tidak ada timestamp

# Alias kolom sesuai FEATURES_ANOMALY di app.py
df["temperature"] = df["temperature_c"]
df["humidity"]    = df["humidity_rh"]
df["ammonia"]     = df["nh3_ppm"]

# Comfort index — formula identik dengan app.py
def comfort_index(t, h, n):
    ts = 1 - np.abs(t - 28.0) / 15.0
    hs = 1 - np.abs(h - 80.0) / 35.0
    ns = 1 - (n / 20.0)
    ts = np.clip(ts, 0, 1)
    hs = np.clip(hs, 0, 1)
    ns = np.clip(ns, 0, 1)
    return (ts * 0.35 + hs * 0.35 + ns * 0.30) * 100.0

df["comfort_index"] = comfort_index(
    df["temperature"].values,
    df["humidity"].values,
    df["ammonia"].values,
)

# Delta 1h — persentase perubahan antar baris (sorted by time)
# Di inference, ini dihitung dari buffer 1 jam; di training, pakai diff antar baris
WINDOW = 10  # sliding window untuk rolling reference

def pct_delta(series, window=WINDOW):
    ref = series.shift(window).fillna(series.median())
    delta = (series - ref) / (ref.abs() + 1e-6) * 100.0
    return delta.clip(-100, 100)

df["temp_delta_1h"]  = pct_delta(df["temperature"])
df["humid_delta_1h"] = pct_delta(df["humidity"])
df["nh3_delta_1h"]   = pct_delta(df["ammonia"])

df_feat = df[FEATURES].copy()
df_feat = df_feat.fillna(0.0)

print(f"    Feature matrix: {df_feat.shape}")
print(f"    Features: {FEATURES}")

# ── 3. Fit StandardScaler ─────────────────────────────────────
print("[3] Fitting StandardScaler ...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_feat.values)

# ── 4. Fit IsolationForest ────────────────────────────────────
print(f"[4] Training IsolationForest (contamination={CONTAMINATION}, n_estimators={N_ESTIMATORS}) ...")
iso = IsolationForest(
    n_estimators=N_ESTIMATORS,
    contamination=CONTAMINATION,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
iso.fit(X_scaled)

preds = iso.predict(X_scaled)
n_anomaly = (preds == -1).sum()
print(f"    Anomaly flagged: {n_anomaly}/{len(preds)} ({n_anomaly/len(preds)*100:.1f}%)")

# ── 5. Simpan artefak ─────────────────────────────────────────
model_path  = os.path.join(OUTPUT_DIR, "model_anomaly_detection_v2.pkl")
scaler_path = os.path.join(OUTPUT_DIR, "scaler_anomaly_v2.pkl")

joblib.dump(iso,    model_path,  compress=3)
joblib.dump(scaler, scaler_path, compress=3)

print(f"\n[5] Saved:")
print(f"    {model_path}  ({os.path.getsize(model_path):,} bytes)")
print(f"    {scaler_path} ({os.path.getsize(scaler_path):,} bytes)")
print(f"\nDone. Sekarang rebuild Docker image AI engine untuk memuat model baru.")
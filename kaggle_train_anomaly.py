# ============================================================
# Swiftlet AI Engine - Anomaly Detection Training
# ------------------------------------------------------------
# Output:
#  - model_anomaly_detection_v2.pkl (IsolationForest)
#  - scaler_anomaly_v2.pkl          (StandardScaler)
#
# Algoritma: Hybrid (Rule-Based + Isolation Forest)
#   - Rule-based threshold di app.py (runtime)
#   - IsolationForest untuk deteksi anomali multivariat
#
# Features : temperature, humidity, ammonia, hour_of_day,
#            temp_delta_1h, humid_delta_1h, nh3_delta_1h,
#            comfort_index
#
# Dataset  : ml_training_dataset.csv
# Kolom    : recorded_at/timestamp, temperature_c, humidity_rh, nh3_ppm
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib

# -----------------------------
# 0. Path data & artefak
# -----------------------------
# Kaggle — dataset: real + external (GAMS indoor)
DATA_PATH = "/kaggle/input/swiftledml-merged-external/ml_training_with_external.csv"
MODELS_DIR = "/kaggle/working"

# Lokal (uncomment jika di lokal)
# DATA_PATH = "data/ml_training_with_external.csv"
# MODELS_DIR = "ai-engine"

os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 60)
print("SWIFTLET AI - ANOMALY DETECTION TRAINING (Isolation Forest)")
print("=" * 60)
print(f"Data path : {DATA_PATH}")
print(f"Models dir: {MODELS_DIR}")

# -----------------------------
# KONSTANTA
# -----------------------------
RANDOM_STATE = 42
HISTORY_WINDOW = 10
MIN_DELTA_THRESHOLD = 0.5
SPIKE_Z_THRESHOLD = 3.0
SPIKE_PCT_THRESHOLD = 30.0
CONTAMINATION = 0.02  # 2% — hanya sensor error / nilai impossible

# ============================================================
# 1. LOAD & PRE-CLEAN DATA
# ============================================================
print("\n[1] Loading data...")

df = pd.read_csv(DATA_PATH)
print(f"  Shape raw : {df.shape}")

df.columns = [c.strip().lower() for c in df.columns]

required = ["temperature_c", "humidity_rh", "nh3_ppm"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise RuntimeError(f"Kolom wajib hilang: {missing}")

for col in required:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=required, how="all")
print(f"  Shape after dropna: {df.shape}")

# Handle waktu
if "recorded_at" in df.columns:
    dt = pd.to_datetime(df["recorded_at"], errors="coerce")
elif "timestamp" in df.columns:
    dt = pd.to_datetime(df["timestamp"], errors="coerce")
else:
    dt = pd.Series(pd.date_range("2025-01-01", periods=len(df), freq="h"))

df["dt"] = dt
df = df.sort_values("dt")

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
print("\n[2] Engineering features...")

# 2.1 Base features
df["temperature"] = df["temperature_c"]
df["humidity"]    = df["humidity_rh"]
df["ammonia"]     = df["nh3_ppm"]

# 2.2 Temporal features
df["hour_of_day"] = df["dt"].dt.hour.fillna(12).astype(int)

# 2.3 Delta features (percentage-based)
def safe_pct_change(series):
    """
    Percentage change dengan handling untuk:
    - Division by zero
    - Extreme outliers
    - Small absolute changes (< MIN_DELTA_THRESHOLD)
    """
    abs_diff = series.diff()
    pct = series.pct_change() * 100
    pct = pct.where(abs_diff.abs() >= MIN_DELTA_THRESHOLD, 0.0)
    pct = pct.clip(lower=-100, upper=100)
    return pct.fillna(0.0)

df["temp_delta_1h"]  = safe_pct_change(df["temperature"])
df["humid_delta_1h"] = safe_pct_change(df["humidity"])
df["nh3_delta_1h"]   = safe_pct_change(df["ammonia"])

print("  Delta stats (percentage-based):")
print(f"    temp_delta  - mean: {df['temp_delta_1h'].mean():.2f}%, std: {df['temp_delta_1h'].std():.2f}%")
print(f"    humid_delta - mean: {df['humid_delta_1h'].mean():.2f}%, std: {df['humid_delta_1h'].std():.2f}%")
print(f"    nh3_delta   - mean: {df['nh3_delta_1h'].mean():.2f}%, std: {df['nh3_delta_1h'].std():.2f}%")

# 2.4 Spike Detection Features (z-score based, sesuai API doc)
def calculate_z_score(series, window=HISTORY_WINDOW):
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    rolling_std = series.rolling(window=window, min_periods=1).std()
    rolling_std = rolling_std.replace(0, np.nan).fillna(1e-6)
    return (series - rolling_mean) / rolling_std

def detect_spike(series, window=HISTORY_WINDOW):
    """Spike = z-score > 3 ATAU pct_change > 30%"""
    z_score = calculate_z_score(series, window)
    pct_change = safe_pct_change(series)
    return ((z_score.abs() > SPIKE_Z_THRESHOLD) |
            (pct_change.abs() > SPIKE_PCT_THRESHOLD)).astype(int)

df["temp_spike"]  = detect_spike(df["temperature"])
df["humid_spike"] = detect_spike(df["humidity"])
df["nh3_spike"]   = detect_spike(df["ammonia"])
df["any_spike"]   = ((df["temp_spike"] | df["humid_spike"] | df["nh3_spike"])).astype(int)

print(f"\n  Spike detection (z>{SPIKE_Z_THRESHOLD} OR pct>{SPIKE_PCT_THRESHOLD}%):")
print(f"    temp_spike  : {df['temp_spike'].sum()} ({df['temp_spike'].mean()*100:.2f}%)")
print(f"    humid_spike : {df['humid_spike'].sum()} ({df['humid_spike'].mean()*100:.2f}%)")
print(f"    nh3_spike   : {df['nh3_spike'].sum()} ({df['nh3_spike'].mean()*100:.2f}%)")
print(f"    any_spike   : {df['any_spike'].sum()} ({df['any_spike'].mean()*100:.2f}%)")

# 2.5 Comfort Index
def calculate_comfort_index(temperature, humidity, ammonia):
    temp_score  = max(0.0, min(1.0, 1 - abs(temperature - 28.0) / 15.0))
    humid_score = max(0.0, min(1.0, 1 - abs(humidity - 80.0) / 35.0))
    nh3_score   = max(0.0, min(1.0, 1 - (ammonia / 20.0)))
    return round((temp_score * 0.35 + humid_score * 0.35 + nh3_score * 0.30) * 100.0, 2)

df["comfort_index"] = df[["temperature", "humidity", "ammonia"]].apply(
    lambda row: calculate_comfort_index(row["temperature"], row["humidity"], row["ammonia"]),
    axis=1
)

# Drop NaN
FEATURES_ANOMALY = [
    "temperature", "humidity", "ammonia",
    "hour_of_day", "temp_delta_1h", "humid_delta_1h",
    "nh3_delta_1h", "comfort_index"
]

df = df.dropna(subset=FEATURES_ANOMALY).reset_index(drop=True)
print(f"\n  Shape after features: {df.shape}")

# ============================================================
# 3. TRAIN ISOLATION FOREST
# ============================================================
print("\n[3] Training Isolation Forest...")
print(f"  Features     : {FEATURES_ANOMALY}")
print(f"  Contamination: {CONTAMINATION}")
print(f"  Samples      : {len(df)}")

X_anom = df[FEATURES_ANOMALY].copy()

# StandardScaler (disimpan sebagai scaler_anomaly_v2.pkl)
scaler_anom = StandardScaler()
X_anom_scaled = scaler_anom.fit_transform(X_anom)

iso = IsolationForest(
    n_estimators=200,
    contamination=CONTAMINATION,
    max_samples=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
iso.fit(X_anom_scaled)

# ============================================================
# 4. EVALUASI
# ============================================================
print("\n[4] Evaluasi model...")

preds = iso.predict(X_anom_scaled)         # 1 = normal, -1 = anomaly
scores = iso.decision_function(X_anom_scaled)

vals, counts = np.unique(preds, return_counts=True)
dist = {int(v): int(c) for v, c in zip(vals, counts)}
n_anomaly = int((preds == -1).sum())
n_normal  = int((preds == 1).sum())

print(f"  Distribusi: normal={n_normal}, anomaly={n_anomaly} ({n_anomaly/len(preds)*100:.1f}%)")

# Score statistics
print(f"\n  Anomaly score stats:")
print(f"    Mean  : {scores.mean():.4f}")
print(f"    Std   : {scores.std():.4f}")
print(f"    Min   : {scores.min():.4f}")
print(f"    Max   : {scores.max():.4f}")

# Analisis: apakah anomali yang terdeteksi masuk akal?
df["anom_pred"]  = preds
df["anom_score"] = scores
df["is_anomaly"] = (preds == -1).astype(int)

# Cross-check dengan rule-based flags
rule_anomaly = (
    (df["ammonia"] >= 20) |
    (df["humidity"] < 0) | (df["humidity"] > 100) |
    (df["temperature"] < 10) | (df["temperature"] > 45)
)
rule_count = rule_anomaly.sum()
both_count = (rule_anomaly & (df["is_anomaly"] == 1)).sum()

print(f"\n  Rule-based anomalies   : {rule_count}")
print(f"  IsoForest anomalies   : {n_anomaly}")
print(f"  Both (overlap)        : {both_count}")

# Sensor stats untuk anomali vs normal
print("\n  Sensor stats - ANOMALY vs NORMAL:")
for col in ["temperature", "humidity", "ammonia"]:
    anom_mean = df[df["is_anomaly"] == 1][col].mean()
    norm_mean = df[df["is_anomaly"] == 0][col].mean()
    print(f"    {col:12s}: anomaly_mean={anom_mean:.2f}, normal_mean={norm_mean:.2f}")

# Spike overlap
spike_anomaly_overlap = (df["any_spike"] & df["is_anomaly"]).sum()
print(f"\n  Spike + Anomaly overlap: {spike_anomaly_overlap}")

# ============================================================
# 5. SIMPAN ARTEFAK
# ============================================================
print("\n[5] Saving artifacts...")

path_anom_model  = os.path.join(MODELS_DIR, "model_anomaly_detection_v2.pkl")
path_anom_scaler = os.path.join(MODELS_DIR, "scaler_anomaly_v2.pkl")

joblib.dump(iso, path_anom_model, compress=3)
joblib.dump(scaler_anom, path_anom_scaler, compress=3)

print(f"  ✓ {path_anom_model}")
print(f"  ✓ {path_anom_scaler}")

# ============================================================
# 6. QUICK INFERENCE TEST
# ============================================================
print("\n[6] Inference test...")

test_cases = [
    {"label": "NORMAL (good conditions)",
     "temperature": 28.5, "humidity": 78.0, "ammonia": 5.0,
     "hour_of_day": 10, "temp_delta_1h": 0.5, "humid_delta_1h": -0.3,
     "nh3_delta_1h": 0.2, "comfort_index": calculate_comfort_index(28.5, 78.0, 5.0)},
    {"label": "ANOMALY (high NH3)",
     "temperature": 29.0, "humidity": 75.0, "ammonia": 40.0,
     "hour_of_day": 14, "temp_delta_1h": 1.0, "humid_delta_1h": -2.0,
     "nh3_delta_1h": 50.0, "comfort_index": calculate_comfort_index(29.0, 75.0, 40.0)},
    {"label": "ANOMALY (extreme temp)",
     "temperature": 42.0, "humidity": 50.0, "ammonia": 15.0,
     "hour_of_day": 13, "temp_delta_1h": 20.0, "humid_delta_1h": -10.0,
     "nh3_delta_1h": 5.0, "comfort_index": calculate_comfort_index(42.0, 50.0, 15.0)},
]

for tc in test_cases:
    lbl = tc.pop("label")
    X_row = pd.DataFrame([tc], columns=FEATURES_ANOMALY)
    X_scaled = scaler_anom.transform(X_row)
    pred = int(iso.predict(X_scaled)[0])
    score = float(iso.decision_function(X_scaled)[0])
    result = "NORMAL" if pred == 1 else "ANOMALY"
    print(f"  {lbl}")
    print(f"    → Prediction: {result} | Score: {score:.4f}")

# ============================================================
print("\n" + "=" * 60)
print("ANOMALY DETECTION TRAINING COMPLETE")
print("=" * 60)
print(f"\nOutput files:")
print(f"  - model_anomaly_detection_v2.pkl (IsolationForest)")
print(f"  - scaler_anomaly_v2.pkl          (StandardScaler)")
print(f"\nAnomaly rate: {n_anomaly/len(preds)*100:.1f}% ({n_anomaly}/{len(preds)})")
print("=" * 60)

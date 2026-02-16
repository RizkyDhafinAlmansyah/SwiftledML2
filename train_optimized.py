# ============================================================
# Swiftlet AI Engine - Optimized Training Pipeline
# ============================================================
# Enhanced training with:
#  - Multi-resolution rolling features
#  - Time-series aware cross-validation
#  - Hyperparameter optimization
#  - Better anti-overfitting techniques
# ============================================================

import os
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.metrics import classification_report, mean_absolute_error, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 0. CONFIGURATION
# ============================================================

# Paths
DATA_PATH = "data/ml_training_dataset.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Training config
RANDOM_STATE = 42
TEST_SIZE = 0.2
USE_TIME_SERIES_SPLIT = True
N_CV_SPLITS = 5
HYPERPARAMETER_TUNING = False  # Set True for grid search (slower)

# Model parameters (anti-overfitting)
MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_leaf": 50,
    "min_samples_split": 100,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# Rolling window config (matches buffer_manager)
WINDOW_10MIN = 10   # 10 readings at 1 min interval
WINDOW_30MIN = 30
WINDOW_60MIN = 60

# Feature sets
FEATURES_GRADE = [
    "temperature", "humidity", "ammonia",
    "temp_avg_1h", "humid_avg_1h", "nh3_avg_1h",
    "comfort_index", "is_daytime"
]

FEATURES_ANOMALY = [
    "temperature", "humidity", "ammonia",
    "hour_of_day", "temp_delta_1h", "humid_delta_1h",
    "nh3_delta_1h", "comfort_index"
]

FEATURES_PUMP_STATE = [
    "temperature", "humidity", "ammonia",
    "temp_avg_1h", "humid_avg_1h", "humid_delta_1h",
    "hour_of_day"
]

FEATURES_PUMP_DURATION = [
    "temperature", "humidity", "ammonia",
    "temp_avg_1h", "humid_avg_1h", "humid_delta_1h",
    "hour_of_day", "comfort_index"
]

# ============================================================
# 1. LOAD & PREPROCESS DATA
# ============================================================

print("=" * 60)
print("OPTIMIZED ML TRAINING PIPELINE")
print("=" * 60)

print("\n[1] Loading data...")
df = pd.read_csv(DATA_PATH)
df.columns = [c.strip().lower() for c in df.columns]
print(f"   Raw data: {len(df)} rows, {len(df.columns)} columns")

# Ensure required columns exist
required = ["temperature_c", "humidity_rh", "nh3_ppm"]
for col in required:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=required, how="all")
print(f"   After dropna: {len(df)} rows")

# Rename columns
df["temperature"] = df["temperature_c"]
df["humidity"] = df["humidity_rh"]
df["ammonia"] = df["nh3_ppm"]

# Handle timestamp
if "recorded_at" in df.columns:
    df["dt"] = pd.to_datetime(df["recorded_at"], errors="coerce")
elif "timestamp" in df.columns:
    df["dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
else:
    df["dt"] = pd.date_range("2025-01-01", periods=len(df), freq="T")

df = df.sort_values("dt").reset_index(drop=True)

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

print("\n[2] Engineering features...")

# Temporal features
df["hour_of_day"] = df["dt"].dt.hour
df["is_daytime"] = ((df["hour_of_day"] >= 6) & (df["hour_of_day"] < 18)).astype(int)
df["day_of_week"] = df["dt"].dt.dayofweek

# Multi-resolution rolling averages
for window in [WINDOW_10MIN, WINDOW_30MIN, WINDOW_60MIN]:
    suffix = f"_{window}min" if window < 60 else "_1h"
    df[f"temp_avg{suffix}"] = df["temperature"].rolling(window=window, min_periods=1).mean()
    df[f"humid_avg{suffix}"] = df["humidity"].rolling(window=window, min_periods=1).mean()
    df[f"nh3_avg{suffix}"] = df["ammonia"].rolling(window=window, min_periods=1).mean()

# Delta features (percentage change)
def safe_pct_change(series, min_threshold=0.5):
    """Calculate percentage change with safety checks."""
    abs_diff = series.diff()
    pct = series.pct_change() * 100
    pct = pct.where(abs_diff.abs() >= min_threshold, 0.0)
    pct = pct.clip(lower=-100, upper=100)
    return pct.fillna(0.0)

df["temp_delta_1h"] = safe_pct_change(df["temperature"])
df["humid_delta_1h"] = safe_pct_change(df["humidity"])
df["nh3_delta_1h"] = safe_pct_change(df["ammonia"])

# Standard deviation (volatility) over 30 min
df["temp_std_30min"] = df["temperature"].rolling(window=WINDOW_30MIN, min_periods=2).std().fillna(0)
df["humid_std_30min"] = df["humidity"].rolling(window=WINDOW_30MIN, min_periods=2).std().fillna(0)
df["nh3_std_30min"] = df["ammonia"].rolling(window=WINDOW_30MIN, min_periods=2).std().fillna(0)

# Comfort index
def calculate_comfort_index(temp, humid, nh3):
    temp_score = max(0, min(1, 1 - abs(temp - 28) / 15))
    humid_score = max(0, min(1, 1 - abs(humid - 80) / 35))
    nh3_score = max(0, min(1, 1 - (nh3 / 20)))
    return (temp_score * 0.35 + humid_score * 0.35 + nh3_score * 0.30) * 100

df["comfort_index"] = df.apply(
    lambda r: calculate_comfort_index(r["temperature"], r["humidity"], r["ammonia"]),
    axis=1
)

# Drop rows with NaN in computed features
feature_cols = [
    "temperature", "humidity", "ammonia",
    "hour_of_day", "is_daytime",
    "temp_avg_1h", "humid_avg_1h", "nh3_avg_1h",
    "temp_delta_1h", "humid_delta_1h", "nh3_delta_1h",
    "comfort_index"
]
df = df.dropna(subset=feature_cols)
df = df.reset_index(drop=True)

print(f"   Features computed: {len(df)} samples ready")

# ============================================================
# 3. LABEL GENERATION
# ============================================================

print("\n[3] Generating labels...")

# Grade label
def label_grade(row):
    ci = row["comfort_index"]
    temp = row["temperature"]
    rh = row["humidity"]
    nh3 = row["ammonia"]
    
    if (nh3 > 35) or (temp < 20 or temp > 35) or (rh < 60 or rh > 95):
        return "buruk"
    if ci > 70:
        return "bagus"
    elif ci >= 50:
        return "sedang"
    else:
        return "buruk"

df["grade"] = df.apply(label_grade, axis=1)

# Pump state label
def label_pump_state(row):
    h = row["humidity"]
    t = row["temperature"]
    dh = row["humid_delta_1h"]
    
    if h < 75:
        return 1
    if (t > 30) and (h < 80):
        return 1
    if dh < -5:  # Dropping humidity
        return 1
    return 0

df["pump_state_label"] = df.apply(label_pump_state, axis=1)

# Pump duration label
def label_pump_duration(row):
    humidity = row["humidity"]
    temperature = row["temperature"]
    is_daytime = row["is_daytime"]
    humid_delta_1h = row["humid_delta_1h"]
    
    humid_gap = max(0.0, 80.0 - humidity)
    base_duration = humid_gap * 0.4
    temp_factor = 1.0 + max(0.0, temperature - 28.0) * 0.1
    time_factor = 1.2 if is_daytime == 1 else 1.0
    trend_factor = 1.0 + max(0.0, -humid_delta_1h) * 0.01
    
    duration = base_duration * temp_factor * time_factor * trend_factor
    return min(duration, 30.0)

df["pump_duration_label"] = df.apply(label_pump_duration, axis=1)

print(f"   Grade distribution:\n{df['grade'].value_counts()}")
print(f"   Pump state: ON={df['pump_state_label'].sum()}, OFF={(df['pump_state_label']==0).sum()}")
print(f"   Pump duration: mean={df['pump_duration_label'].mean():.2f}, max={df['pump_duration_label'].max():.2f}")

# ============================================================
# 4. TRAIN GRADE PREDICTION MODEL
# ============================================================

print("\n" + "=" * 60)
print("[4] TRAINING GRADE PREDICTION MODEL")
print("=" * 60)

X_grade = df[FEATURES_GRADE].copy()
y_grade = df["grade"].copy()

le_grade = LabelEncoder()
y_grade_enc = le_grade.fit_transform(y_grade)

if USE_TIME_SERIES_SPLIT:
    print("   Using TimeSeriesSplit for validation...")
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    
    scores = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_grade)):
        X_tr, X_val = X_grade.iloc[train_idx], X_grade.iloc[val_idx]
        y_tr, y_val = y_grade_enc[train_idx], y_grade_enc[val_idx]
        
        model = RandomForestClassifier(**MODEL_PARAMS)
        model.fit(X_tr, y_tr)
        score = model.score(X_val, y_val)
        scores.append(score)
        print(f"      Fold {fold+1}: accuracy = {score:.4f}")
    
    print(f"   Mean CV accuracy: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

# Final training on full data with holdout test
X_train, X_test, y_train, y_test = train_test_split(
    X_grade, y_grade_enc, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_grade_enc
)

clf_grade = RandomForestClassifier(**MODEL_PARAMS)
clf_grade.fit(X_train, y_train)

y_pred = clf_grade.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n   Test accuracy: {accuracy:.4f}")
print(f"\n   Classification Report:")
print(classification_report(y_test, y_pred, target_names=list(le_grade.classes_), zero_division=0))

# Save model
path_grade_model = os.path.join(MODELS_DIR, "model_grade_panen_v2.pkl")
path_grade_encoder = os.path.join(MODELS_DIR, "label_encoder_grade_v2.pkl")
joblib.dump(clf_grade, path_grade_model, compress=3)
joblib.dump(le_grade, path_grade_encoder, compress=3)
print(f"   Saved: {path_grade_model}")
print(f"   Saved: {path_grade_encoder}")

# ============================================================
# 5. TRAIN ANOMALY DETECTION MODEL
# ============================================================

print("\n" + "=" * 60)
print("[5] TRAINING ANOMALY DETECTION MODEL")
print("=" * 60)

X_anom = df[FEATURES_ANOMALY].copy()

scaler_anom = StandardScaler()
X_anom_scaled = scaler_anom.fit_transform(X_anom)

iso = IsolationForest(
    n_estimators=MODEL_PARAMS["n_estimators"],
    contamination=0.15,
    max_samples=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
iso.fit(X_anom_scaled)

preds = iso.predict(X_anom_scaled)
n_anomaly = (preds == -1).sum()
n_normal = (preds == 1).sum()
print(f"   Normal: {n_normal}, Anomaly: {n_anomaly} ({n_anomaly/len(preds)*100:.1f}%)")

path_anom_model = os.path.join(MODELS_DIR, "model_anomaly_detection_v2.pkl")
path_anom_scaler = os.path.join(MODELS_DIR, "scaler_anomaly_v2.pkl")
joblib.dump(iso, path_anom_model, compress=3)
joblib.dump(scaler_anom, path_anom_scaler, compress=3)
print(f"   Saved: {path_anom_model}")
print(f"   Saved: {path_anom_scaler}")

# ============================================================
# 6. TRAIN PUMP STATE MODEL
# ============================================================

print("\n" + "=" * 60)
print("[6] TRAINING PUMP STATE MODEL")
print("=" * 60)

X_pump_state = df[FEATURES_PUMP_STATE].copy()
y_pump_state = df["pump_state_label"].copy()

if y_pump_state.nunique() < 2:
    print("   WARNING: Only one class in pump state, skipping stratify")
    X_train_ps, X_test_ps, y_train_ps, y_test_ps = train_test_split(
        X_pump_state, y_pump_state, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
else:
    X_train_ps, X_test_ps, y_train_ps, y_test_ps = train_test_split(
        X_pump_state, y_pump_state, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_pump_state
    )

clf_pump_state = RandomForestClassifier(**MODEL_PARAMS)
clf_pump_state.fit(X_train_ps, y_train_ps)

y_pred_ps = clf_pump_state.predict(X_test_ps)
accuracy_ps = accuracy_score(y_test_ps, y_pred_ps)
print(f"   Test accuracy: {accuracy_ps:.4f}")
print(f"   Classification Report:")
print(classification_report(y_test_ps, y_pred_ps, zero_division=0))

path_pump_state = os.path.join(MODELS_DIR, "model_pump_state_v2.pkl")
joblib.dump(clf_pump_state, path_pump_state, compress=3)
print(f"   Saved: {path_pump_state}")

# ============================================================
# 7. TRAIN PUMP DURATION MODEL
# ============================================================

print("\n" + "=" * 60)
print("[7] TRAINING PUMP DURATION MODEL")
print("=" * 60)

X_pump_dur = df[FEATURES_PUMP_DURATION].copy()
y_pump_dur = df["pump_duration_label"].copy()

X_train_pd, X_test_pd, y_train_pd, y_test_pd = train_test_split(
    X_pump_dur, y_pump_dur, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

reg_pump_dur = RandomForestRegressor(**MODEL_PARAMS)
reg_pump_dur.fit(X_train_pd, y_train_pd)

y_pred_pd = reg_pump_dur.predict(X_test_pd)
mae = mean_absolute_error(y_test_pd, y_pred_pd)
print(f"   Test MAE: {mae:.3f} minutes")

path_pump_duration = os.path.join(MODELS_DIR, "model_pump_duration_v2.pkl")
joblib.dump(reg_pump_dur, path_pump_duration, compress=3)
print(f"   Saved: {path_pump_duration}")

# ============================================================
# 8. SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(f"""
Models saved to: {MODELS_DIR}/
  - model_grade_panen_v2.pkl      (Grade prediction)
  - label_encoder_grade_v2.pkl    (Label encoder)
  - model_anomaly_detection_v2.pkl (Anomaly detection)
  - scaler_anomaly_v2.pkl         (Anomaly scaler)
  - model_pump_state_v2.pkl       (Pump ON/OFF)
  - model_pump_duration_v2.pkl    (Pump duration)

Training Parameters:
  - Test size: {TEST_SIZE}
  - CV folds: {N_CV_SPLITS}
  - Time-series split: {USE_TIME_SERIES_SPLIT}
  - Max depth: {MODEL_PARAMS['max_depth']}
  - Min samples leaf: {MODEL_PARAMS['min_samples_leaf']}

Data Summary:
  - Total samples: {len(df)}
  - Features per model: 7-8
  - Grade accuracy: {accuracy:.2%}
  - Pump state accuracy: {accuracy_ps:.2%}
  - Pump duration MAE: {mae:.2f} min
""")

print("=" * 60)

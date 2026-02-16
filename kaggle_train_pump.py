# ============================================================
# Swiftlet AI Engine - Pump Automation Training (LightGBM)
# ------------------------------------------------------------
# Output:
#  - model_pump_state_v2.pkl    (LightGBM Classifier: ON/OFF)
#  - model_pump_duration_v2.pkl (LightGBM Regressor: durasi detik)
#
# Pendekatan: Synthetic labels yang "smarter" dari pure rule,
# mempertimbangkan:
#  - Comfort index & trend sensor
#  - Waktu hari (pola humidity siang vs malam)
#  - Cost-awareness (penalize unnecessary spray)
#  - Humidity trend (skip spray jika humidity sudah naik)
#
# Dataset  : ml_training_dataset.csv (historical + synthetic)
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
import lightgbm as lgb
import joblib

# -----------------------------
# 0. Path data & artefak
# -----------------------------
# Kaggle
DATA_PATH = "/kaggle/input/sensor-syntesis/ml_training_dataset.csv"
MODELS_DIR = "/kaggle/working"

# Lokal (uncomment jika di lokal)
# DATA_PATH = "data/ml_training_dataset.csv"
# MODELS_DIR = "models"

os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 60)
print("SWIFTLET AI - PUMP AUTOMATION TRAINING (LightGBM)")
print("=" * 60)
print(f"Data path : {DATA_PATH}")
print(f"Models dir: {MODELS_DIR}")

# -----------------------------
# KONSTANTA
# -----------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
HISTORY_WINDOW = 10
MIN_DELTA_THRESHOLD = 0.5

# Threshold sprayer (bisa di-tune)
TEMP_IDEAL_MIN = 26.0
TEMP_IDEAL_MAX = 30.0
HUMID_IDEAL_MIN = 75.0
HUMID_IDEAL_MAX = 85.0
NH3_WARN = 15.0
NH3_CRITICAL = 25.0

# Durasi constraints
MAX_SPRAY_DURATION = 30.0  # detik
MIN_SPRAY_DURATION = 5.0   # detik

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

# 2.2 Temporal
df["hour_of_day"] = df["dt"].dt.hour.fillna(12).astype(int)
df["is_daytime"]  = ((df["hour_of_day"] >= 6) & (df["hour_of_day"] < 18)).astype(int)

# 2.3 Rolling averages
ROLL_WINDOW = HISTORY_WINDOW
df["temp_avg_1h"]  = df["temperature"].rolling(window=ROLL_WINDOW, min_periods=1).mean()
df["humid_avg_1h"] = df["humidity"].rolling(window=ROLL_WINDOW, min_periods=1).mean()
df["nh3_avg_1h"]   = df["ammonia"].rolling(window=ROLL_WINDOW, min_periods=1).mean()

# 2.4 Delta features (percentage-based)
def safe_pct_change(series):
    abs_diff = series.diff()
    pct = series.pct_change() * 100
    pct = pct.where(abs_diff.abs() >= MIN_DELTA_THRESHOLD, 0.0)
    pct = pct.clip(lower=-100, upper=100)
    return pct.fillna(0.0)

df["temp_delta_1h"]  = safe_pct_change(df["temperature"])
df["humid_delta_1h"] = safe_pct_change(df["humidity"])
df["nh3_delta_1h"]   = safe_pct_change(df["ammonia"])

# 2.5 Comfort Index
def calculate_comfort_index(temperature, humidity, ammonia):
    temp_score  = max(0.0, min(1.0, 1 - abs(temperature - 28.0) / 15.0))
    humid_score = max(0.0, min(1.0, 1 - abs(humidity - 80.0) / 35.0))
    nh3_score   = max(0.0, min(1.0, 1 - (ammonia / 20.0)))
    return round((temp_score * 0.35 + humid_score * 0.35 + nh3_score * 0.30) * 100.0, 2)

df["comfort_index"] = df.apply(
    lambda row: calculate_comfort_index(row["temperature"], row["humidity"], row["ammonia"]),
    axis=1
)

# 2.6 Humidity trend (apakah naik/turun dalam window terakhir)
df["humid_trend_slope"] = (
    df["humidity"].rolling(window=ROLL_WINDOW, min_periods=2)
    .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0.0, raw=False)
    .fillna(0.0)
)

# Drop NaN
feature_cols_all = [
    "temperature", "humidity", "ammonia",
    "temp_avg_1h", "humid_avg_1h", "humid_delta_1h",
    "hour_of_day", "comfort_index", "is_daytime",
    "humid_trend_slope"
]
df = df.dropna(subset=feature_cols_all).reset_index(drop=True)
print(f"  Shape after features: {df.shape}")

# ============================================================
# 3. SMART SYNTHETIC LABELS
# ============================================================
print("\n[3] Generating smart synthetic labels...")

# ----- 3A. Pump State Label (ON/OFF) -----
def smart_pump_state(row):
    """
    Label ON/OFF yang lebih cerdas dari pure threshold:
    - Pertimbangkan comfort index (bukan cuma humidity)
    - Pertimbangkan trend (jika humidity sudah naik → skip spray)
    - Pertimbangkan waktu hari (malam hari lingkungan stabil)
    - Cost-aware: jangan spray jika tidak perlu
    """
    temp  = row["temperature"]
    humid = row["humidity"]
    nh3   = row["ammonia"]
    ci    = row["comfort_index"]
    trend = row["humid_trend_slope"]
    hour  = row["hour_of_day"]
    h_avg = row["humid_avg_1h"]

    # --- PASTI ON: kondisi kritis ---
    # NH3 sangat tinggi → spray untuk dispersi
    if nh3 >= NH3_CRITICAL:
        return 1
    # Suhu sangat tinggi + humidity rendah
    if temp >= 33 and humid < 70:
        return 1

    # --- PASTI OFF: kondisi sudah baik ---
    # Humidity sudah ideal dan suhu OK
    if HUMID_IDEAL_MIN <= humid <= HUMID_IDEAL_MAX and TEMP_IDEAL_MIN <= temp <= TEMP_IDEAL_MAX:
        return 0
    # Comfort index tinggi → tidak perlu spray
    if ci >= 75:
        return 0

    # --- SMART DECISIONS ---
    # Jika humidity sedang naik (trend positif) → skip spray (hemat listrik)
    if trend > 0.3 and humid >= 70:
        return 0

    # Humidity rendah dan suhu di atas ideal
    if humid < HUMID_IDEAL_MIN and temp > TEMP_IDEAL_MAX:
        return 1

    # Humidity sangat rendah
    if humid < 65:
        return 1

    # NH3 mulai tinggi dan comfort turun
    if nh3 >= NH3_WARN and ci < 60:
        return 1

    # Siang hari, humidity turun → preventif spray
    if row["is_daytime"] == 1 and humid < 72 and trend < -0.2:
        return 1

    # Default: OFF (hemat listrik)
    return 0


df["pump_state_label"] = df.apply(smart_pump_state, axis=1)

print("  Pump state distribution:")
vc = df["pump_state_label"].value_counts()
print(f"    OFF (0): {vc.get(0, 0)} ({vc.get(0, 0)/len(df)*100:.1f}%)")
print(f"    ON  (1): {vc.get(1, 0)} ({vc.get(1, 0)/len(df)*100:.1f}%)")

if df["pump_state_label"].nunique() < 2:
    print("  [WARN] Hanya 1 kelas! Adjusting thresholds...")
    # Fallback: gunakan median split
    median_ci = df["comfort_index"].median()
    df["pump_state_label"] = (df["comfort_index"] < median_ci).astype(int)
    vc = df["pump_state_label"].value_counts()
    print(f"    OFF (0): {vc.get(0, 0)} | ON (1): {vc.get(1, 0)} (after fallback)")

# ----- 3B. Pump Duration Label (detik) -----
def smart_pump_duration(row):
    """
    Durasi spray yang adaptif:
    - Humidity gap menentukan base duration
    - Suhu tinggi → spray lebih lama
    - Siang hari → spray sedikit lebih lama (evaporasi cepat)
    - Trend humidity turun → spray lebih lama
    - NH3 tinggi → spray lebih lama (untuk dispersi)
    - Tapi jika pump_state = OFF → durasi = 0
    """
    if row["pump_state_label"] == 0:
        return 0.0

    humid = row["humidity"]
    temp  = row["temperature"]
    nh3   = row["ammonia"]
    trend = row["humid_trend_slope"]

    # Base: dari humidity gap (target 80%)
    humid_gap = max(0.0, 80.0 - humid)
    base_duration = humid_gap * 0.5  # 0.5 detik per 1% gap

    # Temperature factor: semakin panas, spray lebih lama
    temp_factor = 1.0 + max(0.0, temp - TEMP_IDEAL_MAX) * 0.08

    # Time factor: siang hari evaporasi lebih cepat
    time_factor = 1.15 if row["is_daytime"] == 1 else 1.0

    # Trend factor: jika humidity turun cepat, spray lebih lama
    # trend < 0 berarti humidity turun
    trend_factor = 1.0 + max(0.0, -trend) * 0.15

    # NH3 factor: NH3 tinggi → butuh lebih banyak dispersi
    nh3_factor = 1.0 + max(0.0, nh3 - NH3_WARN) * 0.03

    duration = base_duration * temp_factor * time_factor * trend_factor * nh3_factor

    # Clamp
    duration = max(MIN_SPRAY_DURATION, min(duration, MAX_SPRAY_DURATION))

    return round(duration, 1)


df["pump_duration_label"] = df.apply(smart_pump_duration, axis=1)

# Stats
on_mask = df["pump_state_label"] == 1
if on_mask.sum() > 0:
    print(f"\n  Pump duration stats (hanya saat ON):")
    dur_on = df.loc[on_mask, "pump_duration_label"]
    print(f"    Mean  : {dur_on.mean():.1f} detik")
    print(f"    Median: {dur_on.median():.1f} detik")
    print(f"    Min   : {dur_on.min():.1f} detik")
    print(f"    Max   : {dur_on.max():.1f} detik")

# ============================================================
# 4. TRAIN PUMP STATE MODEL (LightGBM Classifier)
# ============================================================
print("\n[4] Training Pump State (LightGBM Classifier)...")

FEATURES_PUMP_STATE = [
    "temperature", "humidity", "ammonia",
    "temp_avg_1h", "humid_avg_1h", "humid_delta_1h",
    "hour_of_day"
]

X_ps = df[FEATURES_PUMP_STATE].copy()
y_ps = df["pump_state_label"].copy()

print(f"  Features: {FEATURES_PUMP_STATE}")
print(f"  Samples : {len(X_ps)}")

Xs_train, Xs_test, ys_train, ys_test = train_test_split(
    X_ps, y_ps, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_ps
)

# Hitung scale_pos_weight untuk handle imbalanced classes
n_neg = (ys_train == 0).sum()
n_pos = (ys_train == 1).sum()
scale_weight = n_neg / max(n_pos, 1)

clf_pump = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=200,
    max_depth=8,
    num_leaves=31,
    min_child_samples=50,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_weight,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)

clf_pump.fit(Xs_train, ys_train)

ys_pred = clf_pump.predict(Xs_test)
print("\n  Classification Report:")
print(classification_report(ys_test, ys_pred, target_names=["OFF", "ON"]))

# Cross-validation
cv_pump = cross_val_score(clf_pump, X_ps, y_ps, cv=5, scoring="f1")
print(f"  CV F1: {cv_pump.mean():.4f} (+/- {cv_pump.std():.4f})")

# Feature importance
print("\n  Feature Importance (Pump State):")
for feat, imp in sorted(zip(FEATURES_PUMP_STATE, clf_pump.feature_importances_), key=lambda x: -x[1]):
    print(f"    {feat:20s}: {imp}")

# ============================================================
# 5. TRAIN PUMP DURATION MODEL (LightGBM Regressor)
# ============================================================
print("\n[5] Training Pump Duration (LightGBM Regressor)...")

FEATURES_PUMP_DURATION = [
    "temperature", "humidity", "ammonia",
    "temp_avg_1h", "humid_avg_1h", "humid_delta_1h",
    "hour_of_day", "comfort_index"
]

# Hanya train pada data yang pump ON (durasi > 0)
df_on = df[df["pump_state_label"] == 1].copy()
print(f"  Training samples (pump ON only): {len(df_on)}")

if len(df_on) < 50:
    print("  [WARN] Terlalu sedikit data ON, menggunakan semua data")
    df_on = df.copy()

X_pd = df_on[FEATURES_PUMP_DURATION].copy()
y_pd = df_on["pump_duration_label"].copy()

Xd_train, Xd_test, yd_train, yd_test = train_test_split(
    X_pd, y_pd, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

reg_pump = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=200,
    max_depth=8,
    num_leaves=31,
    min_child_samples=30,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)

reg_pump.fit(Xd_train, yd_train)

yd_pred = reg_pump.predict(Xd_test)
mae = mean_absolute_error(yd_test, yd_pred)
r2 = r2_score(yd_test, yd_pred)
print(f"\n  MAE : {mae:.2f} detik")
print(f"  R²  : {r2:.4f}")

# Feature importance
print("\n  Feature Importance (Pump Duration):")
for feat, imp in sorted(zip(FEATURES_PUMP_DURATION, reg_pump.feature_importances_), key=lambda x: -x[1]):
    print(f"    {feat:20s}: {imp}")

# ============================================================
# 6. SIMPAN ARTEFAK
# ============================================================
print("\n[6] Saving artifacts...")

path_pump_state    = os.path.join(MODELS_DIR, "model_pump_state_v2.pkl")
path_pump_duration = os.path.join(MODELS_DIR, "model_pump_duration_v2.pkl")

joblib.dump(clf_pump, path_pump_state, compress=3)
joblib.dump(reg_pump, path_pump_duration, compress=3)

print(f"  ✓ {path_pump_state}")
print(f"  ✓ {path_pump_duration}")

# ============================================================
# 7. QUICK INFERENCE TEST
# ============================================================
print("\n[7] Inference test...")

test_scenarios = [
    {"desc": "Panas + kering (should ON, long spray)",
     "temperature": 32.0, "humidity": 65.0, "ammonia": 8.0,
     "temp_avg_1h": 31.5, "humid_avg_1h": 67.0, "humid_delta_1h": -3.0,
     "hour_of_day": 13, "comfort_index": calculate_comfort_index(32.0, 65.0, 8.0)},
    {"desc": "Kondisi ideal (should OFF)",
     "temperature": 28.0, "humidity": 80.0, "ammonia": 5.0,
     "temp_avg_1h": 28.0, "humid_avg_1h": 79.0, "humid_delta_1h": 0.5,
     "hour_of_day": 10, "comfort_index": calculate_comfort_index(28.0, 80.0, 5.0)},
    {"desc": "NH3 tinggi (should ON)",
     "temperature": 29.0, "humidity": 75.0, "ammonia": 28.0,
     "temp_avg_1h": 28.5, "humid_avg_1h": 76.0, "humid_delta_1h": -1.0,
     "hour_of_day": 15, "comfort_index": calculate_comfort_index(29.0, 75.0, 28.0)},
    {"desc": "Humidity naik (should OFF, hemat listrik)",
     "temperature": 29.5, "humidity": 73.0, "ammonia": 7.0,
     "temp_avg_1h": 29.0, "humid_avg_1h": 71.0, "humid_delta_1h": 3.0,
     "hour_of_day": 8, "comfort_index": calculate_comfort_index(29.5, 73.0, 7.0)},
    {"desc": "Malam hari, sedikit kering (borderline)",
     "temperature": 27.0, "humidity": 70.0, "ammonia": 6.0,
     "temp_avg_1h": 27.5, "humid_avg_1h": 72.0, "humid_delta_1h": -1.5,
     "hour_of_day": 22, "comfort_index": calculate_comfort_index(27.0, 70.0, 6.0)},
]

for sc in test_scenarios:
    desc = sc.pop("desc")
    X_state = pd.DataFrame([{k: sc[k] for k in FEATURES_PUMP_STATE}], columns=FEATURES_PUMP_STATE)
    X_dur   = pd.DataFrame([{k: sc[k] for k in FEATURES_PUMP_DURATION}], columns=FEATURES_PUMP_DURATION)

    state_pred = int(clf_pump.predict(X_state)[0])
    state_prob = clf_pump.predict_proba(X_state)[0]
    dur_pred   = float(reg_pump.predict(X_dur)[0])
    dur_pred   = max(0.0, min(MAX_SPRAY_DURATION, dur_pred))

    action = "ON" if state_pred == 1 else "OFF"
    print(f"\n  {desc}")
    print(f"    temp={sc['temperature']}°C, rh={sc['humidity']}%, nh3={sc['ammonia']}ppm")
    print(f"    → Action: {action} (confidence: OFF={state_prob[0]:.3f}, ON={state_prob[1]:.3f})")
    if state_pred == 1:
        print(f"    → Duration: {dur_pred:.1f} detik")

# ============================================================
# 8. COST ANALYSIS
# ============================================================
print("\n\n[8] Cost analysis (estimasi)...")

total_rows = len(df)
spray_on_count = df["pump_state_label"].sum()
spray_off_count = total_rows - spray_on_count
avg_duration = df.loc[df["pump_state_label"] == 1, "pump_duration_label"].mean() if spray_on_count > 0 else 0

# Bandingkan dengan pure rule (always spray jika humidity < 80)
naive_on = (df["humidity"] < 80).sum()
naive_ratio = naive_on / total_rows * 100
smart_ratio = spray_on_count / total_rows * 100
saving = naive_ratio - smart_ratio

print(f"  Naive rule (humidity < 80)  : {naive_on}/{total_rows} ({naive_ratio:.1f}%) spray events")
print(f"  Smart ML labels             : {spray_on_count}/{total_rows} ({smart_ratio:.1f}%) spray events")
print(f"  Estimated savings           : {saving:.1f}% fewer spray events")
print(f"  Average spray duration      : {avg_duration:.1f} detik (saat ON)")

# ============================================================
print("\n" + "=" * 60)
print("PUMP AUTOMATION TRAINING COMPLETE (LightGBM)")
print("=" * 60)
print(f"\nOutput files:")
print(f"  - model_pump_state_v2.pkl    (LightGBM Classifier: ON/OFF)")
print(f"  - model_pump_duration_v2.pkl (LightGBM Regressor: durasi)")
print(f"\nPump State  CV F1: {cv_pump.mean():.4f}")
print(f"Pump Duration MAE: {mae:.2f} detik, R²: {r2:.4f}")
print(f"Est. savings vs naive rule: {saving:.1f}% fewer sprays")
print("=" * 60)

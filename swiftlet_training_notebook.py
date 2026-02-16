# ============================================================
# Swiftlet AI Engine - Training Notebook (Spec v2 - FIXED)
# ------------------------------------------------------------
# Output model HARUS sesuai spec:
#  - model_anomaly_detection_v2.pkl
#  - scaler_anomaly_v2.pkl
#  - model_grade_panen_v2.pkl
#  - label_encoder_grade_v2.pkl
#  - model_pump_state_v2.pkl
#  - model_pump_duration_v2.pkl
#
# FIXED:
#  - Spike detection menggunakan z-score dan percentage change
#  - Delta features menggunakan percentage-based (bukan absolute)
#  - Threshold spike sesuai API doc (z > 3 atau pct > 30%)
#
# Dataset: sensor_data_average.csv
# Kolom minimal: recorded_at / timestamp, temperature_c, humidity_rh, nh3_ppm
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.metrics import classification_report, mean_absolute_error
import joblib

# -----------------------------
# 0. Path data & artefak
# -----------------------------
# Untuk training lokal
DATA_PATH = "data/ml_training_dataset.csv"
MODELS_DIR = "models"

# Untuk Kaggle (uncomment jika di Kaggle):
# DATA_PATH = "/kaggle/input/sensor-syntesis/ml_training_dataset.csv"
# MODELS_DIR = "/kaggle/working"

os.makedirs(MODELS_DIR, exist_ok=True)

print("Data path :", DATA_PATH)
print("Models dir:", MODELS_DIR)

# -----------------------------
# KONSTANTA SPIKE DETECTION (Sesuai API Doc)
# -----------------------------
SPIKE_Z_THRESHOLD = 3.0        # Z-score > 3 = spike
SPIKE_PCT_THRESHOLD = 30.0     # Percentage change > 30% = spike
HISTORY_WINDOW = 10            # Window untuk rolling stats
MIN_DELTA_THRESHOLD = 0.5      # Minimum absolute change untuk dianggap signifikan

# -----------------------------
# KONSTANTA ANTI-OVERFITTING
# -----------------------------
RANDOM_STATE = 42
NOISE_LEVEL = 0.1              # 10% noise untuk label sintetis
MAX_DEPTH = 10                 # Limit kedalaman tree
MIN_SAMPLES_LEAF = 50          # Minimum samples per leaf
MIN_SAMPLES_SPLIT = 100        # Minimum samples untuk split
N_ESTIMATORS = 100             # Kurangi jumlah trees

# -----------------------------
# 1. Load & pre-clean data
# -----------------------------
df = pd.read_csv(DATA_PATH)
print("Shape raw :", df.shape)
print("Columns   :", df.columns.tolist())

# Normalisasi nama kolom
df.columns = [c.strip().lower() for c in df.columns]
print("\nAfter lower():", df.columns.tolist())

# Pastikan kolom sensor ada
required = ["temperature_c", "humidity_rh", "nh3_ppm"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise RuntimeError(f"Kolom wajib hilang: {missing}")

# Konversi ke numerik
for col in required:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop baris yang semua sensor kosong
df = df.dropna(subset=required, how="all")
print("Shape after drop NA (all sensors null):", df.shape)

# --- Handle waktu (recorded_at / timestamp) ---
if "recorded_at" in df.columns:
    dt = pd.to_datetime(df["recorded_at"], errors="coerce")
elif "timestamp" in df.columns:
    dt = pd.to_datetime(df["timestamp"], errors="coerce")
else:
    dt = pd.Series(pd.date_range("2025-01-01", periods=len(df), freq="H"))

df["dt"] = dt
df = df.sort_values("dt")

# -----------------------------
# 2. Derived Features (sesuai spec)
# -----------------------------

# 2.1 Base features dengan nama generic (sesuai spec)
df["temperature"] = df["temperature_c"]
df["humidity"]    = df["humidity_rh"]
df["ammonia"]     = df["nh3_ppm"]

# 2.2 Temporal features
df["hour_of_day"] = df["dt"].dt.hour
df["is_daytime"]  = ((df["hour_of_day"] >= 6) & (df["hour_of_day"] < 18)).astype(int)

# 2.3 Rolling averages (1 jam)
ROLL_WINDOW = HISTORY_WINDOW

df["temp_avg_1h"]  = df["temperature"].rolling(window=ROLL_WINDOW, min_periods=1).mean()
df["humid_avg_1h"] = df["humidity"].rolling(window=ROLL_WINDOW, min_periods=1).mean()
df["nh3_avg_1h"]   = df["ammonia"].rolling(window=ROLL_WINDOW, min_periods=1).mean()

# -----------------------------
# 2.4 Delta Features (FIXED - Percentage-based)
# -----------------------------
def safe_pct_change(series):
    """
    Calculate percentage change dengan handling untuk:
    - Division by zero
    - Extreme outliers
    - Small absolute changes yang tidak signifikan
    """
    # Hitung absolute diff dulu
    abs_diff = series.diff()
    
    # Hitung percentage change
    pct = series.pct_change() * 100  # dalam persen
    
    # Jika absolute change < threshold, anggap 0 (bukan spike)
    # Ini mencegah deteksi spike pada perubahan kecil seperti 0.1
    pct = pct.where(abs_diff.abs() >= MIN_DELTA_THRESHOLD, 0.0)
    
    # Clip extreme values untuk menghindari outlier
    pct = pct.clip(lower=-100, upper=100)
    
    return pct.fillna(0.0)

# Percentage-based delta (lebih robust terhadap small absolute changes)
df["temp_delta_1h"]  = safe_pct_change(df["temperature"])
df["humid_delta_1h"] = safe_pct_change(df["humidity"])
df["nh3_delta_1h"]   = safe_pct_change(df["ammonia"])

print("\nDelta features statistics (percentage-based):")
print(f"  temp_delta_1h  - mean: {df['temp_delta_1h'].mean():.2f}%, std: {df['temp_delta_1h'].std():.2f}%")
print(f"  humid_delta_1h - mean: {df['humid_delta_1h'].mean():.2f}%, std: {df['humid_delta_1h'].std():.2f}%")
print(f"  nh3_delta_1h   - mean: {df['nh3_delta_1h'].mean():.2f}%, std: {df['nh3_delta_1h'].std():.2f}%")

# -----------------------------
# 2.5 Spike Detection Features (Z-score based, sesuai API doc)
# -----------------------------
def calculate_z_score(series, window=HISTORY_WINDOW):
    """
    Hitung z-score untuk setiap data point.
    Sesuai doc: z_score = |value - mean| / std
    """
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    rolling_std = series.rolling(window=window, min_periods=1).std()
    
    # Handle division by zero - jika std = 0, gunakan nilai kecil
    rolling_std = rolling_std.replace(0, np.nan).fillna(1e-6)
    
    z_score = (series - rolling_mean) / rolling_std
    return z_score

def detect_spike(series, window=HISTORY_WINDOW):
    """
    Deteksi spike berdasarkan kriteria API doc:
    - Z-score > 3 ATAU
    - Percentage change > 30%
    
    Returns: Series of 0/1 (is_spike)
    """
    # Calculate z-score
    z_score = calculate_z_score(series, window)
    
    # Calculate percentage change
    pct_change = safe_pct_change(series)
    
    # Spike jika z-score > 3 ATAU pct_change > 30%
    is_spike = ((z_score.abs() > SPIKE_Z_THRESHOLD) | 
                (pct_change.abs() > SPIKE_PCT_THRESHOLD)).astype(int)
    
    return is_spike

# Buat fitur spike untuk setiap sensor
df["temp_spike"]  = detect_spike(df["temperature"])
df["humid_spike"] = detect_spike(df["humidity"])
df["nh3_spike"]   = detect_spike(df["ammonia"])

# Gabungkan jadi satu fitur: ada spike di salah satu sensor
df["any_spike"] = ((df["temp_spike"] == 1) | 
                   (df["humid_spike"] == 1) | 
                   (df["nh3_spike"] == 1)).astype(int)

print("\nSpike detection statistics (z>3 OR pct>30%):")
print(f"  temp_spike  : {df['temp_spike'].sum()} / {len(df)} ({df['temp_spike'].mean()*100:.2f}%)")
print(f"  humid_spike : {df['humid_spike'].sum()} / {len(df)} ({df['humid_spike'].mean()*100:.2f}%)")
print(f"  nh3_spike   : {df['nh3_spike'].sum()} / {len(df)} ({df['nh3_spike'].mean()*100:.2f}%)")
print(f"  any_spike   : {df['any_spike'].sum()} / {len(df)} ({df['any_spike'].mean()*100:.2f}%)")

# -----------------------------
# 2.6 Comfort Index
# -----------------------------
def calculate_comfort_index(temperature, humidity, ammonia):
    temp_optimal = 28.0
    humid_optimal = 80.0
    nh3_max = 20.0

    temp_score  = 1 - abs(temperature - temp_optimal) / 15.0
    humid_score = 1 - abs(humidity - humid_optimal) / 35.0
    nh3_score   = 1 - (ammonia / nh3_max)

    temp_score  = max(0.0, min(1.0, temp_score))
    humid_score = max(0.0, min(1.0, humid_score))
    nh3_score   = max(0.0, min(1.0, nh3_score))

    comfort_index = (temp_score * 0.35 + humid_score * 0.35 + nh3_score * 0.30) * 100.0
    return round(comfort_index, 2)

df["comfort_index"] = df[["temperature", "humidity", "ammonia"]].apply(
    lambda row: calculate_comfort_index(row["temperature"], row["humidity"], row["ammonia"]),
    axis=1
)

# Buang baris yang derived-nya masih NaN (kalau ada)
derived_cols = [
    "temperature", "humidity", "ammonia",
    "hour_of_day", "is_daytime",
    "temp_avg_1h", "humid_avg_1h", "nh3_avg_1h",
    "temp_delta_1h", "humid_delta_1h", "nh3_delta_1h",
    "comfort_index"
]
df = df.dropna(subset=derived_cols)
df = df.reset_index(drop=True)

print("\nShape after derived features:", df.shape)

# -----------------------------
# 3. Label Grade (bagus / sedang / buruk) dengan NOISE untuk anti-overfitting
# -----------------------------
np.random.seed(RANDOM_STATE)

def label_grade(row):
    ci   = row["comfort_index"]
    temp = row["temperature"]
    rh   = row["humidity"]
    nh3  = row["ammonia"]

    # Hard rule kalau kondisi benar-benar buruk
    if (nh3 is not None and nh3 > 35) or \
       (temp is not None and (temp < 20 or temp > 35)) or \
       (rh is not None and (rh < 60 or rh > 95)):
        return "buruk"

    # Dari comfort index (sesuai doc: > 70 = bagus)
    if ci > 70:
        return "bagus"
    elif ci >= 50:
        return "sedang"
    else:
        return "buruk"

df["grade"] = df.apply(label_grade, axis=1)

# Tambahkan noise pada label untuk mencegah overfitting
# Secara random flip beberapa label (NOISE_LEVEL = 10%)
grade_mapping = {"bagus": 0, "sedang": 1, "buruk": 2}
grade_reverse = {0: "bagus", 1: "sedang", 2: "buruk"}

def add_label_noise(grade, noise_prob=NOISE_LEVEL):
    """Randomly flip label with probability noise_prob"""
    if np.random.random() < noise_prob:
        current = grade_mapping[grade]
        # Ubah ke kelas tetangga (lebih realistis)
        if current == 0:  # bagus -> sedang
            return "sedang"
        elif current == 2:  # buruk -> sedang
            return "sedang"
        else:  # sedang -> random bagus/buruk
            return np.random.choice(["bagus", "buruk"])
    return grade

df["grade"] = df["grade"].apply(add_label_noise)

print("\nDistribusi grade (dengan noise untuk anti-overfit):")
print(df["grade"].value_counts())

if df["grade"].nunique() < 2:
    raise RuntimeError("Kelas grade kurang dari 2, susah dilatih. Cek threshold / data.")

# -----------------------------
# 4. DEFINISI FEATURE SET SESUAI SPEC
# -----------------------------
FEATURES_ANOMALY = [
    "temperature", "humidity", "ammonia",
    "hour_of_day", "temp_delta_1h", "humid_delta_1h",
    "nh3_delta_1h", "comfort_index"
]

FEATURES_GRADE = [
    "temperature", "humidity", "ammonia",
    "temp_avg_1h", "humid_avg_1h", "nh3_avg_1h",
    "comfort_index", "is_daytime"
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
# 5. TRAIN GRADE PREDICTION MODEL (model_grade_panen_v2.pkl)
# ============================================================

X_grade = df[FEATURES_GRADE].copy()
y_grade = df["grade"].copy()

# LabelEncoder untuk kelas: bagus / sedang / buruk
le_grade = LabelEncoder()
y_grade_enc = le_grade.fit_transform(y_grade)

Xg_train, Xg_test, yg_train, yg_test = train_test_split(
    X_grade, y_grade_enc, test_size=0.2, random_state=42, stratify=y_grade_enc
)

clf_grade = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    min_samples_split=MIN_SAMPLES_SPLIT,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
clf_grade.fit(Xg_train, yg_train)

yg_pred = clf_grade.predict(Xg_test)
print("\n[GRADE] Classification report (encoded):")
print(classification_report(yg_test, yg_pred, target_names=le_grade.classes_))

# Simpan model & encoder sesuai spec
path_grade_model  = os.path.join(MODELS_DIR, "model_grade_panen_v2.pkl")
path_grade_encoder = os.path.join(MODELS_DIR, "label_encoder_grade_v2.pkl")

joblib.dump(clf_grade, path_grade_model, compress=3)
joblib.dump(le_grade, path_grade_encoder, compress=3)

print("\n[GRADE] Saved:")
print(" -", path_grade_model)
print(" -", path_grade_encoder)

# ============================================================
# 6. TRAIN ANOMALY DETECTION MODEL (model_anomaly_detection_v2.pkl)
# ============================================================

X_anom = df[FEATURES_ANOMALY].copy()

# StandardScaler khusus anomaly (sesuai spec: scaler_anomaly_v2.pkl)
scaler_anom = StandardScaler()
X_anom_scaled = scaler_anom.fit_transform(X_anom)

iso = IsolationForest(
    n_estimators=N_ESTIMATORS,
    contamination=0.15,  # 15% dianggap anomali
    max_samples=0.8,     # Subsample untuk anti-overfitting
    random_state=RANDOM_STATE,
    n_jobs=-1
)
iso.fit(X_anom_scaled)

preds = iso.predict(X_anom_scaled)  # 1 normal, -1 anomaly
vals, counts = np.unique(preds, return_counts=True)
dist = {int(v): int(c) for v, c in zip(vals, counts)}
print("\n[ANOM] Distribusi prediksi (1=normal, -1=anomali):", dist)

path_anom_model  = os.path.join(MODELS_DIR, "model_anomaly_detection_v2.pkl")
path_anom_scaler = os.path.join(MODELS_DIR, "scaler_anomaly_v2.pkl")

joblib.dump(iso, path_anom_model, compress=3)
joblib.dump(scaler_anom, path_anom_scaler, compress=3)

print("[ANOM] Saved:")
print(" -", path_anom_model)
print(" -", path_anom_scaler)

# ============================================================
# 7. TRAIN PUMP STATE MODEL (model_pump_state_v2.pkl)
# ============================================================

def synth_pump_state(row):
    """
    Label ON/OFF sintetis sesuai logika di dokumen (FIXED):
    - ON jika humidity < 75%
    - ATAU (temperature > 30 dan humidity < 80%)
    - ATAU humid_delta_1h turun cepat (> 5% per jam, PERCENTAGE bukan absolute!)
    """
    h = row["humidity"]
    t = row["temperature"]
    dh = row["humid_delta_1h"]  # Sekarang dalam percentage

    if h < 75:
        return 1
    if (t > 30) and (h < 80):
        return 1
    
    # FIXED: Check percentage drop > 5% (bukan absolute 5 unit)
    # dh adalah percentage change, jadi -5 berarti turun 5%
    if dh < -5:
        return 1
    
    return 0

df["pump_state_label"] = df.apply(synth_pump_state, axis=1)

# Tambahkan noise pada label pump state
def add_binary_noise(label, noise_prob=NOISE_LEVEL):
    if np.random.random() < noise_prob:
        return 1 - label  # Flip 0 -> 1 atau 1 -> 0
    return label

df["pump_state_label"] = df["pump_state_label"].apply(add_binary_noise)
print("\n[PUMP STATE] Distribusi label ON/OFF:")
print(df["pump_state_label"].value_counts())

X_pump_state = df[FEATURES_PUMP_STATE].copy()
y_pump_state = df["pump_state_label"].copy()

# Handle case jika semua label sama (tidak bisa stratify)
if y_pump_state.nunique() < 2:
    print("\n[WARNING] Pump state hanya punya 1 kelas, skip stratify")
    Xs_train, Xs_test, ys_train, ys_test = train_test_split(
        X_pump_state, y_pump_state, test_size=0.2, random_state=42
    )
else:
    Xs_train, Xs_test, ys_train, ys_test = train_test_split(
        X_pump_state, y_pump_state, test_size=0.2, random_state=42, stratify=y_pump_state
    )

clf_pump_state = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    min_samples_split=MIN_SAMPLES_SPLIT,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
clf_pump_state.fit(Xs_train, ys_train)

ys_pred = clf_pump_state.predict(Xs_test)
print("\n[PUMP STATE] Classification report:")
print(classification_report(ys_test, ys_pred))

path_pump_state = os.path.join(MODELS_DIR, "model_pump_state_v2.pkl")
joblib.dump(clf_pump_state, path_pump_state, compress=3)
print("[PUMP STATE] Saved:", path_pump_state)

# ============================================================
# 8. TRAIN PUMP DURATION MODEL (model_pump_duration_v2.pkl)
# ============================================================

def synth_pump_duration(row):
    """
    Durasi sintetis pakai formula fallback dari dokumen:
      humid_gap = max(0, 80 - humidity)
      base_duration = humid_gap * 0.4
      temp_factor = 1 + max(0, temperature - 28) * 0.1
      time_factor = 1.2 if is_daytime else 1.0
      trend_factor = 1 + max(0, -humid_delta_1h) * 0.1  # FIXED: humid_delta now in %
      duration = min(base_duration * temp_factor * time_factor * trend_factor, 30)
    """
    humidity       = row["humidity"]
    temperature    = row["temperature"]
    is_daytime     = row["is_daytime"]
    humid_delta_1h = row["humid_delta_1h"]  # Sekarang dalam percentage

    humid_gap     = max(0.0, 80.0 - humidity)
    base_duration = humid_gap * 0.4

    temp_factor  = 1.0 + max(0.0, temperature - 28.0) * 0.1
    time_factor  = 1.2 if is_daytime == 1 else 1.0
    
    # FIXED: humid_delta_1h sekarang dalam percentage
    # -10% drop -> trend_factor = 1 + 10 * 0.01 = 1.1
    trend_factor = 1.0 + max(0.0, -humid_delta_1h) * 0.01  # Scaled untuk percentage

    duration = base_duration * temp_factor * time_factor * trend_factor
    duration = min(duration, 30.0)
    return float(duration)

df["pump_duration_label"] = df.apply(synth_pump_duration, axis=1)

# Tambahkan noise pada durasi untuk anti-overfitting
# Noise = +/- 10% dari nilai asli
def add_regression_noise(value, noise_pct=NOISE_LEVEL):
    noise = value * noise_pct * np.random.uniform(-1, 1)
    return max(0.0, min(30.0, value + noise))  # Clamp 0-30

df["pump_duration_label"] = df["pump_duration_label"].apply(add_regression_noise)

print("\n[PUMP DURATION] Contoh nilai durasi sintetis (dengan noise):")
print(df["pump_duration_label"].describe())

X_pump_dur = df[FEATURES_PUMP_DURATION].copy()
y_pump_dur = df["pump_duration_label"].copy()

Xd_train, Xd_test, yd_train, yd_test = train_test_split(
    X_pump_dur, y_pump_dur, test_size=0.2, random_state=42
)

reg_pump_dur = RandomForestRegressor(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    min_samples_split=MIN_SAMPLES_SPLIT,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
reg_pump_dur.fit(Xd_train, yd_train)

yd_pred = reg_pump_dur.predict(Xd_test)
mae = mean_absolute_error(yd_test, yd_pred)
print(f"\n[PUMP DURATION] MAE: {mae:.3f} menit")

path_pump_duration = os.path.join(MODELS_DIR, "model_pump_duration_v2.pkl")
joblib.dump(reg_pump_dur, path_pump_duration, compress=3)
print("[PUMP DURATION] Saved:", path_pump_duration)

# ============================================================
# 9. RINGKASAN OUTPUT
# ============================================================
print("\n" + "="*60)
print("SEMUA MODEL SELESAI DITRAIN (SESUAI SPEC v2 - FIXED)")
print("="*60)
print("\nFile yang dihasilkan di /kaggle/working:")
print(" - model_anomaly_detection_v2.pkl")
print(" - scaler_anomaly_v2.pkl")
print(" - model_grade_panen_v2.pkl")
print(" - label_encoder_grade_v2.pkl")
print(" - model_pump_state_v2.pkl")
print(" - model_pump_duration_v2.pkl")

print("\n" + "-"*60)
print("PERBAIKAN YANG DILAKUKAN:")
print("-"*60)
print("1. Delta features sekarang percentage-based (bukan absolute)")
print("2. Spike detection menggunakan z-score > 3 OR pct > 30%")
print("3. Minimum threshold untuk ignore small changes (< 0.5 unit)")
print("4. Grade threshold strict sesuai doc (> 70, bukan >= 70)")
print("5. Pump duration trend_factor disesuaikan untuk percentage input")
print("-"*60)

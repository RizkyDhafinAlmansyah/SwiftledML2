# ============================================================
# Swiftlet AI Engine - Grade Prediction Training (LightGBM)
# ------------------------------------------------------------
# Output:
#  - model_grade_panen_v2.pkl   (LightGBM Classifier)
#  - label_encoder_grade_v2.pkl (LabelEncoder: bagus/sedang/buruk)
#
# Algoritma: LightGBM (optimal untuk data tabular sensor-based)
# Features : temperature, humidity, ammonia, temp_avg_1h,
#            humid_avg_1h, nh3_avg_1h, comfort_index, is_daytime
#
# Dataset  : ml_training_dataset.csv
# Kolom    : recorded_at/timestamp, temperature_c, humidity_rh, nh3_ppm
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import lightgbm as lgb
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
print("SWIFTLET AI - GRADE PREDICTION TRAINING (LightGBM)")
print("=" * 60)
print(f"Data path : {DATA_PATH}")
print(f"Models dir: {MODELS_DIR}")

# -----------------------------
# KONSTANTA
# -----------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
NOISE_LEVEL = 0.10          # 10% label noise untuk anti-overfitting
HISTORY_WINDOW = 10
MIN_DELTA_THRESHOLD = 0.5

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
df["is_daytime"]  = ((df["hour_of_day"] >= 6) & (df["hour_of_day"] < 18)).astype(int)

# 2.3 Rolling averages (1 jam)
ROLL_WINDOW = HISTORY_WINDOW
df["temp_avg_1h"]  = df["temperature"].rolling(window=ROLL_WINDOW, min_periods=1).mean()
df["humid_avg_1h"] = df["humidity"].rolling(window=ROLL_WINDOW, min_periods=1).mean()
df["nh3_avg_1h"]   = df["ammonia"].rolling(window=ROLL_WINDOW, min_periods=1).mean()

# 2.4 Comfort Index (sesuai spec ML)
def calculate_comfort_index(temperature, humidity, ammonia):
    temp_optimal = 28.0
    humid_optimal = 80.0
    nh3_max = 20.0

    temp_score  = max(0.0, min(1.0, 1 - abs(temperature - temp_optimal) / 15.0))
    humid_score = max(0.0, min(1.0, 1 - abs(humidity - humid_optimal) / 35.0))
    nh3_score   = max(0.0, min(1.0, 1 - (ammonia / nh3_max)))

    return round((temp_score * 0.35 + humid_score * 0.35 + nh3_score * 0.30) * 100.0, 2)

df["comfort_index"] = df[["temperature", "humidity", "ammonia"]].apply(
    lambda row: calculate_comfort_index(row["temperature"], row["humidity"], row["ammonia"]),
    axis=1
)

# 2.5 Optuna v2 features
df["day_of_week"] = df["dt"].dt.dayofweek.fillna(0).astype(int)
df["temp_humid_interaction"] = (df["temperature"] * df["humidity"]) / 100.0
df["time_norm"] = df["hour_of_day"] / 24.0

# Drop NaN
derived_cols = [
    "temperature", "humidity", "ammonia",
    "temp_avg_1h", "humid_avg_1h", "nh3_avg_1h",
    "comfort_index", "is_daytime",
    "day_of_week", "temp_humid_interaction", "time_norm",
]
df = df.dropna(subset=derived_cols).reset_index(drop=True)
print(f"  Shape after features: {df.shape}")

# ============================================================
# 3. LABEL GRADE (bagus / sedang / buruk) + NOISE
# ============================================================
print("\n[3] Labeling grade...")

def label_grade(row):
    """
    Revised labeling berdasarkan domain knowledge:
    - Suhu < 30°C + RH 75-85% + NH3 < 10 → bagus
    - Suhu > 33°C → buruk
    - Suhu > 32°C + RH < 72% → buruk
    - Extreme values → buruk
    - Sisanya → sedang
    """
    temp = row["temperature"]
    rh   = row["humidity"]
    nh3  = row["ammonia"]

    # Extreme conditions → buruk
    if nh3 > 25 or temp > 35 or temp < 20 or rh < 55 or rh > 95:
        return "buruk"

    # BAGUS: suhu sejuk + humidity ideal + NH3 rendah
    if temp < 30 and 75 <= rh <= 85 and nh3 < 10:
        return "bagus"

    # BURUK: suhu panas + humidity rendah
    if temp > 32 and rh < 72:
        return "buruk"

    # BURUK: suhu sangat panas
    if temp > 33:
        return "buruk"

    # SEDANG: suhu OK tapi humidity kurang ideal
    if temp < 30 and (rh < 75 or rh > 85):
        return "sedang"

    # SEDANG: suhu agak panas tapi humidity masih OK
    if 30 <= temp <= 32 and rh >= 75:
        return "sedang"

    # SEDANG/BURUK: suhu agak panas + humidity kurang
    if 30 <= temp <= 32 and rh < 75:
        return "buruk" if nh3 > 10 else "sedang"

    return "sedang"

df["grade"] = df.apply(label_grade, axis=1)

# Tambahkan noise pada label untuk anti-overfitting
np.random.seed(RANDOM_STATE)
grade_mapping = {"bagus": 0, "sedang": 1, "buruk": 2}
grade_reverse = {0: "bagus", 1: "sedang", 2: "buruk"}

def add_label_noise(grade, noise_prob=NOISE_LEVEL):
    # JANGAN noise kelas 'buruk' — datanya sudah sedikit
    if grade == "buruk":
        return grade
    if np.random.random() < noise_prob:
        current = grade_mapping[grade]
        other = [v for v in grade_mapping.values() if v != current]
        new_label = np.random.choice(other)
        return grade_reverse[new_label]
    return grade

df["grade"] = df["grade"].apply(add_label_noise)

print("  Distribusi grade (dengan noise):")
print(df["grade"].value_counts().to_string().replace("\n", "\n  "))

if df["grade"].nunique() < 2:
    raise RuntimeError("Kelas grade < 2, cek threshold / data.")

# ============================================================
# 4. TRAIN LIGHTGBM GRADE MODEL
# ============================================================
print("\n[4] Training LightGBM...")

FEATURES_GRADE = [
    "temperature", "humidity", "ammonia",
    "temp_avg_1h", "humid_avg_1h", "nh3_avg_1h",
    "comfort_index", "is_daytime",
    # Optuna v2 features
    "day_of_week", "temp_humid_interaction", "time_norm",
]

X_grade = df[FEATURES_GRADE].copy()
y_grade = df["grade"].copy()

# 3.5 Augmentasi: tambah data sintetis untuk edge cases
# Tambah "buruk" extreme + "sedang" boundary supaya model tau bedanya
print("\n  Augmenting edge case samples...")

np.random.seed(RANDOM_STATE)

def make_sample(t, h, n, daytime=None):
    if daytime is None:
        daytime = np.random.choice([0, 1])
    hour = np.random.randint(6, 18) if daytime else np.random.randint(0, 6)
    return {
        "temperature": round(t, 1),
        "humidity": round(h, 1),
        "ammonia": round(n, 1),
        "temp_avg_1h": round(t + np.random.uniform(-1, 1), 1),
        "humid_avg_1h": round(h + np.random.uniform(-2, 2), 1),
        "nh3_avg_1h": round(n + np.random.uniform(-2, 2), 1),
        "comfort_index": calculate_comfort_index(t, h, n),
        "is_daytime": daytime,
        "day_of_week": np.random.randint(0, 7),
        "temp_humid_interaction": round(t * h / 100.0, 2),
        "time_norm": round(hour / 24.0, 3),
    }

# --- BURUK: hanya kondisi yang JELAS buruk (sesuai rule label_grade) ---
N_BURUK = 200
buruk_samples = []
for _ in range(N_BURUK):
    case = np.random.choice(["high_temp", "very_low_humid", "high_nh3", "combo"])
    if case == "high_temp":
        t, h, n = np.random.uniform(36, 42), np.random.uniform(55, 70), np.random.uniform(5, 15)
    elif case == "very_low_humid":
        t, h, n = np.random.uniform(28, 33), np.random.uniform(35, 58), np.random.uniform(5, 12)
    elif case == "high_nh3":
        t, h, n = np.random.uniform(27, 32), np.random.uniform(65, 80), np.random.uniform(36, 55)
    else:
        t, h, n = np.random.uniform(36, 42), np.random.uniform(40, 58), np.random.uniform(25, 45)
    buruk_samples.append(make_sample(t, h, n))

# --- SEDANG: kondisi boundary (ci 50-70, not extreme) ---
N_SEDANG = 150
sedang_samples = []
for _ in range(N_SEDANG):
    # Temperatur agak di bawah/atas optimal tapi masih OK
    t = np.random.uniform(24, 32)
    # Humidity di zona sedang (60-75 atau 85-90)
    h = np.random.choice([np.random.uniform(62, 75), np.random.uniform(85, 93)])
    # NH3 moderate
    n = np.random.uniform(8, 20)
    ci = calculate_comfort_index(t, h, n)
    # Pastikan memang masuk zona sedang (ci 50-70)
    if 50 <= ci <= 70:
        sedang_samples.append(make_sample(t, h, n))

df_buruk = pd.DataFrame(buruk_samples, columns=FEATURES_GRADE)
df_sedang = pd.DataFrame(sedang_samples, columns=FEATURES_GRADE)

X_grade = pd.concat([X_grade, df_buruk, df_sedang], ignore_index=True)
y_grade = pd.concat([
    y_grade,
    pd.Series(["buruk"] * len(df_buruk)),
    pd.Series(["sedang"] * len(df_sedang)),
], ignore_index=True)

print(f"  Added {len(df_buruk)} synthetic 'buruk' + {len(df_sedang)} synthetic 'sedang'")
print(f"  New distribution:")
print(y_grade.value_counts().to_string().replace("\n", "\n  "))

le_grade = LabelEncoder()
y_grade_enc = le_grade.fit_transform(y_grade)
n_classes = len(le_grade.classes_)

print(f"\n  Classes : {list(le_grade.classes_)}")
print(f"  Features: {FEATURES_GRADE}")
print(f"  Samples : {len(X_grade)}")

# Train/test split
Xg_train, Xg_test, yg_train, yg_test = train_test_split(
    X_grade, y_grade_enc,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_grade_enc
)

from collections import Counter
print(f"  Train class distribution: {dict(Counter(yg_train))}")

clf_grade = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=n_classes,
    n_estimators=300,
    max_depth=8,
    num_leaves=31,
    min_child_samples=50,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)

clf_grade.fit(Xg_train, yg_train)

# ============================================================
# 5. EVALUASI
# ============================================================
print("\n[5] Evaluasi model...")

# Test set evaluation
yg_pred = clf_grade.predict(Xg_test)
print("\n  Classification Report:")
print(classification_report(yg_test, yg_pred, target_names=le_grade.classes_))

# Cross-validation
print("  Cross-validation (5-fold)...")
cv_scores = cross_val_score(clf_grade, X_grade, y_grade_enc, cv=5, scoring="f1_macro")
print(f"  CV F1-macro: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Feature importance
print("\n  Feature Importance:")
importances = clf_grade.feature_importances_
for feat, imp in sorted(zip(FEATURES_GRADE, importances), key=lambda x: -x[1]):
    print(f"    {feat:20s}: {imp}")

# Confidence analysis
if hasattr(clf_grade, "predict_proba"):
    proba = clf_grade.predict_proba(Xg_test)
    max_conf = proba.max(axis=1)
    print(f"\n  Confidence stats:")
    print(f"    Mean  : {max_conf.mean():.4f}")
    print(f"    Median: {np.median(max_conf):.4f}")
    print(f"    Min   : {max_conf.min():.4f}")
    print(f"    > 0.7 : {(max_conf > 0.7).sum()}/{len(max_conf)} ({(max_conf > 0.7).mean()*100:.1f}%)")

# ============================================================
# 6. SIMPAN ARTEFAK
# ============================================================
print("\n[6] Saving artifacts...")

path_grade_model  = os.path.join(MODELS_DIR, "model_grade_panen_v2.pkl")
path_grade_encoder = os.path.join(MODELS_DIR, "label_encoder_grade_v2.pkl")

joblib.dump(clf_grade, path_grade_model, compress=3)
joblib.dump(le_grade, path_grade_encoder, compress=3)

print(f"  ✓ {path_grade_model}")
print(f"  ✓ {path_grade_encoder}")

# ============================================================
# 7. QUICK INFERENCE TEST
# ============================================================
print("\n[7] Inference test...")

test_samples = [
    {"temperature": 28.5, "humidity": 78.0, "ammonia": 5.0,
     "temp_avg_1h": 28.3, "humid_avg_1h": 77.5, "nh3_avg_1h": 5.2,
     "comfort_index": calculate_comfort_index(28.5, 78.0, 5.0),
     "is_daytime": 1, "day_of_week": 2,
     "temp_humid_interaction": round(28.5 * 78.0 / 100.0, 2),
     "time_norm": round(14 / 24.0, 3)},
    {"temperature": 33.0, "humidity": 60.0, "ammonia": 25.0,
     "temp_avg_1h": 32.5, "humid_avg_1h": 62.0, "nh3_avg_1h": 23.0,
     "comfort_index": calculate_comfort_index(33.0, 60.0, 25.0),
     "is_daytime": 1, "day_of_week": 2,
     "temp_humid_interaction": round(33.0 * 60.0 / 100.0, 2),
     "time_norm": round(14 / 24.0, 3)},
    {"temperature": 25.0, "humidity": 85.0, "ammonia": 12.0,
     "temp_avg_1h": 25.5, "humid_avg_1h": 84.0, "nh3_avg_1h": 11.5,
     "comfort_index": calculate_comfort_index(25.0, 85.0, 12.0),
     "is_daytime": 0, "day_of_week": 5,
     "temp_humid_interaction": round(25.0 * 85.0 / 100.0, 2),
     "time_norm": round(3 / 24.0, 3)},
]

for i, sample in enumerate(test_samples, 1):
    X_test_row = pd.DataFrame([sample], columns=FEATURES_GRADE)
    pred_enc = int(clf_grade.predict(X_test_row)[0])
    pred_label = le_grade.inverse_transform([pred_enc])[0]
    proba = clf_grade.predict_proba(X_test_row)[0]
    probs_dict = {le_grade.inverse_transform([c])[0]: round(float(p), 4)
                  for c, p in zip(clf_grade.classes_, proba)}
    print(f"  Sample {i}: temp={sample['temperature']}°C, rh={sample['humidity']}%, "
          f"nh3={sample['ammonia']}ppm")
    print(f"    → Grade: {pred_label} | Confidence: {probs_dict}")

# ============================================================
print("\n" + "=" * 60)
print("GRADE TRAINING COMPLETE (LightGBM)")
print("=" * 60)
print(f"\nOutput files:")
print(f"  - model_grade_panen_v2.pkl   (LightGBM Classifier)")
print(f"  - label_encoder_grade_v2.pkl (LabelEncoder)")
print(f"\nCV F1-macro: {cv_scores.mean():.4f}")
print("=" * 60)

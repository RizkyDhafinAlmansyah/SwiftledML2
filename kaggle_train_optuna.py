# ============================================================
# Swiftlet AI Engine - Optuna-Tuned Training Pipeline
# ------------------------------------------------------------
# Unified training script with:
#   1. Advanced Feature Engineering (5 fitur tambahan)
#   2. Optuna Hyperparameter Tuning (50 trials per model)
#
# Output (sama dengan script lama):
#   - model_grade_panen_v2.pkl   (LightGBM Classifier)
#   - label_encoder_grade_v2.pkl (LabelEncoder)
#   - model_pump_state_v2.pkl    (LightGBM Classifier)
#   - model_pump_duration_v2.pkl (LightGBM Regressor)
#
# Dataset: ml_training_dataset.csv
# ============================================================

import os
import warnings
import time as time_module
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    mean_absolute_error, r2_score, f1_score
)
import lightgbm as lgb
import optuna
import joblib

# Suppress Optuna logs (hanya tampilkan summary)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# -----------------------------
# 0. CONFIG
# -----------------------------
# Kaggle — dataset: real + external (GAMS indoor)
DATA_PATH = "/kaggle/input/swiftledml-merged-external/ml_training_with_external.csv"
MODELS_DIR = "/kaggle/working"

# Lokal (uncomment jika di lokal)
# DATA_PATH = "data/ml_training_with_external.csv"
# MODELS_DIR = "ai-engine"

os.makedirs(MODELS_DIR, exist_ok=True)

# Konstanta
RANDOM_STATE = 42
TEST_SIZE = 0.2
NOISE_LEVEL = 0.20
HISTORY_WINDOW = 10
MIN_DELTA_THRESHOLD = 0.5
N_OPTUNA_TRIALS = 50
CV_FOLDS = 3

# Threshold sprayer
TEMP_IDEAL_MIN = 26.0
TEMP_IDEAL_MAX = 30.0
HUMID_IDEAL_MIN = 75.0
HUMID_IDEAL_MAX = 85.0
NH3_WARN = 15.0
NH3_CRITICAL = 25.0
MAX_SPRAY_DURATION = 30.0
MIN_SPRAY_DURATION = 5.0

print("=" * 60)
print("SWIFTLET AI - OPTUNA-TUNED TRAINING PIPELINE")
print("=" * 60)
print(f"Data path     : {DATA_PATH}")
print(f"Models dir    : {MODELS_DIR}")
print(f"Optuna trials : {N_OPTUNA_TRIALS} per model")
print(f"CV folds      : {CV_FOLDS}")

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
# 2. FEATURE ENGINEERING (Original + 5 fitur baru)
# ============================================================
print("\n[2] Engineering features (enhanced)...")

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

# 2.4 Delta features
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

# 2.6 Humidity trend slope
df["humid_trend_slope"] = (
    df["humidity"].rolling(window=ROLL_WINDOW, min_periods=2)
    .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0.0, raw=False)
    .fillna(0.0)
)

# ──────────────────────────────────────────────
# 2.7 NEW FEATURES (5 tambahan)
# ──────────────────────────────────────────────
print("  Adding 5 new features...")

# (A) day_of_week: pola mingguan (0=Senin, 6=Minggu)
df["day_of_week"] = df["dt"].dt.dayofweek.fillna(0).astype(int)

# (B) temp_humidity_interaction: interaksi suhu × kelembaban
df["temp_humid_interaction"] = (df["temperature"] * df["humidity"]) / 100.0

# (C) nh3_rate_of_change: kecepatan perubahan NH3 (absolut per step)
df["nh3_rate_of_change"] = df["ammonia"].diff().fillna(0.0).clip(-20, 20)

# (D) rolling_std_30min: volatilitas suhu (std rolling 5 readings ≈ 30 menit)
df["temp_rolling_std"] = (
    df["temperature"].rolling(window=5, min_periods=1).std().fillna(0.0)
)

# (E) time_since_midnight_norm: waktu ternormalisasi (0.0-1.0)
df["time_norm"] = df["hour_of_day"] / 24.0

print(f"  New features: day_of_week, temp_humid_interaction, nh3_rate_of_change, "
      f"temp_rolling_std, time_norm")

# Drop NaN
all_feature_cols = [
    "temperature", "humidity", "ammonia",
    "temp_avg_1h", "humid_avg_1h", "nh3_avg_1h",
    "temp_delta_1h", "humid_delta_1h", "nh3_delta_1h",
    "comfort_index", "is_daytime", "hour_of_day",
    "humid_trend_slope",
    "day_of_week", "temp_humid_interaction", "nh3_rate_of_change",
    "temp_rolling_std", "time_norm",
]
df = df.dropna(subset=all_feature_cols).reset_index(drop=True)
print(f"  Shape after features: {df.shape}")

# ============================================================
# 3. SYNTHETIC LABELS
# ============================================================
print("\n[3] Generating labels...")

# --- 3A. Grade Label (Revised: stricter temperature rules) ---
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

# Label noise
np.random.seed(RANDOM_STATE)
grade_mapping = {"bagus": 0, "sedang": 1, "buruk": 2}
grade_reverse = {0: "bagus", 1: "sedang", 2: "buruk"}

def add_label_noise(grade, noise_prob=NOISE_LEVEL):
    if grade == "buruk":
        return grade
    if np.random.random() < noise_prob:
        current = grade_mapping[grade]
        other = [v for v in grade_mapping.values() if v != current]
        return grade_reverse[np.random.choice(other)]
    return grade

df["grade"] = df["grade"].apply(add_label_noise)
print(f"  Grade distribution: {dict(df['grade'].value_counts())}")

# --- 3B. Pump State Label ---
def smart_pump_state(row):
    temp  = row["temperature"]
    humid = row["humidity"]
    nh3   = row["ammonia"]
    ci    = row["comfort_index"]
    trend = row["humid_trend_slope"]
    hour  = row["hour_of_day"]

    if nh3 >= NH3_CRITICAL:
        return 1
    if temp >= 33 and humid < 70:
        return 1
    if HUMID_IDEAL_MIN <= humid <= HUMID_IDEAL_MAX and TEMP_IDEAL_MIN <= temp <= TEMP_IDEAL_MAX:
        return 0
    if ci >= 75:
        return 0
    if trend > 0.3 and humid >= 70:
        return 0
    if humid < HUMID_IDEAL_MIN and temp > TEMP_IDEAL_MAX:
        return 1
    if humid < 65:
        return 1
    if nh3 >= NH3_WARN and ci < 60:
        return 1
    if row["is_daytime"] == 1 and humid < 72 and trend < -0.2:
        return 1
    return 0

df["pump_state_label"] = df.apply(smart_pump_state, axis=1)

if df["pump_state_label"].nunique() < 2:
    median_ci = df["comfort_index"].median()
    df["pump_state_label"] = (df["comfort_index"] < median_ci).astype(int)

print(f"  Pump state: OFF={int((df['pump_state_label']==0).sum())}, "
      f"ON={int((df['pump_state_label']==1).sum())}")

# --- 3C. Pump Duration Label ---
def smart_pump_duration(row):
    if row["pump_state_label"] == 0:
        return 0.0
    humid = row["humidity"]
    temp  = row["temperature"]
    nh3   = row["ammonia"]
    trend = row["humid_trend_slope"]

    humid_gap = max(0.0, 80.0 - humid)
    base_duration = humid_gap * 0.5
    temp_factor = 1.0 + max(0.0, temp - TEMP_IDEAL_MAX) * 0.08
    time_factor = 1.15 if row["is_daytime"] == 1 else 1.0
    trend_factor = 1.0 + max(0.0, -trend) * 0.15
    nh3_factor = 1.0 + max(0.0, nh3 - NH3_WARN) * 0.03
    duration = base_duration * temp_factor * time_factor * trend_factor * nh3_factor
    return round(max(MIN_SPRAY_DURATION, min(duration, MAX_SPRAY_DURATION)), 1)

df["pump_duration_label"] = df.apply(smart_pump_duration, axis=1)

on_mask = df["pump_state_label"] == 1
if on_mask.sum() > 0:
    dur_on = df.loc[on_mask, "pump_duration_label"]
    print(f"  Pump duration (ON): mean={dur_on.mean():.1f}s, "
          f"median={dur_on.median():.1f}s, range=[{dur_on.min():.1f}, {dur_on.max():.1f}]")

# ============================================================
# 4. AUGMENT EDGE CASES (for grade)
# ============================================================
print("\n[4] Augmenting edge cases...")

np.random.seed(RANDOM_STATE)

def make_sample(t, h, n, feature_list, daytime=None):
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
        "hour_of_day": hour,
        "day_of_week": np.random.randint(0, 7),
        "temp_humid_interaction": round(t * h / 100.0, 2),
        "nh3_rate_of_change": round(np.random.uniform(-2, 2), 2),
        "temp_rolling_std": round(np.random.uniform(0, 2), 2),
        "time_norm": round(hour / 24.0, 3),
    }

# BAGUS samples (suhu < 30, RH 75-85, NH3 < 10) — rare in real data
bagus_samples = []
for _ in range(200):
    t = np.random.uniform(26, 29.9)
    h = np.random.uniform(75, 85)
    n = np.random.uniform(0.5, 9)
    bagus_samples.append(make_sample(t, h, n, all_feature_cols))

# BURUK samples — extreme conditions
buruk_samples = []
for _ in range(200):
    case = np.random.choice(["high_temp", "very_low_humid", "high_nh3", "combo", "hot_dry"])
    if case == "high_temp":
        t, h, n = np.random.uniform(36, 42), np.random.uniform(55, 70), np.random.uniform(5, 15)
    elif case == "very_low_humid":
        t, h, n = np.random.uniform(28, 33), np.random.uniform(35, 58), np.random.uniform(5, 12)
    elif case == "high_nh3":
        t, h, n = np.random.uniform(27, 32), np.random.uniform(65, 80), np.random.uniform(26, 55)
    elif case == "hot_dry":
        # Common real-world case: suhu 32-34 + RH < 72
        t, h, n = np.random.uniform(32.1, 34), np.random.uniform(60, 71.9), np.random.uniform(0.5, 8)
    else:
        t, h, n = np.random.uniform(36, 42), np.random.uniform(40, 58), np.random.uniform(25, 45)
    buruk_samples.append(make_sample(t, h, n, all_feature_cols))

# SEDANG samples — boundary conditions
sedang_samples = []
for _ in range(200):
    case = np.random.choice(["cool_dry", "warm_humid", "warm_dry"])
    if case == "cool_dry":
        # Suhu OK tapi humidity kurang
        t = np.random.uniform(26, 29.9)
        h = np.random.uniform(62, 74.9)
        n = np.random.uniform(1, 8)
    elif case == "warm_humid":
        # Suhu agak panas tapi humidity OK
        t = np.random.uniform(30, 32)
        h = np.random.uniform(75, 85)
        n = np.random.uniform(1, 8)
    else:
        # Suhu agak panas + humidity rendah + NH3 rendah
        t = np.random.uniform(30, 32)
        h = np.random.uniform(65, 74.9)
        n = np.random.uniform(1, 9)
    sedang_samples.append(make_sample(t, h, n, all_feature_cols))

print(f"  Added {len(bagus_samples)} bagus + {len(buruk_samples)} buruk + "
      f"{len(sedang_samples)} sedang synthetic samples")

# ============================================================
# 5. DEFINE FEATURE LISTS PER MODEL
# ============================================================

FEATURES_GRADE = [
    "temperature", "humidity", "ammonia",
    "temp_avg_1h", "humid_avg_1h", "nh3_avg_1h",
    "comfort_index", "is_daytime",
    # NEW
    "day_of_week", "temp_humid_interaction", "time_norm",
]

FEATURES_PUMP_STATE = [
    "temperature", "humidity", "ammonia",
    "temp_avg_1h", "humid_avg_1h", "humid_delta_1h",
    "hour_of_day",
    # NEW
    "day_of_week", "temp_humid_interaction", "nh3_rate_of_change",
    "temp_rolling_std", "time_norm",
]

FEATURES_PUMP_DURATION = [
    "temperature", "humidity", "ammonia",
    "temp_avg_1h", "humid_avg_1h", "humid_delta_1h",
    "hour_of_day", "comfort_index",
    # NEW
    "temp_humid_interaction", "nh3_rate_of_change",
    "temp_rolling_std", "time_norm",
]

print(f"\n  Grade features     : {len(FEATURES_GRADE)}")
print(f"  Pump state features: {len(FEATURES_PUMP_STATE)}")
print(f"  Pump dur. features : {len(FEATURES_PUMP_DURATION)}")

# ============================================================
# 6. OPTUNA HYPERPARAMETER TUNING
# ============================================================

def lgbm_search_space(trial, objective_type="classification"):
    """Constrained search space — simpler models to avoid overfitting."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 150, step=25),
        "max_depth": trial.suggest_int("max_depth", 3, 5),
        "num_leaves": trial.suggest_int("num_leaves", 7, 20),
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.3, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 100, 300),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.8),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.5, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1,
    }
    return params


# ──────────────────────────────────────────────
# 6A. TUNE GRADE MODEL
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("[6A] OPTUNA: Tuning Grade Model")
print("=" * 60)

# Prepare grade data with augmented samples
X_grade = df[FEATURES_GRADE].copy()
y_grade = df["grade"].copy()

df_bagus_g = pd.DataFrame(bagus_samples)[FEATURES_GRADE]
df_buruk_g = pd.DataFrame(buruk_samples)[FEATURES_GRADE]
df_sedang_g = pd.DataFrame(sedang_samples)[FEATURES_GRADE]

X_grade = pd.concat([X_grade, df_bagus_g, df_buruk_g, df_sedang_g], ignore_index=True)
y_grade = pd.concat([
    y_grade,
    pd.Series(["bagus"] * len(df_bagus_g)),
    pd.Series(["buruk"] * len(df_buruk_g)),
    pd.Series(["sedang"] * len(df_sedang_g)),
], ignore_index=True)

le_grade = LabelEncoder()
y_grade_enc = le_grade.fit_transform(y_grade)
n_classes = len(le_grade.classes_)

print(f"  Samples: {len(X_grade)}, Classes: {list(le_grade.classes_)}")
print(f"  Distribution: {dict(y_grade.value_counts())}")

Xg_train, Xg_test, yg_train, yg_test = train_test_split(
    X_grade, y_grade_enc,
    test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_grade_enc
)

def objective_grade(trial):
    params = lgbm_search_space(trial)
    params["objective"] = "multiclass"
    params["num_class"] = n_classes

    model = lgb.LGBMClassifier(**params)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, Xg_train, yg_train, cv=cv, scoring="f1_macro")
    return scores.mean()

t0 = time_module.time()
study_grade = optuna.create_study(direction="maximize", study_name="grade_tuning")
study_grade.optimize(objective_grade, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)
t_grade = time_module.time() - t0

best_grade_params = study_grade.best_params
best_grade_params.update({
    "objective": "multiclass", "num_class": n_classes,
    "random_state": RANDOM_STATE, "n_jobs": -1, "verbose": -1,
})

print(f"\n  Best F1-macro (CV): {study_grade.best_value:.4f}")
print(f"  Tuning time: {t_grade:.1f}s")
print(f"  Best params: {best_grade_params}")

# Train final grade model
clf_grade = lgb.LGBMClassifier(**best_grade_params)
clf_grade.fit(Xg_train, yg_train)

yg_pred = clf_grade.predict(Xg_test)
print("\n  Test Classification Report:")
print(classification_report(yg_test, yg_pred, target_names=le_grade.classes_))

cv_grade_final = cross_val_score(clf_grade, X_grade, y_grade_enc, cv=5, scoring="f1_macro")
print(f"  Final CV F1-macro (5-fold): {cv_grade_final.mean():.4f} "
      f"(+/- {cv_grade_final.std():.4f})")

# Feature importance
print("  Feature Importance (Grade):")
for feat, imp in sorted(zip(FEATURES_GRADE, clf_grade.feature_importances_), key=lambda x: -x[1]):
    print(f"    {feat:28s}: {imp}")


# ──────────────────────────────────────────────
# 6B. TUNE PUMP STATE MODEL
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("[6B] OPTUNA: Tuning Pump State Model")
print("=" * 60)

X_ps = df[FEATURES_PUMP_STATE].copy()
y_ps = df["pump_state_label"].copy()

n_neg = (y_ps == 0).sum()
n_pos = (y_ps == 1).sum()
base_scale_weight = n_neg / max(n_pos, 1)

print(f"  Samples: {len(X_ps)}, OFF: {int(n_neg)}, ON: {int(n_pos)}")
print(f"  Base scale_pos_weight: {base_scale_weight:.2f}")

Xs_train, Xs_test, ys_train, ys_test = train_test_split(
    X_ps, y_ps, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_ps
)

def objective_pump_state(trial):
    params = lgbm_search_space(trial)
    params["objective"] = "binary"
    params["scale_pos_weight"] = base_scale_weight

    model = lgb.LGBMClassifier(**params)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, Xs_train, ys_train, cv=cv, scoring="f1")
    return scores.mean()

t0 = time_module.time()
study_pump = optuna.create_study(direction="maximize", study_name="pump_state_tuning")
study_pump.optimize(objective_pump_state, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)
t_pump = time_module.time() - t0

best_pump_params = study_pump.best_params
best_pump_params.update({
    "objective": "binary", "scale_pos_weight": base_scale_weight,
    "random_state": RANDOM_STATE, "n_jobs": -1, "verbose": -1,
})

print(f"\n  Best F1 (CV): {study_pump.best_value:.4f}")
print(f"  Tuning time: {t_pump:.1f}s")
print(f"  Best params: {best_pump_params}")

# Train final pump state model
clf_pump = lgb.LGBMClassifier(**best_pump_params)
clf_pump.fit(Xs_train, ys_train)

ys_pred = clf_pump.predict(Xs_test)
print("\n  Test Classification Report:")
print(classification_report(ys_test, ys_pred, target_names=["OFF", "ON"]))

cv_pump_final = cross_val_score(clf_pump, X_ps, y_ps, cv=5, scoring="f1")
print(f"  Final CV F1 (5-fold): {cv_pump_final.mean():.4f} "
      f"(+/- {cv_pump_final.std():.4f})")

# Feature importance
print("  Feature Importance (Pump State):")
for feat, imp in sorted(zip(FEATURES_PUMP_STATE, clf_pump.feature_importances_), key=lambda x: -x[1]):
    print(f"    {feat:28s}: {imp}")


# ──────────────────────────────────────────────
# 6C. TUNE PUMP DURATION MODEL
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("[6C] OPTUNA: Tuning Pump Duration Model")
print("=" * 60)

df_on = df[df["pump_state_label"] == 1].copy()
if len(df_on) < 50:
    print("  [WARN] Terlalu sedikit data ON, menggunakan semua data")
    df_on = df.copy()

X_pd = df_on[FEATURES_PUMP_DURATION].copy()
y_pd = df_on["pump_duration_label"].copy()

print(f"  Training samples (pump ON only): {len(X_pd)}")
print(f"  Duration range: [{y_pd.min():.1f}, {y_pd.max():.1f}]s")

Xd_train, Xd_test, yd_train, yd_test = train_test_split(
    X_pd, y_pd, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

def objective_pump_duration(trial):
    params = lgbm_search_space(trial, "regression")
    params["objective"] = "regression"

    model = lgb.LGBMRegressor(**params)
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, Xd_train, yd_train, cv=cv,
                             scoring="neg_mean_absolute_error")
    return scores.mean()

t0 = time_module.time()
study_dur = optuna.create_study(direction="maximize", study_name="pump_duration_tuning")
study_dur.optimize(objective_pump_duration, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)
t_dur = time_module.time() - t0

best_dur_params = study_dur.best_params
best_dur_params.update({
    "objective": "regression",
    "random_state": RANDOM_STATE, "n_jobs": -1, "verbose": -1,
})

print(f"\n  Best neg-MAE (CV): {study_dur.best_value:.4f}")
print(f"  Tuning time: {t_dur:.1f}s")
print(f"  Best params: {best_dur_params}")

# Train final pump duration model
reg_pump = lgb.LGBMRegressor(**best_dur_params)
reg_pump.fit(Xd_train, yd_train)

yd_pred = reg_pump.predict(Xd_test)
mae = mean_absolute_error(yd_test, yd_pred)
r2 = r2_score(yd_test, yd_pred)
print(f"\n  Test MAE: {mae:.2f} detik")
print(f"  Test R² : {r2:.4f}")

# Feature importance
print("  Feature Importance (Pump Duration):")
for feat, imp in sorted(zip(FEATURES_PUMP_DURATION, reg_pump.feature_importances_), key=lambda x: -x[1]):
    print(f"    {feat:28s}: {imp}")


# ============================================================
# 7. SAVE ARTIFACTS
# ============================================================
print("\n" + "=" * 60)
print("[7] Saving artifacts...")
print("=" * 60)

path_grade_model   = os.path.join(MODELS_DIR, "model_grade_panen_v2.pkl")
path_grade_encoder = os.path.join(MODELS_DIR, "label_encoder_grade_v2.pkl")
path_pump_state    = os.path.join(MODELS_DIR, "model_pump_state_v2.pkl")
path_pump_duration = os.path.join(MODELS_DIR, "model_pump_duration_v2.pkl")

joblib.dump(clf_grade, path_grade_model, compress=3)
joblib.dump(le_grade, path_grade_encoder, compress=3)
joblib.dump(clf_pump, path_pump_state, compress=3)
joblib.dump(reg_pump, path_pump_duration, compress=3)

print(f"  ✓ {path_grade_model}")
print(f"  ✓ {path_grade_encoder}")
print(f"  ✓ {path_pump_state}")
print(f"  ✓ {path_pump_duration}")


# ============================================================
# 8. INFERENCE TEST
# ============================================================
print("\n" + "=" * 60)
print("[8] Quick inference test...")
print("=" * 60)

test_cases = [
    {"desc": "Optimal", "temp": 28.5, "humid": 78.0, "nh3": 5.0, "daytime": 1},
    {"desc": "Hot+dry",  "temp": 33.0, "humid": 60.0, "nh3": 25.0, "daytime": 1},
    {"desc": "Cool+wet", "temp": 25.0, "humid": 85.0, "nh3": 12.0, "daytime": 0},
]

for tc in test_cases:
    t, h, n = tc["temp"], tc["humid"], tc["nh3"]
    hour = 14 if tc["daytime"] else 3

    grade_row = pd.DataFrame([{
        "temperature": t, "humidity": h, "ammonia": n,
        "temp_avg_1h": t, "humid_avg_1h": h, "nh3_avg_1h": n,
        "comfort_index": calculate_comfort_index(t, h, n),
        "is_daytime": tc["daytime"],
        "day_of_week": 2,
        "temp_humid_interaction": t * h / 100.0,
        "time_norm": hour / 24.0,
    }], columns=FEATURES_GRADE)

    pred_enc = int(clf_grade.predict(grade_row)[0])
    pred_label = le_grade.inverse_transform([pred_enc])[0]
    proba = clf_grade.predict_proba(grade_row)[0]
    probs_dict = {le_grade.inverse_transform([c])[0]: round(float(p), 4)
                  for c, p in zip(clf_grade.classes_, proba)}

    pump_row = pd.DataFrame([{
        "temperature": t, "humidity": h, "ammonia": n,
        "temp_avg_1h": t, "humid_avg_1h": h,
        "humid_delta_1h": 0.0, "hour_of_day": hour,
        "day_of_week": 2,
        "temp_humid_interaction": t * h / 100.0,
        "nh3_rate_of_change": 0.0,
        "temp_rolling_std": 0.5,
        "time_norm": hour / 24.0,
    }], columns=FEATURES_PUMP_STATE)

    pump_pred = int(clf_pump.predict(pump_row)[0])
    pump_conf = max(clf_pump.predict_proba(pump_row)[0])

    dur_row = pd.DataFrame([{
        "temperature": t, "humidity": h, "ammonia": n,
        "temp_avg_1h": t, "humid_avg_1h": h,
        "humid_delta_1h": 0.0, "hour_of_day": hour,
        "comfort_index": calculate_comfort_index(t, h, n),
        "temp_humid_interaction": t * h / 100.0,
        "nh3_rate_of_change": 0.0,
        "temp_rolling_std": 0.5,
        "time_norm": hour / 24.0,
    }], columns=FEATURES_PUMP_DURATION)

    dur_pred = float(reg_pump.predict(dur_row)[0])

    print(f"\n  [{tc['desc']}] temp={t}°C, rh={h}%, nh3={n}ppm")
    print(f"    Grade : {pred_label} {probs_dict}")
    print(f"    Pump  : {'ON' if pump_pred else 'OFF'} (conf={pump_conf:.3f})")
    print(f"    Duration: {dur_pred:.1f}s")


# ============================================================
# 9. SUMMARY
# ============================================================
print("\n\n" + "=" * 60)
print("TRAINING COMPLETE — SUMMARY")
print("=" * 60)

print(f"""
Model             | Metric           | Score
------------------|------------------|--------
Grade (LightGBM)  | CV F1-macro      | {cv_grade_final.mean():.4f}
Pump State (LGBM) | CV F1            | {cv_pump_final.mean():.4f}
Pump Duration     | Test MAE         | {mae:.2f}s
Pump Duration     | Test R²          | {r2:.4f}

Optuna Trials     : {N_OPTUNA_TRIALS} per model
Total tuning time : {t_grade + t_pump + t_dur:.1f}s
Feature count     : Grade={len(FEATURES_GRADE)}, PumpState={len(FEATURES_PUMP_STATE)}, PumpDur={len(FEATURES_PUMP_DURATION)}
New features      : day_of_week, temp_humid_interaction, nh3_rate_of_change, temp_rolling_std, time_norm

Output files:
  - {path_grade_model}
  - {path_grade_encoder}
  - {path_pump_state}
  - {path_pump_duration}
""")

# Save best params as JSON for reference
import json
params_summary = {
    "optuna_trials": N_OPTUNA_TRIALS,
    "cv_folds": CV_FOLDS,
    "features_grade": FEATURES_GRADE,
    "features_pump_state": FEATURES_PUMP_STATE,
    "features_pump_duration": FEATURES_PUMP_DURATION,
    "best_params_grade": {k: v for k, v in best_grade_params.items()
                          if k not in ("random_state", "n_jobs", "verbose")},
    "best_params_pump_state": {k: v for k, v in best_pump_params.items()
                               if k not in ("random_state", "n_jobs", "verbose")},
    "best_params_pump_duration": {k: v for k, v in best_dur_params.items()
                                  if k not in ("random_state", "n_jobs", "verbose")},
    "metrics": {
        "grade_cv_f1_macro": round(float(cv_grade_final.mean()), 4),
        "pump_state_cv_f1": round(float(cv_pump_final.mean()), 4),
        "pump_duration_test_mae": round(float(mae), 2),
        "pump_duration_test_r2": round(float(r2), 4),
    },
}

params_path = os.path.join(MODELS_DIR, "optuna_best_params.json")
with open(params_path, "w") as f:
    json.dump(params_summary, f, indent=2)
print(f"Best params saved to: {params_path}")
print("=" * 60)

# ============================================================
# SwiftledML - Evaluation Report Generator
# ============================================================
# Menghasilkan semua grafik dan metrik untuk dokumen akademik:
#   1. Classification Report (Grade & Pump State)
#   2. Confusion Matrix (heatmap)
#   3. Feature Importance (bar chart)
#   4. Perbandingan Before vs After (bar chart)
#   5. Cross-Validation Scores
#   6. Confidence Distribution
#   7. Skenario Pengujian API
# ============================================================

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    mean_absolute_error, r2_score, accuracy_score
)
import joblib

# ----- CONFIG -----
# Kaggle
DATA_PATH = "/kaggle/input/swiftledml-merged-external/ml_training_with_external.csv"
MODELS_DIR = "/kaggle/working"
OUTPUT_DIR = "/kaggle/working/evaluation_output"

# Lokal (uncomment jika di lokal)
# DATA_PATH = r"c:\SwiftledML\data\ml_training_with_external.csv"
# MODELS_DIR = r"c:\SwiftledML\ai-engine"
# OUTPUT_DIR = r"c:\SwiftledML\evaluation_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Style
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

RANDOM_STATE = 42
TEST_SIZE = 0.2
HISTORY_WINDOW = 10
MIN_DELTA_THRESHOLD = 0.5
NOISE_LEVEL = 0.20

print("=" * 60)
print("SWIFTLEDML - EVALUATION REPORT GENERATOR")
print("=" * 60)

# ============================================================
# 1. LOAD DATA & MODELS
# ============================================================
print("\n[1] Loading data & models...")

df = pd.read_csv(DATA_PATH)
df.columns = [c.strip().lower() for c in df.columns]
required = ["temperature_c", "humidity_rh", "nh3_ppm"]
for col in required:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=required, how="all")

if "recorded_at" in df.columns:
    dt = pd.to_datetime(df["recorded_at"], errors="coerce")
elif "timestamp" in df.columns:
    dt = pd.to_datetime(df["timestamp"], errors="coerce")
else:
    dt = pd.Series(pd.date_range("2025-01-01", periods=len(df), freq="h"))
df["dt"] = dt
df = df.sort_values("dt")

print(f"  Dataset: {len(df):,} rows")

# Load models
grade_model = joblib.load(os.path.join(MODELS_DIR, "model_grade_panen_v2.pkl"))
grade_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder_grade_v2.pkl"))
pump_state_model = joblib.load(os.path.join(MODELS_DIR, "model_pump_state_v2.pkl"))
pump_duration_model = joblib.load(os.path.join(MODELS_DIR, "model_pump_duration_v2.pkl"))
try:
    anomaly_model = joblib.load(os.path.join(MODELS_DIR, "model_anomaly_detection_v2.pkl"))
except (ValueError, Exception) as e:
    print(f"  [WARN] Cannot load anomaly model (version mismatch): {e}")
    print("  [INFO] Retraining IsolationForest locally for evaluation...")
    from sklearn.ensemble import IsolationForest
    anomaly_model = None  # will retrain after feature engineering

try:
    anomaly_scaler = joblib.load(os.path.join(MODELS_DIR, "scaler_anomaly_v2.pkl"))
except (ValueError, Exception):
    anomaly_scaler = None
print("  All models loaded OK")

# ============================================================
# 2. FEATURE ENGINEERING (same as training)
# ============================================================
print("\n[2] Feature engineering...")

df["temperature"] = df["temperature_c"]
df["humidity"] = df["humidity_rh"]
df["ammonia"] = df["nh3_ppm"]
df["hour_of_day"] = df["dt"].dt.hour.fillna(12).astype(int)
df["is_daytime"] = ((df["hour_of_day"] >= 6) & (df["hour_of_day"] < 18)).astype(int)

ROLL_WINDOW = HISTORY_WINDOW
df["temp_avg_1h"] = df["temperature"].rolling(window=ROLL_WINDOW, min_periods=1).mean()
df["humid_avg_1h"] = df["humidity"].rolling(window=ROLL_WINDOW, min_periods=1).mean()
df["nh3_avg_1h"] = df["ammonia"].rolling(window=ROLL_WINDOW, min_periods=1).mean()

def safe_pct_change(series):
    abs_diff = series.diff()
    pct = series.pct_change() * 100
    pct = pct.where(abs_diff.abs() >= MIN_DELTA_THRESHOLD, 0.0)
    pct = pct.clip(lower=-100, upper=100)
    return pct.fillna(0.0)

df["temp_delta_1h"] = safe_pct_change(df["temperature"])
df["humid_delta_1h"] = safe_pct_change(df["humidity"])
df["nh3_delta_1h"] = safe_pct_change(df["ammonia"])

def calculate_comfort_index(temperature, humidity, ammonia):
    temp_score = max(0.0, min(1.0, 1 - abs(temperature - 28.0) / 15.0))
    humid_score = max(0.0, min(1.0, 1 - abs(humidity - 80.0) / 35.0))
    nh3_score = max(0.0, min(1.0, 1 - (ammonia / 20.0)))
    return round((temp_score * 0.35 + humid_score * 0.35 + nh3_score * 0.30) * 100.0, 2)

df["comfort_index"] = df.apply(
    lambda row: calculate_comfort_index(row["temperature"], row["humidity"], row["ammonia"]), axis=1
)

df["humid_trend_slope"] = (
    df["humidity"].rolling(window=ROLL_WINDOW, min_periods=2)
    .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0.0, raw=False)
    .fillna(0.0)
)

df["day_of_week"] = df["dt"].dt.dayofweek.fillna(0).astype(int)
df["temp_humid_interaction"] = (df["temperature"] * df["humidity"]) / 100.0
df["nh3_rate_of_change"] = df["ammonia"].diff().fillna(0.0).clip(-20, 20)
df["temp_rolling_std"] = df["temperature"].rolling(window=5, min_periods=1).std().fillna(0.0)
df["time_norm"] = df["hour_of_day"] / 24.0

# Labels
TEMP_IDEAL_MIN, TEMP_IDEAL_MAX = 26.0, 30.0
HUMID_IDEAL_MIN, HUMID_IDEAL_MAX = 75.0, 85.0
NH3_WARN, NH3_CRITICAL = 15.0, 25.0

def label_grade(row):
    temp, rh, nh3 = row["temperature"], row["humidity"], row["ammonia"]
    if nh3 > 25 or temp > 35 or temp < 20 or rh < 55 or rh > 95:
        return "buruk"
    if temp < 30 and 75 <= rh <= 85 and nh3 < 10:
        return "bagus"
    if temp > 32 and rh < 72:
        return "buruk"
    if temp > 33:
        return "buruk"
    if temp < 30 and (rh < 75 or rh > 85):
        return "sedang"
    if 30 <= temp <= 32 and rh >= 75:
        return "sedang"
    if 30 <= temp <= 32 and rh < 75:
        return "buruk" if nh3 > 10 else "sedang"
    return "sedang"

df["grade"] = df.apply(label_grade, axis=1)

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

def smart_pump_state(row):
    temp, humid, nh3, ci = row["temperature"], row["humidity"], row["ammonia"], row["comfort_index"]
    trend, hour = row["humid_trend_slope"], row["hour_of_day"]
    if nh3 >= NH3_CRITICAL: return 1
    if temp >= 33 and humid < 70: return 1
    if HUMID_IDEAL_MIN <= humid <= HUMID_IDEAL_MAX and TEMP_IDEAL_MIN <= temp <= TEMP_IDEAL_MAX: return 0
    if ci >= 75: return 0
    if trend > 0.3 and humid >= 70: return 0
    if humid < HUMID_IDEAL_MIN and temp > TEMP_IDEAL_MAX: return 1
    if humid < 65: return 1
    if nh3 >= NH3_WARN and ci < 60: return 1
    if row["is_daytime"] == 1 and humid < 72 and trend < -0.2: return 1
    return 0

df["pump_state_label"] = df.apply(smart_pump_state, axis=1)
if df["pump_state_label"].nunique() < 2:
    df["pump_state_label"] = (df["comfort_index"] < df["comfort_index"].median()).astype(int)

def smart_pump_duration(row):
    if row["pump_state_label"] == 0: return 0.0
    humid_gap = max(0.0, 80.0 - row["humidity"])
    base = humid_gap * 0.5
    tf = 1.0 + max(0.0, row["temperature"] - TEMP_IDEAL_MAX) * 0.08
    timef = 1.15 if row["is_daytime"] == 1 else 1.0
    trendf = 1.0 + max(0.0, -row["humid_trend_slope"]) * 0.15
    nf = 1.0 + max(0.0, row["ammonia"] - NH3_WARN) * 0.03
    return round(max(5.0, min(base * tf * timef * trendf * nf, 30.0)), 1)

df["pump_duration_label"] = df.apply(smart_pump_duration, axis=1)

all_cols = [
    "temperature", "humidity", "ammonia", "temp_avg_1h", "humid_avg_1h", "nh3_avg_1h",
    "temp_delta_1h", "humid_delta_1h", "nh3_delta_1h", "comfort_index", "is_daytime",
    "hour_of_day", "humid_trend_slope", "day_of_week", "temp_humid_interaction",
    "nh3_rate_of_change", "temp_rolling_std", "time_norm",
]
df = df.dropna(subset=all_cols).reset_index(drop=True)
print(f"  Final dataset: {len(df):,} rows")

# Feature lists
FEATURES_GRADE = ["temperature","humidity","ammonia","temp_avg_1h","humid_avg_1h","nh3_avg_1h",
                  "comfort_index","is_daytime","day_of_week","temp_humid_interaction","time_norm"]
FEATURES_PUMP_STATE = ["temperature","humidity","ammonia","temp_avg_1h","humid_avg_1h","humid_delta_1h",
                       "hour_of_day","day_of_week","temp_humid_interaction","nh3_rate_of_change",
                       "temp_rolling_std","time_norm"]
FEATURES_PUMP_DURATION = ["temperature","humidity","ammonia","temp_avg_1h","humid_avg_1h","humid_delta_1h",
                          "hour_of_day","comfort_index","temp_humid_interaction","nh3_rate_of_change",
                          "temp_rolling_std","time_norm"]
FEATURES_ANOMALY = ["temperature","humidity","ammonia","hour_of_day","temp_delta_1h","humid_delta_1h",
                    "nh3_delta_1h","comfort_index"]

# ============================================================
# 3. GRADE MODEL EVALUATION
# ============================================================
print("\n[3] Evaluating Grade Model...")

le = LabelEncoder()
X_grade = df[FEATURES_GRADE].copy()
y_grade = le.fit_transform(df["grade"])
Xg_train, Xg_test, yg_train, yg_test = train_test_split(
    X_grade, y_grade, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_grade
)

yg_pred = grade_model.predict(Xg_test)
yg_proba = grade_model.predict_proba(Xg_test)

# Classification Report
cr_grade = classification_report(yg_test, yg_pred, target_names=le.classes_, output_dict=True)
cr_grade_text = classification_report(yg_test, yg_pred, target_names=le.classes_)
print(cr_grade_text)

# Save text report
with open(os.path.join(OUTPUT_DIR, "classification_report_grade.txt"), "w") as f:
    f.write("CLASSIFICATION REPORT - GRADE PREDICTION\n")
    f.write("=" * 50 + "\n")
    f.write(cr_grade_text)
    f.write(f"\nAccuracy: {accuracy_score(yg_test, yg_pred):.4f}\n")

# --- Fig 1: Confusion Matrix - Grade ---
cm_grade = confusion_matrix(yg_test, yg_pred)
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.heatmap(cm_grade, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax,
            annot_kws={"size": 16})
ax.set_xlabel('Prediksi', fontsize=13)
ax.set_ylabel('Aktual', fontsize=13)
ax.set_title('Confusion Matrix - Grade Prediction (LightGBM)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_grade.png"), dpi=150)
plt.close()
print("  Saved: confusion_matrix_grade.png")

# --- Fig 2: Feature Importance - Grade ---
importances_grade = grade_model.feature_importances_
feat_imp_grade = sorted(zip(FEATURES_GRADE, importances_grade), key=lambda x: x[1])
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(feat_imp_grade)))
ax.barh([f[0] for f in feat_imp_grade], [f[1] for f in feat_imp_grade], color=colors)
ax.set_xlabel('Importance Score', fontsize=12)
ax.set_title('Feature Importance - Grade Prediction Model', fontsize=14, fontweight='bold')
for i, (feat, imp) in enumerate(feat_imp_grade):
    ax.text(imp + 5, i, str(imp), va='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance_grade.png"), dpi=150)
plt.close()
print("  Saved: feature_importance_grade.png")

# --- Fig 3: Confidence Distribution - Grade ---
max_conf = yg_proba.max(axis=1)
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.hist(max_conf, bins=50, color='#3498db', edgecolor='white', alpha=0.85)
ax.axvline(x=0.7, color='red', linestyle='--', label='Threshold 0.7')
ax.set_xlabel('Confidence Score', fontsize=12)
ax.set_ylabel('Jumlah Prediksi', fontsize=12)
ax.set_title('Distribusi Confidence - Grade Prediction', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
pct_above = (max_conf > 0.7).sum() / len(max_conf) * 100
ax.text(0.72, ax.get_ylim()[1] * 0.9, f'{pct_above:.1f}% > 0.7', fontsize=12, color='red')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confidence_distribution_grade.png"), dpi=150)
plt.close()
print("  Saved: confidence_distribution_grade.png")

# Cross-validation
cv_grade = cross_val_score(grade_model, X_grade, y_grade, cv=5, scoring="f1_macro")
print(f"  CV F1-macro: {cv_grade.mean():.4f} (+/- {cv_grade.std():.4f})")

# ============================================================
# 4. PUMP STATE MODEL EVALUATION
# ============================================================
print("\n[4] Evaluating Pump State Model...")

X_ps = df[FEATURES_PUMP_STATE].copy()
y_ps = df["pump_state_label"].copy()
Xs_train, Xs_test, ys_train, ys_test = train_test_split(
    X_ps, y_ps, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_ps
)

ys_pred = pump_state_model.predict(Xs_test)
cr_pump_text = classification_report(ys_test, ys_pred, target_names=["OFF", "ON"])
print(cr_pump_text)

with open(os.path.join(OUTPUT_DIR, "classification_report_pump.txt"), "w") as f:
    f.write("CLASSIFICATION REPORT - PUMP STATE\n")
    f.write("=" * 50 + "\n")
    f.write(cr_pump_text)

# --- Fig 4: Confusion Matrix - Pump State ---
cm_pump = confusion_matrix(ys_test, ys_pred)
fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))
sns.heatmap(cm_pump, annot=True, fmt='d', cmap='Greens',
            xticklabels=["OFF", "ON"], yticklabels=["OFF", "ON"], ax=ax,
            annot_kws={"size": 18})
ax.set_xlabel('Prediksi', fontsize=13)
ax.set_ylabel('Aktual', fontsize=13)
ax.set_title('Confusion Matrix - Pump State (LightGBM)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_pump.png"), dpi=150)
plt.close()
print("  Saved: confusion_matrix_pump.png")

# --- Fig 5: Feature Importance - Pump ---
importances_pump = pump_state_model.feature_importances_
feat_imp_pump = sorted(zip(FEATURES_PUMP_STATE, importances_pump), key=lambda x: x[1])
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(feat_imp_pump)))
ax.barh([f[0] for f in feat_imp_pump], [f[1] for f in feat_imp_pump], color=colors)
ax.set_xlabel('Importance Score', fontsize=12)
ax.set_title('Feature Importance - Pump State Model', fontsize=14, fontweight='bold')
for i, (feat, imp) in enumerate(feat_imp_pump):
    ax.text(imp + 2, i, str(imp), va='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance_pump.png"), dpi=150)
plt.close()
print("  Saved: feature_importance_pump.png")

cv_pump = cross_val_score(pump_state_model, X_ps, y_ps, cv=5, scoring="f1")
print(f"  CV F1: {cv_pump.mean():.4f} (+/- {cv_pump.std():.4f})")

# ============================================================
# 5. PUMP DURATION EVALUATION
# ============================================================
print("\n[5] Evaluating Pump Duration Model...")

df_on = df[df["pump_state_label"] == 1].copy()
if len(df_on) < 50:
    df_on = df.copy()
X_pd = df_on[FEATURES_PUMP_DURATION].copy()
y_pd = df_on["pump_duration_label"].copy()
Xd_train, Xd_test, yd_train, yd_test = train_test_split(
    X_pd, y_pd, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
yd_pred = pump_duration_model.predict(Xd_test)
mae = mean_absolute_error(yd_test, yd_pred)
r2 = r2_score(yd_test, yd_pred)
print(f"  MAE: {mae:.2f}s | R2: {r2:.4f}")

# --- Fig 6: Actual vs Predicted Duration ---
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.scatter(yd_test, yd_pred, alpha=0.3, s=10, color='#2ecc71')
lims = [min(yd_test.min(), yd_pred.min()), max(yd_test.max(), yd_pred.max())]
ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect prediction')
ax.set_xlabel('Durasi Aktual (detik)', fontsize=12)
ax.set_ylabel('Durasi Prediksi (detik)', fontsize=12)
ax.set_title(f'Pump Duration: Actual vs Predicted\nMAE={mae:.2f}s | R\u00b2={r2:.4f}', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted_duration.png"), dpi=150)
plt.close()
print("  Saved: actual_vs_predicted_duration.png")

# ============================================================
# 6. ANOMALY DETECTION EVALUATION
# ============================================================
# Retrain anomaly model locally if Kaggle version incompatible
X_anom = df[FEATURES_ANOMALY].copy()

if anomaly_scaler is None:
    anomaly_scaler = StandardScaler()
    X_anom_scaled = anomaly_scaler.fit_transform(X_anom)
else:
    X_anom_scaled = anomaly_scaler.transform(X_anom)

if anomaly_model is None:
    from sklearn.ensemble import IsolationForest
    anomaly_model = IsolationForest(
        n_estimators=200, contamination=0.02, random_state=RANDOM_STATE, n_jobs=-1
    )
    anomaly_model.fit(X_anom_scaled)
    print("  Retrained IsolationForest locally for evaluation")

anom_preds = anomaly_model.predict(X_anom_scaled)
anom_scores = anomaly_model.decision_function(X_anom_scaled)

n_anomaly = (anom_preds == -1).sum()
n_normal = (anom_preds == 1).sum()
print(f"  Normal: {n_normal:,} | Anomaly: {n_anomaly:,} ({n_anomaly/len(anom_preds)*100:.1f}%)")

# --- Fig 7: Anomaly Score Distribution ---
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.hist(anom_scores[anom_preds == 1], bins=50, alpha=0.7, color='#3498db', label=f'Normal ({n_normal:,})')
ax.hist(anom_scores[anom_preds == -1], bins=30, alpha=0.7, color='#e74c3c', label=f'Anomaly ({n_anomaly:,})')
ax.set_xlabel('Anomaly Score (IsolationForest)', fontsize=12)
ax.set_ylabel('Jumlah Data', fontsize=12)
ax.set_title('Distribusi Anomaly Score', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "anomaly_score_distribution.png"), dpi=150)
plt.close()
print("  Saved: anomaly_score_distribution.png")

# ============================================================
# 7. COMPARISON CHART: Before vs After
# ============================================================
print("\n[7] Creating comparison charts...")

# Metrics from optuna_best_params.json (before = old, after = new)
metrics_before = {"Grade F1-macro": 0.3283, "Pump State F1": 0.4702, "Duration R\u00b2": 0.924}
metrics_after = {"Grade F1-macro": 0.4998, "Pump State F1": 0.9618, "Duration R\u00b2": 0.9927}

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
x = np.arange(len(metrics_before))
width = 0.35
bars1 = ax.bar(x - width/2, list(metrics_before.values()), width, label='Sebelum (Real Only)',
               color='#e74c3c', alpha=0.85, edgecolor='white')
bars2 = ax.bar(x + width/2, list(metrics_after.values()), width, label='Sesudah (+ External GAMS)',
               color='#2ecc71', alpha=0.85, edgecolor='white')

ax.set_ylabel('Score', fontsize=13)
ax.set_title('Perbandingan Performa Model: Sebelum vs Sesudah External Dataset',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(list(metrics_before.keys()), fontsize=12)
ax.legend(fontsize=11, loc='upper left')
ax.set_ylim(0, 1.15)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
            f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
            f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "comparison_before_after.png"), dpi=150)
plt.close()
print("  Saved: comparison_before_after.png")

# --- Fig 9: Dataset Composition Pie Chart ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Dataset composition
sizes_data = [24190, 10000]
labels_data = ['Data Real\n(24,190 rows)', 'Data External GAMS\n(10,000 rows)']
colors_data = ['#3498db', '#e67e22']
axes[0].pie(sizes_data, labels=labels_data, colors=colors_data, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 11})
axes[0].set_title('Komposisi Dataset Training', fontsize=13, fontweight='bold')

# Grade distribution
grade_dist = df["grade"].value_counts()
sizes_grade = [grade_dist.get(g, 0) for g in ["bagus", "sedang", "buruk"]]
labels_grade = [f'Bagus\n({sizes_grade[0]:,})', f'Sedang\n({sizes_grade[1]:,})', f'Buruk\n({sizes_grade[2]:,})']
colors_grade = ['#2ecc71', '#f39c12', '#e74c3c']
axes[1].pie(sizes_grade, labels=labels_grade, colors=colors_grade, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 11})
axes[1].set_title('Distribusi Label Grade', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "dataset_composition.png"), dpi=150)
plt.close()
print("  Saved: dataset_composition.png")

# --- Fig 10: Cross-Validation Scores ---
fig, ax = plt.subplots(1, 1, figsize=(9, 5))
cv_data = {
    'Grade (F1-macro)': cv_grade,
    'Pump State (F1)': cv_pump,
}
positions = range(len(cv_data))
bp = ax.boxplot(list(cv_data.values()), positions=positions, widths=0.5, patch_artist=True)
colors_cv = ['#3498db', '#2ecc71']
for patch, color in zip(bp['boxes'], colors_cv):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_xticklabels(list(cv_data.keys()), fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Cross-Validation Scores (5-Fold)', fontsize=14, fontweight='bold')
for i, (name, scores) in enumerate(cv_data.items()):
    ax.text(i, scores.mean() + 0.01, f'Mean: {scores.mean():.4f}', ha='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cross_validation_scores.png"), dpi=150)
plt.close()
print("  Saved: cross_validation_scores.png")

# ============================================================
# 8. SUMMARY METRICS TABLE
# ============================================================
print("\n[8] Creating summary...")

summary = {
    "dataset": {
        "total_rows": len(df),
        "real_data_rows": 24190,
        "external_data_rows": 10000,
        "external_source": "GAMS Indoor Air Quality (GitHub)",
        "features_grade": len(FEATURES_GRADE),
        "features_pump_state": len(FEATURES_PUMP_STATE),
        "features_pump_duration": len(FEATURES_PUMP_DURATION),
        "features_anomaly": len(FEATURES_ANOMALY),
    },
    "grade_model": {
        "algorithm": "LightGBM Classifier",
        "classes": list(le.classes_),
        "test_accuracy": round(accuracy_score(yg_test, yg_pred), 4),
        "test_f1_macro": round(f1_score(yg_test, yg_pred, average='macro'), 4),
        "cv_f1_macro_mean": round(cv_grade.mean(), 4),
        "cv_f1_macro_std": round(cv_grade.std(), 4),
        "confidence_mean": round(float(max_conf.mean()), 4),
        "confidence_above_70pct": round(float((max_conf > 0.7).mean() * 100), 1),
        "per_class": {cls: {
            "precision": round(cr_grade[cls]["precision"], 4),
            "recall": round(cr_grade[cls]["recall"], 4),
            "f1": round(cr_grade[cls]["f1-score"], 4),
        } for cls in le.classes_},
    },
    "pump_state_model": {
        "algorithm": "LightGBM Classifier",
        "test_accuracy": round(accuracy_score(ys_test, ys_pred), 4),
        "test_f1": round(f1_score(ys_test, ys_pred), 4),
        "cv_f1_mean": round(cv_pump.mean(), 4),
        "cv_f1_std": round(cv_pump.std(), 4),
    },
    "pump_duration_model": {
        "algorithm": "LightGBM Regressor",
        "test_mae_seconds": round(mae, 2),
        "test_r2": round(r2, 4),
    },
    "anomaly_model": {
        "algorithm": "Isolation Forest",
        "normal_count": int(n_normal),
        "anomaly_count": int(n_anomaly),
        "anomaly_rate_pct": round(n_anomaly / len(anom_preds) * 100, 1),
    },
    "improvement_vs_before": {
        "grade_f1_before": 0.3283,
        "grade_f1_after": 0.4998,
        "grade_improvement_pct": round((0.4998 - 0.3283) / 0.3283 * 100, 1),
        "pump_f1_before": 0.4702,
        "pump_f1_after": 0.9618,
        "pump_improvement_pct": round((0.9618 - 0.4702) / 0.4702 * 100, 1),
        "duration_r2_before": 0.924,
        "duration_r2_after": 0.9927,
    },
}

with open(os.path.join(OUTPUT_DIR, "evaluation_summary.json"), "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print("  Saved: evaluation_summary.json")

# ============================================================
# DONE
# ============================================================
print(f"""
{'='*60}
EVALUATION REPORT COMPLETE
{'='*60}

Output files di: {OUTPUT_DIR}

  Grafik:
    1. confusion_matrix_grade.png      - Confusion matrix grade
    2. confusion_matrix_pump.png       - Confusion matrix pump state
    3. feature_importance_grade.png    - Feature importance grade
    4. feature_importance_pump.png     - Feature importance pump
    5. confidence_distribution_grade.png - Distribusi confidence
    6. actual_vs_predicted_duration.png - Actual vs predicted duration
    7. anomaly_score_distribution.png  - Distribusi anomaly score
    8. comparison_before_after.png     - Perbandingan sebelum/sesudah
    9. dataset_composition.png         - Komposisi dataset & grade
   10. cross_validation_scores.png     - Cross-validation boxplot

  Laporan:
    - classification_report_grade.txt  - Detail per kelas
    - classification_report_pump.txt   - Detail pump state
    - evaluation_summary.json          - Semua metrik (JSON)

{'='*60}
""")

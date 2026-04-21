# ============================================================
# Preprocess External Dataset → SwiftledML Format
# ============================================================
# Sumber: GAMS Indoor Air Quality (GitHub)
#   - 135K rows, 1-minute interval
#   - Kolom: ts, co2, humidity, pm10, pm25, temperature, voc
#
# Output: data/external_aligned.csv
#   - Kolom: recorded_at, temperature_c, humidity_rh, nh3_ppm, source
#   - Siap di-merge dengan ml_training_smart.csv
# ============================================================

import numpy as np
import pandas as pd
import os

np.random.seed(42)

# ----- CONFIG -----
GAMS_PATH = r"c:\SwiftledML\data\external_gams_indoor.csv"
OUTPUT_DIR = r"c:\SwiftledML\data"

print("=" * 60)
print("PREPROCESS EXTERNAL DATASET -> SWIFTLEDML FORMAT")
print("=" * 60)

# ============================================================
# 1. LOAD GAMS INDOOR DATASET
# ============================================================
print("\n[1] Loading GAMS Indoor dataset...")

df_gams = pd.read_csv(GAMS_PATH)
print(f"  Shape: {df_gams.shape}")
print(f"  Columns: {list(df_gams.columns)}")
print(f"  Temp range: {df_gams['temperature'].min():.1f} - {df_gams['temperature'].max():.1f}°C")
print(f"  Humidity range: {df_gams['humidity'].min():.1f} - {df_gams['humidity'].max():.1f}%")
print(f"  CO2 range: {df_gams['co2'].min():.0f} - {df_gams['co2'].max():.0f} ppm")
print(f"  VOC range: {df_gams['voc'].min():.3f} - {df_gams['voc'].max():.3f}")

# ============================================================
# 2. MAP KE FORMAT SWIFTLEDML
# ============================================================
print("\n[2] Mapping to SwiftledML format...")

# --- Temperature: GAMS = 17-28°C → perlu shift ke range tropis (24-36°C) ---
# Swiftlet house di Indonesia: typically 26-35°C
# GAMS recorded in temperate climate office → shift +6°C untuk simulasi tropis
TEMP_SHIFT = 6.0
df_gams["temperature_c"] = df_gams["temperature"] + TEMP_SHIFT

# --- Humidity: GAMS = 22-72% → extend ke range swiftlet (55-90%) ---
# GAMS humidity agak rendah (indoor AC). Scale up sedikit.
# Formula: h_new = h_original * 1.15 + 5 (capped at 95)
df_gams["humidity_rh"] = np.clip(df_gams["humidity"] * 1.15 + 5, 40, 95)

# --- NH3: proxy dari CO2 + VOC ---
# CO2 dan NH3 sama-sama gas sisa metabolisme.
# Mapping: CO2 400-2600ppm → NH3 0.5-25ppm (proportional)
# Formula: nh3 = (co2 - 400) / (2600 - 400) * 24.5 + 0.5
co2_min, co2_max = 400, 2600
nh3_min, nh3_max = 0.5, 25.0
df_gams["nh3_ppm"] = np.clip(
    (df_gams["co2"] - co2_min) / (co2_max - co2_min) * (nh3_max - nh3_min) + nh3_min,
    nh3_min, nh3_max
)

# Tambah noise realistis dari VOC (korrelasi gas indoor)
voc_noise = df_gams["voc"] * np.random.uniform(0.5, 2.0, len(df_gams))
df_gams["nh3_ppm"] = np.clip(df_gams["nh3_ppm"] + voc_noise, 0.1, 50.0)

# --- Timestamp ---
df_gams["recorded_at"] = pd.to_datetime(df_gams["ts"]).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

# --- Source marker ---
df_gams["source"] = "external_gams"

print(f"  After mapping:")
print(f"    Temp   : {df_gams['temperature_c'].min():.1f} - {df_gams['temperature_c'].max():.1f}°C "
      f"(mean {df_gams['temperature_c'].mean():.1f})")
print(f"    RH     : {df_gams['humidity_rh'].min():.1f} - {df_gams['humidity_rh'].max():.1f}% "
      f"(mean {df_gams['humidity_rh'].mean():.1f})")
print(f"    NH3    : {df_gams['nh3_ppm'].min():.2f} - {df_gams['nh3_ppm'].max():.2f} ppm "
      f"(mean {df_gams['nh3_ppm'].mean():.2f})")

# ============================================================
# 3. FILTER DATA YANG REALISTIS UNTUK SWIFTLET
# ============================================================
print("\n[3] Filtering to realistic swiftlet conditions...")

# Keep only rows yang masuk range swiftlet house
# Suhu: 22-40°C (walet bisa toleransi)
# RH: 55-95%
# NH3: 0.1-50 ppm
mask = (
    (df_gams["temperature_c"] >= 22) & (df_gams["temperature_c"] <= 40) &
    (df_gams["humidity_rh"] >= 55) & (df_gams["humidity_rh"] <= 95) &
    (df_gams["nh3_ppm"] >= 0.1) & (df_gams["nh3_ppm"] <= 50)
)
df_filtered = df_gams[mask].copy()
print(f"  Kept: {len(df_filtered)} / {len(df_gams)} rows ({len(df_filtered)/len(df_gams)*100:.1f}%)")

# ============================================================
# 4. SAMPLE — jangan terlalu banyak sampai flooding real data
# ============================================================
print("\n[4] Sampling to balance with real data...")

# Real data kita: ~24K rows. External max 30% dari total final.
# Target: ~10K rows external
TARGET_EXTERNAL = 10000

if len(df_filtered) > TARGET_EXTERNAL:
    df_sampled = df_filtered.sample(n=TARGET_EXTERNAL, random_state=42)
    print(f"  Sampled {TARGET_EXTERNAL} from {len(df_filtered)} rows")
else:
    df_sampled = df_filtered
    print(f"  Using all {len(df_sampled)} rows (under target)")

# ============================================================
# 5. LABEL GRADE (consistent with training scripts)
# ============================================================
print("\n[5] Labeling grade (consistent with kaggle_train_optuna.py)...")

def label_grade(t, h, n):
    if n > 25 or t > 35 or t < 20 or h < 55 or h > 95:
        return "buruk"
    if t < 30 and 75 <= h <= 85 and n < 10:
        return "bagus"
    if t > 32 and h < 72:
        return "buruk"
    if t > 33:
        return "buruk"
    if t < 30 and (h < 75 or h > 85):
        return "sedang"
    if 30 <= t <= 32 and h >= 75:
        return "sedang"
    if 30 <= t <= 32 and h < 75:
        return "buruk" if n > 10 else "sedang"
    return "sedang"

df_sampled["grade"] = df_sampled.apply(
    lambda r: label_grade(r["temperature_c"], r["humidity_rh"], r["nh3_ppm"]), axis=1
)

grade_dist = df_sampled["grade"].value_counts()
print(f"  Grade distribution:")
for g in ["bagus", "sedang", "buruk"]:
    v = grade_dist.get(g, 0)
    print(f"    {g}: {v:5d} ({v/len(df_sampled)*100:.1f}%)")

# ============================================================
# 6. SAVE
# ============================================================
print("\n[6] Saving...")

final_cols = ["recorded_at", "temperature_c", "humidity_rh", "nh3_ppm"]
out_aligned = os.path.join(OUTPUT_DIR, "external_aligned.csv")
df_sampled[final_cols].to_csv(out_aligned, index=False)
print(f"  ✓ {out_aligned} ({len(df_sampled)} rows)")

# Also save debug version with source and grade
out_debug = os.path.join(OUTPUT_DIR, "external_aligned_debug.csv")
df_sampled[final_cols + ["source", "grade"]].to_csv(out_debug, index=False)
print(f"  ✓ {out_debug} (with source & grade columns)")

# ============================================================
# 7. COMPARE WITH REAL DATA
# ============================================================
print("\n[7] Comparison: Real vs External data coverage...")

real = pd.read_csv(os.path.join(OUTPUT_DIR, "ml_training_smart.csv"))
ext = df_sampled

print(f"\n  {'Metric':<20} {'Real Data':>15} {'External':>15}")
print(f"  {'-'*50}")
print(f"  {'Rows':<20} {len(real):>15,} {len(ext):>15,}")
print(f"  {'Temp min':<20} {real['temperature_c'].min():>15.1f} {ext['temperature_c'].min():>15.1f}")
print(f"  {'Temp max':<20} {real['temperature_c'].max():>15.1f} {ext['temperature_c'].max():>15.1f}")
print(f"  {'Temp mean':<20} {real['temperature_c'].mean():>15.1f} {ext['temperature_c'].mean():>15.1f}")
print(f"  {'RH min':<20} {real['humidity_rh'].min():>15.1f} {ext['humidity_rh'].min():>15.1f}")
print(f"  {'RH max':<20} {real['humidity_rh'].max():>15.1f} {ext['humidity_rh'].max():>15.1f}")
print(f"  {'RH mean':<20} {real['humidity_rh'].mean():>15.1f} {ext['humidity_rh'].mean():>15.1f}")
print(f"  {'NH3 min':<20} {real['nh3_ppm'].min():>15.2f} {ext['nh3_ppm'].min():>15.2f}")
print(f"  {'NH3 max':<20} {real['nh3_ppm'].max():>15.2f} {ext['nh3_ppm'].max():>15.2f}")
print(f"  {'NH3 mean':<20} {real['nh3_ppm'].mean():>15.2f} {ext['nh3_ppm'].mean():>15.2f}")

# Grade comparison
real["grade_check"] = real.apply(
    lambda r: label_grade(r["temperature_c"], r["humidity_rh"], r["nh3_ppm"]), axis=1
)
print(f"\n  Grade Coverage:")
print(f"  {'Grade':<10} {'Real':>10} {'External':>10} {'Combined':>10}")
for g in ["bagus", "sedang", "buruk"]:
    r_cnt = (real["grade_check"] == g).sum()
    e_cnt = (ext["grade"] == g).sum()
    print(f"  {g:<10} {r_cnt:>10,} {e_cnt:>10,} {r_cnt + e_cnt:>10,}")

print(f"""
{'='*60}
DONE!
{'='*60}

Selanjutnya, merge ke training dataset:

  # Di training script, ganti loading data:
  df_real = pd.read_csv("data/ml_training_smart.csv")
  df_ext  = pd.read_csv("data/external_aligned.csv")
  df = pd.concat([df_real, df_ext], ignore_index=True)

Atau buat merged file baru dengan scripts/merge_with_external.py
{'='*60}
""")

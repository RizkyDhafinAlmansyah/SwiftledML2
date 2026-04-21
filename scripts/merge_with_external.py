# ============================================================
# Merge Real Data + External Data -> Training Dataset
# ============================================================
# Input:
#   - data/ml_training_smart.csv     (24K rows - real + synthetic bagus)
#   - data/external_aligned.csv      (10K rows - GAMS indoor, aligned)
#
# Output:
#   - data/ml_training_with_external.csv  (ready for training)
# ============================================================

import pandas as pd
import numpy as np

np.random.seed(42)

# ----- CONFIG -----
REAL_PATH = r"c:\SwiftledML\data\ml_training_smart.csv"
EXTERNAL_PATH = r"c:\SwiftledML\data\external_aligned.csv"
OUTPUT_PATH = r"c:\SwiftledML\data\ml_training_with_external.csv"

print("=" * 60)
print("MERGE REAL + EXTERNAL DATA")
print("=" * 60)

# Load
df_real = pd.read_csv(REAL_PATH)
df_ext = pd.read_csv(EXTERNAL_PATH)

print(f"  Real data    : {len(df_real):,} rows")
print(f"  External data: {len(df_ext):,} rows")

# Ensure same columns
final_cols = ["recorded_at", "temperature_c", "humidity_rh", "nh3_ppm"]
df_real = df_real[final_cols]
df_ext = df_ext[final_cols]

# Merge
df_merged = pd.concat([df_real, df_ext], ignore_index=True)

# Shuffle
df_merged = df_merged.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"  Merged total : {len(df_merged):,} rows")
print(f"  External %   : {len(df_ext)/len(df_merged)*100:.1f}%")

# Grade distribution check
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

df_merged["grade_check"] = df_merged.apply(
    lambda r: label_grade(r["temperature_c"], r["humidity_rh"], r["nh3_ppm"]), axis=1
)

total = len(df_merged)
print(f"\n  Grade distribution (merged):")
for g in ["bagus", "sedang", "buruk"]:
    v = (df_merged["grade_check"] == g).sum()
    print(f"    {g}: {v:6,} ({v/total*100:.1f}%)")

print(f"\n  Sensor ranges (merged):")
print(f"    Temp : {df_merged['temperature_c'].min():.1f} - {df_merged['temperature_c'].max():.1f} (mean {df_merged['temperature_c'].mean():.1f})")
print(f"    RH   : {df_merged['humidity_rh'].min():.1f} - {df_merged['humidity_rh'].max():.1f} (mean {df_merged['humidity_rh'].mean():.1f})")
print(f"    NH3  : {df_merged['nh3_ppm'].min():.2f} - {df_merged['nh3_ppm'].max():.2f} (mean {df_merged['nh3_ppm'].mean():.2f})")

# Save (tanpa kolom grade_check)
df_merged[final_cols].to_csv(OUTPUT_PATH, index=False)
print(f"\n  Saved: {OUTPUT_PATH}")
print(f"  Size : {len(df_merged):,} rows")

print(f"""
{'='*60}
DONE!
{'='*60}

Sekarang update DATA_PATH di training scripts:

  # kaggle_train_optuna.py / kaggle_train_grade.py:
  DATA_PATH = "data/ml_training_with_external.csv"

{'='*60}
""")

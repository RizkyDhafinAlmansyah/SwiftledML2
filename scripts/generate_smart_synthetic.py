# ============================================================
# Smart Synthetic Data Generator
# ============================================================
# Analisis real data → identifikasi gap → generate HANYA
# untuk variasi yang kurang. Tidak flooding dataset.
#
# Prinsip: Synthetic = pelengkap, bukan pengganti real data.
# ============================================================

import numpy as np
import pandas as pd
import os

np.random.seed(42)

# ----- CONFIG -----
REAL_DATA_PATHS = [
    r"c:\SwiftledML\data\Dataset_terbaru_real\sensor_data_lantai_1_combined (1).csv",
    r"c:\SwiftledML\data\Dataset_terbaru_real\sensor_data_lantai_2_combined (1).csv",
]
OUTPUT_DIR = r"c:\SwiftledML\data"

# Label rules (sama dengan kaggle_train_optuna.py)
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


# ============================================================
# 1. ANALISIS DATA REAL
# ============================================================
print("=" * 60)
print("SMART SYNTHETIC DATA GENERATOR")
print("=" * 60)

print("\n[1] Loading & analyzing real data...")
dfs = []
for p in REAL_DATA_PATHS:
    df = pd.read_csv(p)
    dfs.append(df)
real = pd.concat(dfs, ignore_index=True)
real = real[real["temperature_c"] < 50]  # remove outliers

# Label real data
real["grade"] = real.apply(
    lambda r: label_grade(r["temperature_c"], r["humidity_rh"], r["nh3_ppm"]),
    axis=1,
)

n_real = len(real)
vc = real["grade"].value_counts()
print(f"  Total real data: {n_real} rows")
print(f"  Real distribution:")
for g in ["bagus", "sedang", "buruk"]:
    v = vc.get(g, 0)
    print(f"    {g}: {v:6d} ({v/n_real*100:.1f}%)")

print(f"\n  Real sensor ranges:")
print(f"    Suhu : {real.temperature_c.min():.1f} - {real.temperature_c.max():.1f}°C (mean {real.temperature_c.mean():.1f})")
print(f"    RH   : {real.humidity_rh.min():.1f} - {real.humidity_rh.max():.1f}% (mean {real.humidity_rh.mean():.1f})")
print(f"    NH3  : {real.nh3_ppm.min():.2f} - {real.nh3_ppm.max():.2f} ppm (mean {real.nh3_ppm.mean():.2f})")


# ============================================================
# 2. IDENTIFIKASI GAP
# ============================================================
print("\n[2] Identifying gaps...")

n_bagus = vc.get("bagus", 0)
n_sedang = vc.get("sedang", 0)
n_buruk = vc.get("buruk", 0)

# Target: setiap kelas minimal 20% dari total
# Tapi synthetic TIDAK boleh lebih dari 50% total final
target_per_class = int(n_real * 0.20)

need_bagus = max(0, target_per_class - n_bagus)
need_sedang = max(0, target_per_class - n_sedang)
need_buruk = max(0, target_per_class - n_buruk)

# Cap synthetic supaya tidak dominasi
max_synthetic = int(n_real * 0.50)  # max 50% dari real
total_need = need_bagus + need_sedang + need_buruk
if total_need > max_synthetic:
    ratio = max_synthetic / total_need
    need_bagus = int(need_bagus * ratio)
    need_sedang = int(need_sedang * ratio)
    need_buruk = int(need_buruk * ratio)

print(f"  Target per class (20% of real): {target_per_class}")
print(f"  Need to generate:")
print(f"    bagus : {need_bagus:5d} {'← PRIORITY! Real data hampir tidak ada' if need_bagus > 0 else '← cukup'}")
print(f"    sedang: {need_sedang:5d} {'← perlu tambahan' if need_sedang > 0 else '← cukup'}")
print(f"    buruk : {need_buruk:5d} {'← perlu tambahan' if need_buruk > 0 else '← cukup'}")


# ============================================================
# 3. GENERATE TARGETED SYNTHETIC DATA
# ============================================================
print("\n[3] Generating targeted synthetic data...")

from datetime import datetime, timedelta

START_DATE = datetime(2025, 10, 1)

def gen_timestamp(idx):
    return (START_DATE + timedelta(minutes=idx * 6)).strftime("%Y-%m-%dT%H:%M:%SZ")

synthetic_rows = []

# --- BAGUS: suhu < 30, RH 75-85, NH3 < 10 ---
if need_bagus > 0:
    print(f"  Generating {need_bagus} 'bagus' samples...")
    for i in range(need_bagus):
        t = np.random.uniform(26.0, 29.9)
        h = np.random.uniform(75.0, 85.0)
        n = np.random.uniform(0.5, 9.0)
        # Tambah sedikit variasi realistis
        t += np.random.normal(0, 0.5)
        h += np.random.normal(0, 1.5)
        n += np.random.normal(0, 0.3)
        t = round(np.clip(t, 24.0, 29.9), 2)
        h = round(np.clip(h, 73.0, 87.0), 2)
        n = round(np.clip(n, 0.1, 9.9), 2)
        synthetic_rows.append({
            "recorded_at": gen_timestamp(i),
            "temperature_c": t,
            "humidity_rh": h,
            "nh3_ppm": n,
            "source": "synthetic_bagus",
        })

# --- SEDANG: boundary conditions ---
if need_sedang > 0:
    print(f"  Generating {need_sedang} 'sedang' samples...")
    for i in range(need_sedang):
        case = np.random.choice(["cool_dry", "warm_humid", "warm_dry"])
        if case == "cool_dry":
            t = np.random.uniform(26.0, 29.9)
            h = np.random.uniform(62.0, 74.9)
            n = np.random.uniform(1.0, 8.0)
        elif case == "warm_humid":
            t = np.random.uniform(30.0, 32.0)
            h = np.random.uniform(75.0, 85.0)
            n = np.random.uniform(1.0, 8.0)
        else:
            t = np.random.uniform(30.0, 32.0)
            h = np.random.uniform(65.0, 74.9)
            n = np.random.uniform(1.0, 9.0)
        t = round(t + np.random.normal(0, 0.3), 2)
        h = round(h + np.random.normal(0, 1.0), 2)
        n = round(np.clip(n + np.random.normal(0, 0.2), 0.1, 15), 2)
        synthetic_rows.append({
            "recorded_at": gen_timestamp(need_bagus + i),
            "temperature_c": t,
            "humidity_rh": h,
            "nh3_ppm": n,
            "source": "synthetic_sedang",
        })

# --- BURUK: extreme conditions (sensor masih masuk akal) ---
if need_buruk > 0:
    print(f"  Generating {need_buruk} 'buruk' samples...")
    for i in range(need_buruk):
        case = np.random.choice(["hot_dry", "high_nh3", "very_hot", "hot_moderate"])
        if case == "hot_dry":
            t = np.random.uniform(32.1, 34.0)
            h = np.random.uniform(60.0, 71.9)
            n = np.random.uniform(0.5, 8.0)
        elif case == "high_nh3":
            t = np.random.uniform(28.0, 33.0)
            h = np.random.uniform(65.0, 80.0)
            n = np.random.uniform(26.0, 40.0)
        elif case == "very_hot":
            t = np.random.uniform(34.0, 38.0)
            h = np.random.uniform(55.0, 75.0)
            n = np.random.uniform(3.0, 15.0)
        else:
            t = np.random.uniform(33.1, 35.0)
            h = np.random.uniform(72.0, 80.0)
            n = np.random.uniform(2.0, 10.0)
        t = round(t + np.random.normal(0, 0.3), 2)
        h = round(h + np.random.normal(0, 1.0), 2)
        n = round(np.clip(n + np.random.normal(0, 0.5), 0.1, 50), 2)
        synthetic_rows.append({
            "recorded_at": gen_timestamp(need_bagus + need_sedang + i),
            "temperature_c": t,
            "humidity_rh": h,
            "nh3_ppm": n,
            "source": "synthetic_buruk",
        })


# ============================================================
# 4. MERGE & SAVE
# ============================================================
print("\n[4] Merging with real data...")

df_syn = pd.DataFrame(synthetic_rows)
real["source"] = "real"

# Verify synthetic labels
df_syn["grade_check"] = df_syn.apply(
    lambda r: label_grade(r["temperature_c"], r["humidity_rh"], r["nh3_ppm"]),
    axis=1,
)
print(f"  Synthetic grade check:")
for g in ["bagus", "sedang", "buruk"]:
    v = (df_syn["grade_check"] == g).sum()
    print(f"    {g}: {v}")

# Combine
final_cols = ["recorded_at", "temperature_c", "humidity_rh", "nh3_ppm"]
df_final = pd.concat([
    real[final_cols],
    df_syn[final_cols],
], ignore_index=True)

# Shuffle
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

# Final grade distribution
df_final["grade"] = df_final.apply(
    lambda r: label_grade(r["temperature_c"], r["humidity_rh"], r["nh3_ppm"]),
    axis=1,
)
total = len(df_final)
vc_final = df_final["grade"].value_counts()

print(f"\n  FINAL DATASET:")
print(f"    Real data    : {n_real} rows")
print(f"    Synthetic    : {len(df_syn)} rows ({len(df_syn)/total*100:.1f}%)")
print(f"    Total        : {total} rows")
print(f"\n  Grade distribution:")
for g in ["bagus", "sedang", "buruk"]:
    v = vc_final.get(g, 0)
    print(f"    {g}: {v:6d} ({v/total*100:.1f}%)")

# Save
out_training = os.path.join(OUTPUT_DIR, "ml_training_smart.csv")
df_final[final_cols].to_csv(out_training, index=False)
print(f"\n  ✓ Saved: {out_training}")

# Also save with source column for analysis
out_debug = os.path.join(OUTPUT_DIR, "ml_training_smart_debug.csv")
df_all_debug = pd.concat([
    real[final_cols + ["source"]],
    df_syn[final_cols + ["source"]],
], ignore_index=True)
df_all_debug.to_csv(out_debug, index=False)
print(f"  ✓ Saved: {out_debug} (with source column)")

print(f"""
{'='*60}
DONE!
{'='*60}
Upload '{os.path.basename(out_training)}' ke Kaggle.
Update DATA_PATH di semua training script:
  DATA_PATH = "/kaggle/input/<dataset-name>/ml_training_smart.csv"

Synthetic hanya {len(df_syn)/total*100:.1f}% dari total — pelengkap, bukan pengganti.
{'='*60}
""")

# ============================================================
# Swiftlet Synthetic Data Generator
# ============================================================
# Generates realistic sensor data based on real-world patterns
# from Lt1 + Lt2 sensor data, with added variety for ML training.
#
# Features:
#   - Diurnal patterns (siang panas, malam sejuk)
#   - Seasonal variation (musim hujan vs kemarau)
#   - Temp-RH negative correlation (real: -0.81)
#   - Multiple building profiles (ideal, hot, mediocre)
#   - Realistic NH3 patterns (spike events)
#   - Sensor noise
#
# Output: data/synthetic_expanded.csv
# ============================================================

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

# ----- CONFIG -----
OUTPUT_PATH = "data/synthetic_expanded.csv"
TOTAL_DAYS = 90           # 3 bulan data
READINGS_PER_HOUR = 1     # 1 reading per jam
TOTAL_READINGS = TOTAL_DAYS * 24 * READINGS_PER_HOUR
START_DATE = datetime(2025, 6, 1)  # Mulai musim kemarau

print("=" * 60)
print("SWIFTLET SYNTHETIC DATA GENERATOR")
print("=" * 60)
print(f"  Total days    : {TOTAL_DAYS}")
print(f"  Total readings: {TOTAL_READINGS}")
print(f"  Start date    : {START_DATE}")

# ============================================================
# 1. BUILDING PROFILES
# ============================================================
# Simulasi beberapa gedung walet dengan kondisi berbeda

PROFILES = {
    "ideal": {
        # Gedung ideal: insulasi bagus, ventilasi optimal
        "temp_base": 27.5,
        "temp_amplitude": 1.5,   # variasi diurnal kecil
        "rh_base": 80.0,
        "rh_amplitude": 3.0,
        "nh3_base": 2.0,
        "nh3_spike_prob": 0.02,
        "weight": 0.25,  # 25% data
    },
    "mediocre": {
        # Gedung sedang: insulasi cukup
        "temp_base": 30.5,
        "temp_amplitude": 2.0,
        "rh_base": 73.0,
        "rh_amplitude": 4.0,
        "nh3_base": 5.0,
        "nh3_spike_prob": 0.05,
        "weight": 0.35,  # 35% data
    },
    "hot": {
        # Gedung panas: mirip data real Lt1/Lt2
        "temp_base": 32.5,
        "temp_amplitude": 1.2,
        "rh_base": 70.0,
        "rh_amplitude": 3.5,
        "nh3_base": 0.5,
        "nh3_spike_prob": 0.03,
        "weight": 0.25,  # 25% data
    },
    "poor": {
        # Gedung buruk: panas, kering, NH3 tinggi
        "temp_base": 34.0,
        "temp_amplitude": 2.5,
        "rh_base": 65.0,
        "rh_amplitude": 5.0,
        "nh3_base": 12.0,
        "nh3_spike_prob": 0.08,
        "weight": 0.15,  # 15% data
    },
}


def generate_building(profile_name, profile, n_readings, start_dt):
    """Generate time-series sensor data for one building profile."""

    timestamps = [start_dt + timedelta(hours=i) for i in range(n_readings)]
    hours = np.array([t.hour for t in timestamps])
    days = np.array([(t - start_dt).days for t in timestamps])

    # --- Temperature ---
    # Base + diurnal (peak at 14:00) + seasonal + noise
    diurnal = profile["temp_amplitude"] * np.sin(2 * np.pi * (hours - 6) / 24)

    # Seasonal: kemarau (bulan 6-9) lebih panas, hujan (10-2) lebih sejuk
    seasonal = 1.5 * np.sin(2 * np.pi * days / 365)

    # Random daily variation
    daily_noise = np.repeat(
        np.random.normal(0, 0.8, TOTAL_DAYS),
        24 * READINGS_PER_HOUR
    )[:n_readings]

    # Sensor noise
    sensor_noise = np.random.normal(0, 0.3, n_readings)

    temperature = profile["temp_base"] + diurnal + seasonal + daily_noise + sensor_noise
    temperature = np.clip(temperature, 20.0, 42.0).round(2)

    # --- Humidity ---
    # Inversely correlated with temperature (real correlation: -0.81)
    rh_from_temp = -0.8 * (temperature - profile["temp_base"])  # negative correlation
    rh_diurnal = profile["rh_amplitude"] * np.sin(2 * np.pi * (hours - 18) / 24)  # peak at night
    rh_seasonal = -2.0 * np.sin(2 * np.pi * days / 365)  # kemarau lebih kering
    rh_noise = np.random.normal(0, 1.5, n_readings)

    # Rain events (random days with high humidity)
    rain_days = np.random.choice(TOTAL_DAYS, size=int(TOTAL_DAYS * 0.15), replace=False)
    rain_boost = np.zeros(n_readings)
    for rd in rain_days:
        start_idx = rd * 24 * READINGS_PER_HOUR
        end_idx = min(start_idx + np.random.randint(4, 12) * READINGS_PER_HOUR, n_readings)
        rain_boost[start_idx:end_idx] = np.random.uniform(5, 15)

    humidity = profile["rh_base"] + rh_from_temp + rh_diurnal + rh_seasonal + rh_noise + rain_boost
    humidity = np.clip(humidity, 45.0, 98.0).round(2)

    # --- NH3 ---
    # Base + occasional spikes + slow drift
    nh3_base = np.full(n_readings, profile["nh3_base"])
    nh3_drift = 2.0 * np.sin(2 * np.pi * days / 30)  # monthly cycle (cleaning)
    nh3_noise = np.random.exponential(0.5, n_readings)  # right-skewed noise

    # Spike events (poor ventilation, bird droppings accumulation)
    spikes = np.random.random(n_readings) < profile["nh3_spike_prob"]
    nh3_spikes = np.zeros(n_readings)
    nh3_spikes[spikes] = np.random.uniform(10, 35, spikes.sum())

    # Decay spike over next few readings
    for i in range(1, n_readings):
        if nh3_spikes[i-1] > 5 and nh3_spikes[i] == 0:
            nh3_spikes[i] = nh3_spikes[i-1] * 0.7  # decay

    ammonia = nh3_base + nh3_drift + nh3_noise + nh3_spikes
    ammonia = np.clip(ammonia, 0.0, 60.0).round(2)

    return pd.DataFrame({
        "recorded_at": [t.strftime("%Y-%m-%dT%H:%M:%SZ") for t in timestamps],
        "temperature_c": temperature,
        "humidity_rh": humidity,
        "nh3_ppm": ammonia,
        "source": profile_name,
    })


# ============================================================
# 2. GENERATE DATA
# ============================================================
print("\n[1] Generating data per building profile...")

all_dfs = []
for name, profile in PROFILES.items():
    n = int(TOTAL_READINGS * profile["weight"])
    df = generate_building(name, profile, n, START_DATE)
    all_dfs.append(df)

    # Quick stats
    print(f"  {name:10s}: {len(df):5d} rows | "
          f"temp={df.temperature_c.mean():.1f}±{df.temperature_c.std():.1f} | "
          f"rh={df.humidity_rh.mean():.1f}±{df.humidity_rh.std():.1f} | "
          f"nh3={df.nh3_ppm.mean():.1f}±{df.nh3_ppm.std():.1f}")

# ============================================================
# 3. COMBINE & SHUFFLE
# ============================================================
print("\n[2] Combining and shuffling...")

df_all = pd.concat(all_dfs, ignore_index=True)
df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"  Total rows: {len(df_all)}")
print(f"  Date range: {df_all.recorded_at.min()} → {df_all.recorded_at.max()}")

# ============================================================
# 4. SIMULATE GRADE DISTRIBUTION
# ============================================================
print("\n[3] Simulating grade distribution (with new stricter rules)...")

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

df_all["sim_grade"] = df_all.apply(
    lambda r: label_grade(r.temperature_c, r.humidity_rh, r.nh3_ppm), axis=1
)
vc = df_all["sim_grade"].value_counts()
total = len(df_all)
for g in ["bagus", "sedang", "buruk"]:
    v = vc.get(g, 0)
    print(f"  {g}: {v:5d} ({v/total*100:.1f}%)")

# Drop sim_grade (labeling dilakukan di training script)
df_all = df_all.drop(columns=["sim_grade"])

# ============================================================
# 5. SAVE
# ============================================================
print(f"\n[4] Saving to {OUTPUT_PATH}...")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
# Save without source column for Kaggle compatibility
df_export = df_all[["recorded_at", "temperature_c", "humidity_rh", "nh3_ppm"]].copy()
df_export.to_csv(OUTPUT_PATH, index=False)

# Also save with source for analysis
df_all.to_csv(OUTPUT_PATH.replace(".csv", "_with_source.csv"), index=False)

print(f"  ✓ {OUTPUT_PATH} ({len(df_export)} rows)")
print(f"  ✓ {OUTPUT_PATH.replace('.csv', '_with_source.csv')} (with source column)")

# ============================================================
# 6. SUMMARY
# ============================================================
print(f"""
{'='*60}
GENERATION COMPLETE
{'='*60}
Total : {len(df_export)} rows ({TOTAL_DAYS} days)
Suhu  : {df_export.temperature_c.mean():.1f}°C (range: {df_export.temperature_c.min():.1f}-{df_export.temperature_c.max():.1f})
RH    : {df_export.humidity_rh.mean():.1f}% (range: {df_export.humidity_rh.min():.1f}-{df_export.humidity_rh.max():.1f})
NH3   : {df_export.nh3_ppm.mean():.1f}ppm (range: {df_export.nh3_ppm.min():.1f}-{df_export.nh3_ppm.max():.1f})

Building profiles:
  ideal     (25%): suhu ~27.5°C, RH ~80% → mostly "bagus"
  mediocre  (35%): suhu ~30.5°C, RH ~73% → mostly "sedang"
  hot       (25%): suhu ~32.5°C, RH ~70% → "sedang"/"buruk" (mirip data real)
  poor      (15%): suhu ~34°C, RH ~65%   → mostly "buruk"

Next:
  1. Merge dengan real data:
     python -c "import pandas as pd; \\
       real=pd.read_csv('data/ml_training_merged.csv'); \\
       syn=pd.read_csv('data/synthetic_expanded.csv'); \\
       pd.concat([real,syn]).to_csv('data/ml_training_final.csv',index=False); \\
       print(f'Final: {{len(real)+len(syn)}} rows')"
  2. Upload ml_training_final.csv ke Kaggle
  3. Update DATA_PATH di kaggle_train_optuna.py
""")

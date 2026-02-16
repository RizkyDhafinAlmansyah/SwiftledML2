# 00_fetch.py
import os
import datetime as dt
from pathlib import Path

import requests
import pandas as pd
from dotenv import load_dotenv

# ==========================
# 0. Load ENV TRAIN
# ==========================
BASE_DIR = Path(__file__).resolve().parent
ENV_TRAIN = BASE_DIR / ".env.train"

if not ENV_TRAIN.exists():
    raise SystemExit(f".env.train tidak ditemukan di {ENV_TRAIN}")

load_dotenv(ENV_TRAIN)

BASE_URL = os.getenv("BASE_URL", "").rstrip("/")
RAW_TOKEN = os.getenv("FARMER_TOKEN", "").strip()

if not BASE_URL:
    raise SystemExit("BASE_URL kosong di .env.train")
if not RAW_TOKEN:
    raise SystemExit("FARMER_TOKEN kosong di .env.train")

# Pastikan format Authorization: Bearer <token>
if RAW_TOKEN.lower().startswith("bearer "):
    AUTH_HEADER = RAW_TOKEN
else:
    AUTH_HEADER = f"Bearer {RAW_TOKEN}"

HEADERS = {
    "Authorization": AUTH_HEADER,
    "Accept": "application/json"
}

TEMP_SENSOR_ID = os.getenv("TEMP_SENSOR_ID")
HUMID_SENSOR_ID = os.getenv("HUMID_SENSOR_ID")
NH3_SENSOR_ID = os.getenv("NH3_SENSOR_ID")

RBW_ID = os.getenv("RBW_ID")
NODE_ID_NEST = os.getenv("NODE_ID_NEST") or os.getenv("NODE_ID")  # fallback

for name, val in [
    ("TEMP_SENSOR_ID", TEMP_SENSOR_ID),
    ("HUMID_SENSOR_ID", HUMID_SENSOR_ID),
    ("NH3_SENSOR_ID", NH3_SENSOR_ID),
    ("RBW_ID", RBW_ID),
    ("NODE_ID_NEST", NODE_ID_NEST),
]:
    if not val:
        raise SystemExit(f"ENV {name} belum diisi di .env.train")

# ==========================
# 1. Config fetch
# ==========================
# ambil N menit terakhir (kalau mau 10.000+ baris bisa naikin ke 6 jam, 12 jam, dst)
WINDOW_MINUTES = 60   # 1 jam terakhir
PER_PAGE       = 200  # jumlah data per halaman (limit=)
PAGE_LIMIT     = 4   # maksimal halaman yang di-scan per sensor


def readings_url(sensor_id: str) -> str:
    return f"{BASE_URL}/api/v1/sensors/{sensor_id}/readings"


# ==========================
# 2. Fungsi fetch per sensor
# ==========================
def fetch_readings(sensor_id: str, label: str) -> pd.DataFrame:
    """Ambil data sensor tertentu dalam window waktu terakhir."""
    end_dt = dt.datetime.utcnow()
    start_dt = end_dt - dt.timedelta(minutes=WINDOW_MINUTES)
    print(f"[INFO] Fetch window: {start_dt.isoformat()} .. {end_dt.isoformat()}")
    print(f"[INFO] Sensor {label}: id={sensor_id}")

    all_rows = []
    page = 1

    while page <= PAGE_LIMIT:
        params = {
            "limit": PER_PAGE,
            "page": page,
            # NOTE:
            # kalau backend nanti support filter waktu, bisa ditambah:
            # "start": start_dt.isoformat() + "Z",
            # "end": end_dt.isoformat() + "Z",
        }

        print(f"[{label}] GET page={page} limit={PER_PAGE} ...", end="", flush=True)
        try:
            resp = requests.get(
                readings_url(sensor_id),
                headers=HEADERS,
                params=params,
                timeout=30,
            )
        except Exception as e:
            print(f" -> request error: {e}")
            break

        status = resp.status_code
        if status == 401:
            print(" -> 401 UNAUTHORIZED (token salah/expired). STOP.")
            print(resp.text)
            break
        if status == 403:
            print(" -> 403 FORBIDDEN (Access denied). STOP.")
            print(resp.text)
            break
        if status == 404:
            print(" -> 404 NOT FOUND (sensor_id salah?). STOP.")
            print(resp.text)
            break
        if status != 200:
            print(f" -> {status} {resp.text[:150]}")
            break

        payload = resp.json()
        data = payload.get("data", payload)
        print(f" -> OK rows={len(data)}")

        if not data:
            break

        for d in data:
            all_rows.append(
                {
                    "recorded_at": d.get("recorded_at"),
                    "value": d.get("value"),
                    "node_id": d.get("node_id"),
                    "rbw_id": RBW_ID,
                }
            )

        # kalau rows < PER_PAGE, berarti halaman terakhir
        if len(data) < PER_PAGE:
            break

        page += 1

    df = pd.DataFrame(all_rows)
    print(f"[OK] {label} rows: {len(df)}")
    return df


# ==========================
# 3. Join tiga sensor & agregasi per menit
# ==========================
def main():
    print("=== Fetch temperature ===")
    df_temp = fetch_readings(TEMP_SENSOR_ID, "TEMP")

    print("\n=== Fetch humidity ===")
    df_hum = fetch_readings(HUMID_SENSOR_ID, "HUMID")

    print("\n=== Fetch ammonia ===")
    df_nh3 = fetch_readings(NH3_SENSOR_ID, "NH3")

    # rename kolom value per sensor
    if not df_temp.empty:
        df_temp = df_temp.rename(columns={"value": "temperature_c"})
    if not df_hum.empty:
        df_hum = df_hum.rename(columns={"value": "humidity_rh"})
    if not df_nh3.empty:
        df_nh3 = df_nh3.rename(columns={"value": "nh3_ppm"})

    # merge outer di recorded_at + node_id + rbw_id
    df_merged = None
    for df_part in [df_temp, df_hum, df_nh3]:
        if df_part is None or df_part.empty:
            continue
        if df_merged is None:
            df_merged = df_part
        else:
            df_merged = pd.merge(
                df_merged,
                df_part,
                on=["recorded_at", "node_id", "rbw_id"],
                how="outer",
            )

    if df_merged is None or df_merged.empty:
        print("[WARN] Tidak ada data gabungan, CSV tidak dibuat.")
        return

    # urutkan waktu & reset index
    df_merged = df_merged.sort_values("recorded_at").reset_index(drop=True)

    # ========== 3A. Simpan RAW (opsional) ==========
    out_dir = BASE_DIR / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    raw_path = out_dir / f"sensor_joined_RAW_{stamp}.csv"
    df_merged.to_csv(raw_path, index=False)
    print(f"[DONE] RAW merged CSV -> {raw_path}")
    print("RAW shape :", df_merged.shape)

    # ========== 3B. Agregasi per menit ==========
    print("\n[INFO] Mulai agregasi per menit (mean per minute) ...")

    # 1) ubah recorded_at ke datetime
    df_merged["recorded_at"] = pd.to_datetime(df_merged["recorded_at"])

    # 2) floor ke menit (kalau mau 5 menit: .dt.floor('5T'))
    df_merged["recorded_minute"] = df_merged["recorded_at"].dt.floor("T")

    # 3) agregasi: rata-rata per menit
    df_agg = (
        df_merged
        .groupby("recorded_minute")
        .agg({
            "temperature_c": "mean",
            "humidity_rh":   "mean",
            "nh3_ppm":       "mean",
            "rbw_id":        "first",
            "node_id":       "first",
        })
        .reset_index()
        .rename(columns={"recorded_minute": "recorded_at"})
    )

    # sort lagi
    df_agg = df_agg.sort_values("recorded_at").reset_index(drop=True)

    # optional: kalau mau batasi max rows (misal ambil 50.000 terakhir saja)
    # MAX_ROWS = 50000
    # if len(df_agg) > MAX_ROWS:
    #     df_agg = df_agg.tail(MAX_ROWS).reset_index(drop=True)

    agg_path = out_dir / f"sensor_joined_AGG_{stamp}.csv"
    df_agg.to_csv(agg_path, index=False)

    print(f"[DONE] AGG per-minute CSV -> {agg_path}")
    print("AGG shape:", df_agg.shape)
    print(df_agg.head())


if __name__ == "__main__":
    main()

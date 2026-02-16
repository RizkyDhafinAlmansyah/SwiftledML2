import os
import json
import time
import pickle
from uuid import uuid4
from typing import Optional, Dict, Any
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Import buffer manager for sliding window features
from buffer_manager import (
    get_buffer_manager,
    create_features_from_stats,
    RollingStats,
)

# ============================================================
# 0) ENV & BASIC CONFIG
# ============================================================

BASE_DIR = os.path.dirname(__file__)
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH, override=True)

ART_DIR = os.getenv("ART_DIR", BASE_DIR)

# --- Artefak ML v2 (sesuai dokumen spesifikasi) ---
PATH_GRADE_MODEL_V2   = os.path.join(ART_DIR, "model_grade_panen_v2.pkl")
PATH_GRADE_ENCODER_V2 = os.path.join(ART_DIR, "label_encoder_grade_v2.pkl")
PATH_ANOM_MODEL_V2    = os.path.join(ART_DIR, "model_anomaly_detection_v2.pkl")
PATH_ANOM_SCALER_V2   = os.path.join(ART_DIR, "scaler_anomaly_v2.pkl")
PATH_PUMP_STATE_V2    = os.path.join(ART_DIR, "model_pump_state_v2.pkl")
PATH_PUMP_DURATION_V2 = os.path.join(ART_DIR, "model_pump_duration_v2.pkl")

PATH_CTRL = os.path.join(ART_DIR, "control_config.json")
PATH_DATA = os.path.join(ART_DIR, "sensor_cleaned.csv")  # optional, buat median default

DEFAULT_MIN_ON   = float(os.getenv("MIN_ON",   20))
DEFAULT_MIN_OFF  = float(os.getenv("MIN_OFF",  40))
DEFAULT_COOLDOWN = float(os.getenv("COOLDOWN", 60))


def _filesize(p: str) -> int:
    try:
        return os.path.getsize(p)
    except Exception:
        return 0


def _safe_load_pickle(p: str):
    """Load artefak .pkl (joblib dulu, lalu pickle)."""
    if not os.path.exists(p):
        raise FileNotFoundError(f"File not found: {p}")
    try:
        import joblib
        return joblib.load(p)
    except Exception as e_joblib:
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception as e_pickle:
            raise RuntimeError(
                f"Gagal load artefak '{p}'. "
                f"joblib.load error: {e_joblib} | pickle.load error: {e_pickle}. "
                f"Pastikan di-export dengan joblib.dump(obj, '{os.path.basename(p)}', compress=3)."
            )

# ============================================================
# 1) LOAD MODEL V2
# ============================================================

print(f"[ART] grade_v2        ={PATH_GRADE_MODEL_V2} size={_filesize(PATH_GRADE_MODEL_V2)}B")
print(f"[ART] grade_encoder_v2={PATH_GRADE_ENCODER_V2} size={_filesize(PATH_GRADE_ENCODER_V2)}B")
print(f"[ART] anom_v2         ={PATH_ANOM_MODEL_V2} size={_filesize(PATH_ANOM_MODEL_V2)}B")
print(f"[ART] anom_scaler_v2  ={PATH_ANOM_SCALER_V2} size={_filesize(PATH_ANOM_SCALER_V2)}B")
print(f"[ART] pump_state_v2   ={PATH_PUMP_STATE_V2} size={_filesize(PATH_PUMP_STATE_V2)}B")
print(f"[ART] pump_duration_v2={PATH_PUMP_DURATION_V2} size={_filesize(PATH_PUMP_DURATION_V2)}B")
print(f"[ART] ctrl_cfg        ={PATH_CTRL} size={_filesize(PATH_CTRL)}B")

grade_model_v2 = None
grade_encoder_v2 = None
anomaly_model_v2 = None
anomaly_scaler_v2 = None
pump_state_model_v2 = None
pump_duration_model_v2 = None

try:
    if os.path.exists(PATH_GRADE_MODEL_V2):
        grade_model_v2 = _safe_load_pickle(PATH_GRADE_MODEL_V2)
        print("[BOOT] Loaded model_grade_panen_v2.pkl OK")
    else:
        print("[BOOT] model_grade_panen_v2.pkl tidak ditemukan")
except Exception as e:
    print(f"[WARN] gagal load model_grade_panen_v2.pkl: {e}")

try:
    if os.path.exists(PATH_GRADE_ENCODER_V2):
        grade_encoder_v2 = _safe_load_pickle(PATH_GRADE_ENCODER_V2)
        print("[BOOT] Loaded label_encoder_grade_v2.pkl OK")
    else:
        print("[BOOT] label_encoder_grade_v2.pkl tidak ditemukan")
except Exception as e:
    print(f"[WARN] gagal load label_encoder_grade_v2.pkl: {e}")

try:
    if os.path.exists(PATH_ANOM_MODEL_V2):
        anomaly_model_v2 = _safe_load_pickle(PATH_ANOM_MODEL_V2)
        print("[BOOT] Loaded model_anomaly_detection_v2.pkl OK")
    else:
        print("[BOOT] model_anomaly_detection_v2.pkl tidak ditemukan")
except Exception as e:
    print(f"[WARN] gagal load model_anomaly_detection_v2.pkl: {e}")

try:
    if os.path.exists(PATH_ANOM_SCALER_V2):
        anomaly_scaler_v2 = _safe_load_pickle(PATH_ANOM_SCALER_V2)
        print("[BOOT] Loaded scaler_anomaly_v2.pkl OK")
    else:
        print("[BOOT] scaler_anomaly_v2.pkl tidak ditemukan")
except Exception as e:
    print(f"[WARN] gagal load scaler_anomaly_v2.pkl: {e}")

try:
    if os.path.exists(PATH_PUMP_STATE_V2):
        pump_state_model_v2 = _safe_load_pickle(PATH_PUMP_STATE_V2)
        print("[BOOT] Loaded model_pump_state_v2.pkl OK (belum di-wire ke endpoint)")
except Exception as e:
    print(f"[WARN] gagal load model_pump_state_v2.pkl: {e}")

try:
    if os.path.exists(PATH_PUMP_DURATION_V2):
        pump_duration_model_v2 = _safe_load_pickle(PATH_PUMP_DURATION_V2)
        print("[BOOT] Loaded model_pump_duration_v2.pkl OK (belum di-wire ke endpoint)")
except Exception as e:
    print(f"[WARN] gagal load model_pump_duration_v2.pkl: {e}")

# ============================================================
# 2) CONTROL CONFIG (HYSTERESIS PUMP)
# ============================================================

if os.path.exists(PATH_CTRL):
    with open(PATH_CTRL, "r", encoding="utf-8") as f:
        ctrl = json.load(f)
    print("[BOOT] Loaded control_config.json:", ctrl)
else:
    ctrl = {"temp_on": 30.5, "temp_off": 29.5, "rh_on": 83.0, "rh_off": 75.0}
    print("[BOOT] PATH_CTRL tidak ada, pakai default:", ctrl)

# ============================================================
# 3) DEFAULT MEDIAN DATASET (OPTIONAL)
# ============================================================

TEMP_COL = "temperature_c"
RH_COL   = "humidity_rh"
NH3_COL  = "nh3_ppm"

feature_defaults: Dict[str, float] = {
    TEMP_COL: 28.0,
    RH_COL: 80.0,
    NH3_COL: 5.0,
}

if os.path.exists(PATH_DATA):
    try:
        df_default = pd.read_csv(PATH_DATA)
        df_default.columns = [c.strip().lower() for c in df_default.columns]
        for c in [TEMP_COL, RH_COL, NH3_COL]:
            if c in df_default.columns:
                df_default[c] = pd.to_numeric(df_default[c], errors="coerce")
        med = df_default[[TEMP_COL, RH_COL, NH3_COL]].median(numeric_only=True)
        for c in [TEMP_COL, RH_COL, NH3_COL]:
            if c in med:
                feature_defaults[c] = float(np.nan_to_num(med[c], nan=feature_defaults[c]))
        print("[BOOT] feature_defaults dari sensor_cleaned.csv:", feature_defaults)
    except Exception as e:
        print(f"[WARN] Gagal menghitung median dari {PATH_DATA}: {e}")
else:
    print("[INFO] PATH_DATA tidak ada, pakai feature_defaults hard-coded:", feature_defaults)

print(f"[INFO] TEMP_COL={TEMP_COL}, RH_COL={RH_COL}, NH3_COL={NH3_COL}")

# ============================================================
# 4) FEATURE ENGINEERING V2
# ============================================================

FEATURES_ANOMALY = [
    "temperature", "humidity", "ammonia",
    "hour_of_day", "temp_delta_1h", "humid_delta_1h",
    "nh3_delta_1h", "comfort_index",
]

FEATURES_GRADE = [
    "temperature", "humidity", "ammonia",
    "temp_avg_1h", "humid_avg_1h", "nh3_avg_1h",
    "comfort_index", "is_daytime",
]

FEATURES_PUMP_STATE = [
    "temperature", "humidity", "ammonia",
    "temp_avg_1h", "humid_avg_1h", "humid_delta_1h",
    "hour_of_day",
]

FEATURES_PUMP_DURATION = [
    "temperature", "humidity", "ammonia",
    "temp_avg_1h", "humid_avg_1h", "humid_delta_1h",
    "hour_of_day", "comfort_index",
]


def calculate_comfort_index(temperature: float, humidity: float, ammonia: float) -> float:
    """Rumus sama dengan dokumen ML Spec."""
    temp_optimal = 28.0
    humid_optimal = 80.0
    nh3_max = 20.0

    temp_score = 1 - abs(temperature - temp_optimal) / 15.0
    humid_score = 1 - abs(humidity - humid_optimal) / 35.0
    nh3_score = 1 - (ammonia / nh3_max)

    temp_score = max(0.0, min(1.0, temp_score))
    humid_score = max(0.0, min(1.0, humid_score))
    nh3_score = max(0.0, min(1.0, nh3_score))

    comfort_index = (temp_score * 0.35 + humid_score * 0.35 + nh3_score * 0.30) * 100.0
    return float(round(comfort_index, 2))


def build_base_features(
    temp_c: Optional[float],
    rh: Optional[float],
    nh3_ppm: Optional[float],
    recorded_ts: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Bangun 1 dict fitur lengkap (sensor + turunan) dari suhu/RH/NH3 + waktu.
    extra boleh berisi delta/avg kalau memang ada (dari BE).
    """
    extra = extra or {}

    t = float(temp_c) if temp_c is not None else feature_defaults[TEMP_COL]
    h = float(rh) if rh is not None else feature_defaults[RH_COL]
    n = float(nh3_ppm) if nh3_ppm is not None else feature_defaults[NH3_COL]

    if recorded_ts is not None:
        dt = datetime.fromtimestamp(float(recorded_ts), tz=timezone.utc)
    else:
        dt = datetime.now(timezone.utc)

    hour = int(dt.hour)
    is_daytime = 1 if 6 <= hour < 18 else 0

    feat: Dict[str, float] = {
        "temperature": t,
        "humidity": h,
        "ammonia": n,
        "hour_of_day": hour,
        "is_daytime": is_daytime,
        "temp_avg_1h": float(extra.get("temp_avg_1h", t)),
        "humid_avg_1h": float(extra.get("humid_avg_1h", h)),
        "nh3_avg_1h": float(extra.get("nh3_avg_1h", n)),
        "temp_delta_1h": float(extra.get("temp_delta_1h", 0.0)),
        "humid_delta_1h": float(extra.get("humid_delta_1h", 0.0)),
        "nh3_delta_1h": float(extra.get("nh3_delta_1h", 0.0)),
    }

    feat["comfort_index"] = calculate_comfort_index(t, h, n)
    return feat

# ============================================================
# 5) PUMP HYSTERESIS STATE
# ============================================================

class SprayState:
    def __init__(self):
        self.is_on = False
        self.last_change_ts = 0.0
        self.last_off_ts = 0.0
        self.cooldown = DEFAULT_COOLDOWN
        self.min_on   = DEFAULT_MIN_ON
        self.min_off  = DEFAULT_MIN_OFF


STATES: Dict[str, SprayState] = {}


def get_state(node_id: str) -> SprayState:
    if node_id not in STATES:
        STATES[node_id] = SprayState()
    return STATES[node_id]


def decide_spray_with_state(
    state: SprayState,
    temp_c: Optional[float],
    rh: Optional[float],
    nh3_ppm: Optional[float],
    cfg_ctrl: Dict[str, float],
    now_ts: Optional[float] = None,
):
    """Rule-based hysteresis (baseline, belum pakai model_pump_state_v2)."""
    now = time.time() if now_ts is None else now_ts
    if temp_c is None or rh is None:
        return False, "sensor_invalid"

    t_on, t_off   = cfg_ctrl.get("temp_on", 30.5), cfg_ctrl.get("temp_off", 29.5)
    rh_on, rh_off = cfg_ctrl.get("rh_on", 83.0),  cfg_ctrl.get("rh_off", 75.0)

    # TODO: nanti bisa digabung dengan output model_pump_state_v2
    turn_on_cond  = (temp_c >= t_on) or (rh <= rh_on)
    turn_off_cond = (temp_c <= t_off) and (rh >= rh_off)

    elapsed = now - state.last_change_ts

    if state.is_on:
        if elapsed < state.min_on:
            return True, "hold_min_on"
        if turn_off_cond:
            state.is_on = False
            state.last_change_ts = now
            state.last_off_ts = now
            return False, "off_by_condition"
        return True, "stay_on"
    else:
        if (now - state.last_off_ts) < state.cooldown or elapsed < state.min_off:
            return False, "hold_cooldown_or_min_off"
        if turn_on_cond:
            state.is_on = True
            state.last_change_ts = now
            return True, "on_by_condition"
        return False, "stay_off"

# ============================================================
# 6) PREDICT GRADE & ANOMALY HELPERS
# ============================================================

def predict_grade_from_values(
    temp_c: Optional[float],
    rh: Optional[float],
    nh3_ppm: Optional[float],
    recorded_ts: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
):
    if grade_model_v2 is None or grade_encoder_v2 is None:
        return "unknown", None

    base_feat = build_base_features(temp_c, rh, nh3_ppm, recorded_ts, extra)
    X_row = pd.DataFrame([[base_feat[f] for f in FEATURES_GRADE]], columns=FEATURES_GRADE)

    y_enc_pred = int(grade_model_v2.predict(X_row)[0])
    grade_label = str(grade_encoder_v2.inverse_transform([y_enc_pred])[0])

    probs_dict = None
    if hasattr(grade_model_v2, "predict_proba"):
        proba = grade_model_v2.predict_proba(X_row)[0]
        probs_dict = {}
        for enc_class, p in zip(grade_model_v2.classes_, proba):
            cls_name = str(grade_encoder_v2.inverse_transform([enc_class])[0])
            probs_dict[cls_name] = float(round(p, 6))

    return grade_label, probs_dict


def _detect_anomaly(values: Dict[str, Optional[float]]):
    """
    values: {TEMP_COL: t, RH_COL: r, NH3_COL: n}
    Hybrid: rule-based + model_anomaly_detection_v2 (IsolationForest).
    """
    t = values.get(TEMP_COL)
    r = values.get(RH_COL)
    n = values.get(NH3_COL)

    flags: list[str] = []
    if n is not None and n >= 20.0:
        flags.append("NH3_HIGH")
    if r is not None and (r < 0 or r > 100):
        flags.append("RH_OUT_OF_RANGE")
    if t is not None and (t < 10 or t > 45):
        flags.append("TEMP_OUT_OF_RANGE")

    if anomaly_model_v2 is None or anomaly_scaler_v2 is None:
        verdict = "anomaly" if flags else "normal"
        return {
            "verdict": verdict,
            "iso_pred": -1 if verdict == "anomaly" else 1,
            "iso_score": 0.0,
            "flags": flags,
            "engine": "rule-only",
        }

    try:
        base_feat = build_base_features(t, r, n)
        X_anom = pd.DataFrame([[base_feat[f] for f in FEATURES_ANOMALY]], columns=FEATURES_ANOMALY)
        X_scaled = anomaly_scaler_v2.transform(X_anom)

        iso_pred = int(anomaly_model_v2.predict(X_scaled)[0])  # 1 normal, -1 anomaly
        iso_score = float(anomaly_model_v2.decision_function(X_scaled)[0])

        verdict = "anomaly" if (iso_pred == -1 or flags) else "normal"

        return {
            "verdict": verdict,
            "iso_pred": iso_pred,
            "iso_score": iso_score,
            "flags": flags,
            "engine": "ml+rule",
        }
    except Exception as e:
        print(f"[WARN] anomaly_v2 predict error: {e}")
        verdict = "anomaly" if flags else "normal"
        return {
            "verdict": verdict,
            "iso_pred": -1 if verdict == "anomaly" else 1,
            "iso_score": 0.0,
            "flags": flags,
            "engine": "rule-fallback",
        }


def epoch_to_iso8601(ts: float | int) -> str:
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat().replace("+00:00", "Z")


def iso_to_epoch(s: str) -> float:
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.timestamp()
    except Exception:
        return time.time()


def normalize_iso_score(raw_score: float) -> float:
    # mapping kasar (-∞..+∞) -> 0..1
    x = (raw_score + 0.5) / 1.0
    return float(max(0.0, min(1.0, x)))

# ============================================================
# 7) OPTIONAL DB (BOLEH DISET VIA ENV DATABASE_URL)
# ============================================================

from sqlalchemy import create_engine, Column, String, Boolean, Float, JSON, DateTime, text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql import func

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL, pool_pre_ping=True) if DATABASE_URL else None
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class Decision(Base):
    __tablename__ = "decisions"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    node_id = Column(String, index=True, nullable=False)
    recorded_at = Column(DateTime(timezone=True), server_default=func.now())

    temp_c   = Column(Float)
    rh       = Column(Float)
    nh3_ppm  = Column(Float)

    grade_pred = Column(String)
    p_bagus    = Column(Float)
    p_sedang   = Column(Float)
    p_buruk    = Column(Float)

    sprayer_on     = Column(Boolean)
    sprayer_reason = Column(String)

    anom_verdict   = Column(String)
    anom_iso_pred  = Column(Float)
    anom_iso_score = Column(Float)
    anom_flags     = Column(JSON)

    used_temp_on  = Column(Float)
    used_temp_off = Column(Float)
    used_rh_on    = Column(Float)
    used_rh_off   = Column(Float)


if engine:
    Base.metadata.create_all(engine)


def save_decision(row: dict):
    if not engine:
        return
    db = SessionLocal()
    try:
        db.add(Decision(**row))
        db.commit()
    finally:
        db.close()

# ============================================================
# 8) FASTAPI APP & SCHEMAS
# ============================================================

app = FastAPI(title="Swiftlet AI Engine", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Health ----------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "has_anomaly_model": bool(anomaly_model_v2 is not None),
        "has_grade_model": bool(grade_model_v2 is not None),
        "version": "2.0.0",
    }


@app.get("/health_db")
def health_db():
    if not engine:
        return {"db": "disabled", "hint": "Set env DATABASE_URL untuk mengaktifkan DB."}
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"db": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")


@app.get("/version")
def version():
    classes = list(grade_encoder_v2.classes_) if grade_encoder_v2 is not None else []
    model_classes = {int(i): cls for i, cls in enumerate(classes)}
    return {
        "service": "swiftlet-ai-engine",
        "version": "2.0.0",
        "model_classes": model_classes,
    }

# ---------- Legacy decide ----------
class DecideRequest(BaseModel):
    node_id: str
    temperature_c: Optional[float] = None
    humidity_rh:   Optional[float] = None
    nh3_ppm:       Optional[float] = None
    extra_features: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None
    request_id: Optional[str] = None


class DecideResponse(BaseModel):
    grade: str
    probabilities: Optional[Dict[str, float]] = None
    sprayer_on: bool
    sprayer_reason: str
    anomaly: Dict[str, Any]
    used_thresholds: Dict[str, float]
    note: str
    debug: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None


@app.post("/reset_state")
def reset_state(node_id: Optional[str] = Query(default=None)):
    if node_id:
        st = get_state(node_id)
        st.is_on = False
        st.last_change_ts = 0.0
        st.last_off_ts = 0.0
        return {"ok": True, "msg": f"Spray state reset for '{node_id}'"}
    for st in STATES.values():
        st.is_on = False
        st.last_change_ts = 0.0
        st.last_off_ts = 0.0
    return {"ok": True, "msg": "Spray state reset for ALL nodes"}


@app.get("/state")
def state(node_id: str = Query(...)):
    st = get_state(node_id)
    now = time.time()
    return {
        "node_id": node_id,
        "is_on": st.is_on,
        "elapsed_since_change_s": round(now - st.last_change_ts, 2),
        "min_on_s": st.min_on,
        "min_off_s": st.min_off,
        "cooldown_remaining_s": round(max(0.0, st.cooldown - (now - st.last_off_ts)), 2),
        "last_change_ts": st.last_change_ts,
        "last_off_ts": st.last_off_ts,
    }


@app.post("/decide", response_model=DecideResponse)
def decide(req: DecideRequest):
    t = req.temperature_c
    r = req.humidity_rh
    n = req.nh3_ppm

    grade, probs = predict_grade_from_values(t, r, n, recorded_ts=req.timestamp, extra=req.extra_features)

    st = get_state(req.node_id)
    on, reason = decide_spray_with_state(st, t, r, n, ctrl, now_ts=req.timestamp)

    anom = _detect_anomaly({TEMP_COL: t, RH_COL: r, NH3_COL: n})

    save_decision({
        "node_id": req.node_id,
        "temp_c": t,
        "rh": r,
        "nh3_ppm": n,
        "grade_pred": grade,
        "p_bagus": (probs or {}).get("bagus"),
        "p_sedang": (probs or {}).get("sedang"),
        "p_buruk":  (probs or {}).get("buruk"),
        "sprayer_on": on,
        "sprayer_reason": reason,
        "anom_verdict": anom.get("verdict"),
        "anom_iso_pred": anom.get("iso_pred"),
        "anom_iso_score": anom.get("iso_score"),
        "anom_flags": anom.get("flags"),
        "used_temp_on": ctrl.get("temp_on"),
        "used_temp_off": ctrl.get("temp_off"),
        "used_rh_on": ctrl.get("rh_on"),
        "used_rh_off": ctrl.get("rh_off"),
    })

    now_ts = time.time()
    debug = {
        "node_id": req.node_id,
        "is_on": st.is_on,
        "elapsed_since_change_s": round(now_ts - st.last_change_ts, 2),
        "min_on_s": st.min_on,
        "min_off_s": st.min_off,
        "cooldown_remaining_s": round(max(0.0, st.cooldown - (now_ts - st.last_off_ts)), 2),
        "last_change_ts": st.last_change_ts,
        "last_off_ts": st.last_off_ts,
    }

    return DecideResponse(
        grade=grade,
        probabilities=probs,
        sprayer_on=on,
        sprayer_reason=reason,
        anomaly=anom,
        used_thresholds={
            "temp_on": ctrl.get("temp_on"),
            "temp_off": ctrl.get("temp_off"),
            "rh_on":   ctrl.get("rh_on"),
            "rh_off":  ctrl.get("rh_off"),
        },
        note="Hysteresis rule-based aktif; prediksi grade & anomaly pakai model v2.",
        debug=debug,
        request_id=req.request_id,
    )

# ============================================================
# 9) v1 SPEC ENDPOINTS
# ============================================================

class AnomalyRequest(BaseModel):
    sensor_id: str
    sensor_type: str  # "temp" | "humid" | "ammonia"
    rbw_id: str
    node_id: str
    recorded_at: str   # ISO 8601
    value: float


class AnomalyResponse(BaseModel):
    is_anomaly: bool
    score: float
    reason: str
    confidence: Optional[float] = None
    detected_at: Optional[str] = None


class PredictGradeRequest(BaseModel):
    rbw_id: str
    floor_no: int
    node_id: Optional[str] = None
    nests_count: int
    weight_kg: float
    avg_temp_7days: Optional[float] = None
    avg_humid_7days: Optional[float] = None
    avg_ammonia_7days: Optional[float] = None
    days_since_last_harvest: Optional[int] = None


class PredictGradeResponse(BaseModel):
    predicted_grade: str
    confidence: float
    factors: Dict[str, Any]
    recommendation: Optional[str] = None
    predicted_at: Optional[str] = None


class PumpRecommendRequest(BaseModel):
    node_id: str
    rbw_id: str
    floor_no: Optional[int] = None
    current_temp: float
    current_humid: float
    current_ammonia: Optional[float] = None
    temp_trend_1hour: Optional[str] = None  # "rising"|"falling"|"stable"
    humid_trend_1hour: Optional[str] = None
    pump_currently_on: bool


class PumpRecommendResponse(BaseModel):
    action: str                   # "turn_on" | "turn_off" | "keep_current"
    reason: str
    confidence: float
    recommended_duration_minutes: Optional[int] = None
    expected_outcome: Optional[Dict[str, Any]] = None
    recommended_at: Optional[str] = None

# --- /v1/anomaly-detect ---
@app.post("/v1/anomaly-detect", response_model=AnomalyResponse)
def v1_anomaly_detect(req: AnomalyRequest):
    if req.sensor_type not in {"temp", "humid", "ammonia"}:
        raise HTTPException(status_code=400, detail="sensor_type must be one of: temp, humid, ammonia")

    t = r = n = None
    if req.sensor_type == "temp":
        t = req.value
    elif req.sensor_type == "humid":
        r = req.value
    else:
        n = req.value

    epoch_ts = iso_to_epoch(req.recorded_at)

    anom = _detect_anomaly({TEMP_COL: t, RH_COL: r, NH3_COL: n})
    raw_score = float(anom.get("iso_score", 0.0))
    score = normalize_iso_score(raw_score) if anom.get("iso_score") is not None else (
        0.9 if anom.get("verdict") == "anomaly" else 0.1
    )

    flags = anom.get("flags") or []
    reason_bits = []
    if flags:
        reason_bits.append(", ".join(flags))
    if t is not None:
        reason_bits.append(f"Temperature reading: {t}°C")
    if r is not None:
        reason_bits.append(f"Humidity reading: {r}%")
    if n is not None:
        reason_bits.append(f"Ammonia reading: {n} ppm")

    return AnomalyResponse(
        is_anomaly=(anom.get("verdict") == "anomaly"),
        score=round(score, 3),
        reason="; ".join(reason_bits) or "Model evaluation",
        confidence=round(max(0.0, min(1.0, score)), 3),
        detected_at=epoch_to_iso8601(epoch_ts),
    )

# --- /v1/predict-nest-grade ---
@app.post("/v1/predict-nest-grade", response_model=PredictGradeResponse)
def v1_predict_grade(req: PredictGradeRequest):
    t = req.avg_temp_7days
    h = req.avg_humid_7days
    n = req.avg_ammonia_7days

    grade, probs = predict_grade_from_values(t, h, n)

    conf = 0.0
    if probs:
        conf = float(max(probs.values())) if len(probs) > 0 else 0.0

    factors = {
        "weight_per_nest": {
            "value": round((req.weight_kg / req.nests_count), 4) if req.nests_count else None,
            "status": "optimal" if req.nests_count and (req.weight_kg / req.nests_count) >= 0.03 else (
                "good" if req.nests_count else "unknown"
            ),
            "impact": "positive" if grade.lower() in ["bagus", "good"] else "neutral",
        },
        "environmental_conditions": {
            "temp": "optimal" if (t is None or 28 <= t <= 30) else ("too_high" if (t and t > 30) else "too_low"),
            "humidity": "optimal" if (h is None or 70 <= h <= 80) else ("too_high" if (h and h > 80) else "too_low"),
            "ammonia": "normal" if (n is None or n <= 25) else ("elevated" if (n and n <= 50) else "high"),
            "impact": "positive" if grade.lower() in ["bagus", "good"] else "neutral",
        },
        "harvest_cycle": {
            "days": req.days_since_last_harvest,
            "status": "optimal" if (req.days_since_last_harvest is None or 80 <= req.days_since_last_harvest <= 100)
            else ("too_early" if (req.days_since_last_harvest and req.days_since_last_harvest < 80) else "too_late"),
            "impact": "positive" if grade.lower() in ["bagus", "good"] else "neutral",
        },
    }

    ts = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")

    return PredictGradeResponse(
        predicted_grade=grade,
        confidence=float(round(conf, 4)),
        factors=factors,
        recommendation="Pertahankan suhu 28–30°C, kelembaban 70–80%, dan amonia serendah mungkin.",
        predicted_at=ts,
    )

# --- /v1/recommend-pump-action ---
@app.post("/v1/recommend-pump-action", response_model=PumpRecommendResponse)
def v1_recommend_pump(req: PumpRecommendRequest):
    st = get_state(req.node_id)
    st.is_on = bool(req.pump_currently_on)

    on, reason = decide_spray_with_state(
        st,
        temp_c=float(req.current_temp),
        rh=float(req.current_humid),
        nh3_ppm=float(req.current_ammonia) if req.current_ammonia is not None else None,
        cfg_ctrl=ctrl,
        now_ts=None,
    )

    if on and not req.pump_currently_on:
        action = "turn_on"
    elif (not on) and req.pump_currently_on:
        action = "turn_off"
    else:
        action = "keep_current"

    duration_min = int(max(1, round(st.min_on / 60.0))) if action == "turn_on" else None

    target_rh = ctrl.get("rh_off", 75.0)
    delta = abs(target_rh - req.current_humid)
    conf = max(0.5, min(0.99, delta / 20.0 + 0.5))

    expected = {
        "target_humid": float(target_rh),
        "estimated_time_minutes": duration_min if duration_min else 0,
    } if action == "turn_on" else None

    ts = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")

    return PumpRecommendResponse(
        action=action,
        reason=reason if reason else "rule-based hysteresis decision",
        confidence=float(round(conf, 3)),
        recommended_duration_minutes=duration_min,
        expected_outcome=expected,
        recommended_at=ts,
    )

# ============================================================
# 10) REAL-TIME BUFFER ENDPOINTS (v2)
# ============================================================

# Initialize global buffer manager
buffer_manager = get_buffer_manager(max_readings=60)  # 60 readings = 1 hour


class PushReadingRequest(BaseModel):
    """Request for pushing sensor reading to buffer."""
    node_id: str
    temperature_c: float
    humidity_rh: float
    nh3_ppm: float
    timestamp: Optional[float] = None  # Unix epoch, defaults to now


class PushReadingResponse(BaseModel):
    """Response with computed rolling stats."""
    node_id: str
    received_at: str
    buffer_size: int
    rolling_stats: Dict[str, Any]
    features: Dict[str, float]


@app.post("/v1/push-reading", response_model=PushReadingResponse)
def v1_push_reading(req: PushReadingRequest):
    """
    Push a sensor reading to the buffer and get computed rolling stats.
    This endpoint should be called every minute by each sensor node.
    """
    stats = buffer_manager.push_reading(
        node_id=req.node_id,
        temperature=req.temperature_c,
        humidity=req.humidity_rh,
        ammonia=req.nh3_ppm,
        timestamp=req.timestamp
    )
    
    features = create_features_from_stats(stats)
    
    ts = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
    
    return PushReadingResponse(
        node_id=req.node_id,
        received_at=ts,
        buffer_size=stats.buffer_size,
        rolling_stats=stats.to_dict(),
        features=features,
    )


class BufferStatsResponse(BaseModel):
    """Response with buffer statistics for a node."""
    node_id: str
    buffer_size: int
    max_readings: int
    last_update: Optional[float]
    rolling_stats: Dict[str, Any]


@app.get("/v1/buffer-stats/{node_id}", response_model=BufferStatsResponse)
def v1_buffer_stats(node_id: str):
    """Get current buffer stats for a node without pushing new data."""
    info = buffer_manager.get_node_info(node_id)
    buffer = buffer_manager.get_buffer(node_id)
    
    return BufferStatsResponse(
        node_id=node_id,
        buffer_size=buffer.size(),
        max_readings=buffer.max_readings,
        last_update=buffer.last_update if buffer.last_update > 0 else None,
        rolling_stats=info["stats"],
    )


class BufferMemoryResponse(BaseModel):
    """Response with memory usage info."""
    num_nodes: int
    total_readings: int
    estimated_bytes: int
    estimated_mb: float


@app.get("/v1/buffer-memory", response_model=BufferMemoryResponse)
def v1_buffer_memory():
    """Get memory usage estimate for all buffers."""
    mem = buffer_manager.memory_usage_estimate()
    return BufferMemoryResponse(**mem)


@app.get("/v1/buffer-nodes")
def v1_buffer_nodes():
    """Get list of all nodes with buffers."""
    nodes = buffer_manager.get_all_nodes()
    return {
        "nodes": nodes,
        "count": len(nodes),
    }


@app.post("/v1/buffer-clear/{node_id}")
def v1_buffer_clear(node_id: str):
    """Clear buffer for a specific node."""
    cleared = buffer_manager.clear_node(node_id)
    return {
        "ok": cleared,
        "node_id": node_id,
        "msg": f"Buffer cleared for {node_id}" if cleared else f"No buffer found for {node_id}",
    }


@app.post("/v1/buffer-clear-all")
def v1_buffer_clear_all():
    """Clear all buffers."""
    count = buffer_manager.clear_all()
    return {
        "ok": True,
        "cleared_count": count,
        "msg": f"Cleared {count} buffers",
    }


# ============================================================
# 11) ENHANCED DECIDE v2 (with buffer features)
# ============================================================

class DecideV2Request(BaseModel):
    """Enhanced decide request using buffer features."""
    node_id: str
    temperature_c: float
    humidity_rh: float
    nh3_ppm: float
    timestamp: Optional[float] = None
    use_buffer: bool = True  # Whether to use/update buffer
    request_id: Optional[str] = None


class DecideV2Response(BaseModel):
    """Enhanced decide response with buffer-computed features."""
    grade: str
    probabilities: Optional[Dict[str, float]] = None
    sprayer_on: bool
    sprayer_reason: str
    anomaly: Dict[str, Any]
    used_thresholds: Dict[str, float]
    buffer_stats: Optional[Dict[str, Any]] = None
    features_used: Dict[str, float]
    note: str
    request_id: Optional[str] = None


@app.post("/v2/decide", response_model=DecideV2Response)
def v2_decide(req: DecideV2Request):
    """
    Enhanced decide endpoint that uses buffer for rolling features.
    This is the recommended endpoint for production use with 1-minute sensor data.
    """
    t = req.temperature_c
    r = req.humidity_rh
    n = req.nh3_ppm
    
    # Push to buffer and get stats
    if req.use_buffer:
        stats = buffer_manager.push_reading(
            node_id=req.node_id,
            temperature=t,
            humidity=r,
            ammonia=n,
            timestamp=req.timestamp
        )
        buffer_features = create_features_from_stats(stats)
        buffer_stats_dict = stats.to_dict()
    else:
        buffer_features = None
        buffer_stats_dict = None
    
    # Build features - use buffer features if available
    if buffer_features:
        extra_features = {
            "temp_avg_1h": buffer_features.get("temp_avg_1h", t),
            "humid_avg_1h": buffer_features.get("humid_avg_1h", r),
            "nh3_avg_1h": buffer_features.get("nh3_avg_1h", n),
            "temp_delta_1h": buffer_features.get("temp_delta_1h", 0),
            "humid_delta_1h": buffer_features.get("humid_delta_1h", 0),
            "nh3_delta_1h": buffer_features.get("nh3_delta_1h", 0),
        }
    else:
        extra_features = None
    
    # Predict grade using enhanced features
    grade, probs = predict_grade_from_values(t, r, n, recorded_ts=req.timestamp, extra=extra_features)
    
    # Spray decision
    st = get_state(req.node_id)
    on, reason = decide_spray_with_state(st, t, r, n, ctrl, now_ts=req.timestamp)
    
    # Anomaly detection
    anom = _detect_anomaly({TEMP_COL: t, RH_COL: r, NH3_COL: n})
    
    # Save decision to DB
    save_decision({
        "node_id": req.node_id,
        "temp_c": t,
        "rh": r,
        "nh3_ppm": n,
        "grade_pred": grade,
        "p_bagus": (probs or {}).get("bagus"),
        "p_sedang": (probs or {}).get("sedang"),
        "p_buruk":  (probs or {}).get("buruk"),
        "sprayer_on": on,
        "sprayer_reason": reason,
        "anom_verdict": anom.get("verdict"),
        "anom_iso_pred": anom.get("iso_pred"),
        "anom_iso_score": anom.get("iso_score"),
        "anom_flags": anom.get("flags"),
        "used_temp_on": ctrl.get("temp_on"),
        "used_temp_off": ctrl.get("temp_off"),
        "used_rh_on": ctrl.get("rh_on"),
        "used_rh_off": ctrl.get("rh_off"),
    })
    
    # Build features dict for response
    features_used = buffer_features if buffer_features else {
        "temperature": t,
        "humidity": r,
        "ammonia": n,
        "temp_avg_1h": t,
        "humid_avg_1h": r,
        "nh3_avg_1h": n,
    }
    
    return DecideV2Response(
        grade=grade,
        probabilities=probs,
        sprayer_on=on,
        sprayer_reason=reason,
        anomaly=anom,
        used_thresholds={
            "temp_on": ctrl.get("temp_on"),
            "temp_off": ctrl.get("temp_off"),
            "rh_on":   ctrl.get("rh_on"),
            "rh_off":  ctrl.get("rh_off"),
        },
        buffer_stats=buffer_stats_dict,
        features_used=features_used,
        note="Enhanced v2 with sliding window buffer features." if req.use_buffer else "Buffer disabled, using single-point features.",
        request_id=req.request_id,
    )


# ============================================================
# 12) ENTRYPOINT (untuk python app.py langsung)
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)


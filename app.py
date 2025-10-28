# app.py
import os, json, time, pickle, re
from typing import Optional, Dict, Any
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

# ============ 0) Konfigurasi dasar ============
ART_DIR   = os.getenv("ART_DIR", ".")
PATH_MODEL = os.path.join(ART_DIR, "best_model.pkl")
PATH_PREP  = os.path.join(ART_DIR, "preprocessor.pkl")
PATH_CFG   = os.path.join(ART_DIR, "config.json")
PATH_ANOM  = os.path.join(ART_DIR, "iso_anomaly.pkl")
PATH_CTRL  = os.path.join(ART_DIR, "control_config.json")
PATH_DATA  = os.path.join(ART_DIR, "data", "lantai1_data.csv")  # optional untuk median

DEFAULT_MIN_ON   = float(os.getenv("MIN_ON",   20))
DEFAULT_MIN_OFF  = float(os.getenv("MIN_OFF",  40))
DEFAULT_COOLDOWN = float(os.getenv("COOLDOWN", 60))

# ============ 1) Artefak ML ============
def _safe_load_pickle(p):
    with open(p, "rb") as f:
        return pickle.load(f)

def _load_json(p, default=None):
    if not os.path.exists(p):
        return default
    with open(p, "r") as f:
        return json.load(f)

best_model   = _safe_load_pickle(PATH_MODEL)
preprocessor = _safe_load_pickle(PATH_PREP)
cfg  = _load_json(PATH_CFG, {})
ctrl = _load_json(PATH_CTRL, {"temp_on":30.5, "temp_off":29.5, "rh_on":83.0, "rh_off":85.0})
iso  = _safe_load_pickle(PATH_ANOM) if os.path.exists(PATH_ANOM) else None

feature_cols  = cfg.get("feature_cols", [])
idx_to_class  = cfg.get("idx_to_class", {})
class_to_idx  = cfg.get("class_to_idx", {})

if not feature_cols:
    raise RuntimeError("config.json tidak memiliki 'feature_cols' — pastikan artefak training benar.")

# ============ 2) Default fitur (median) ============
def safe_read_table(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    try: return pd.read_csv(path)
    except Exception: pass
    try: return pd.read_csv(path, sep=None, engine="python")
    except Exception: pass
    for enc in ["utf-8-sig","utf-16","latin-1"]:
        for sep in [",",";","\t","|"]:
            try: return pd.read_csv(path, sep=sep, engine="python", encoding=enc)
            except Exception: continue
    raise RuntimeError("Gagal membaca dataset default")

feature_defaults: Dict[str, float] = {c: 0.0 for c in feature_cols}
try:
    if os.path.exists(PATH_DATA):
        df_default = safe_read_table(PATH_DATA)
        df_default.columns = [c.strip().lower() for c in df_default.columns]
        for c in feature_cols:
            if c in df_default.columns:
                df_default[c] = pd.to_numeric(df_default[c], errors="coerce")
        med = df_default[feature_cols].median(numeric_only=True)
        for c in feature_cols:
            feature_defaults[c] = float(np.nan_to_num(med.get(c, 0.0)))
except Exception:
    pass

# ============ 3) Mapping nama kolom inti ============
def _find_col(pats):
    for pat in pats:
        for c in feature_cols:
            if re.search(pat, c, flags=re.I):
                return c
    return None

TEMP_COL = _find_col([r"^suhu$", r"temp"])
RH_COL   = _find_col([r"lembab|lembap|humid|\brh\b"])
NH3_COL  = _find_col([r"amoni|nh ?3|ammonia|amonia"])

# ============ 4) State & logika sprayer (per-node) ============
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
    now_ts=None
):
    now = time.time() if now_ts is None else now_ts
    if temp_c is None or rh is None:
        return False, "sensor_invalid"

    t_on, t_off   = cfg_ctrl.get("temp_on",30.5), cfg_ctrl.get("temp_off",29.5)
    rh_on, rh_off = cfg_ctrl.get("rh_on",83.0),  cfg_ctrl.get("rh_off",85.0)

    # Jika ingin NH3 mempengaruhi ON, aktifkan baris nh3_high & gabungkan ke turn_on_cond
    # nh3_high = (nh3_ppm is not None and nh3_ppm >= 20.0)
    turn_on_cond  = (temp_c >= t_on) or (rh <= rh_on)  # or nh3_high
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

# ============ 5) Schemas ============
class DecideRequest(BaseModel):
    node_id: str = Field(..., description="ID unik node/RBW (mis. RBW_L1)")
    temperature_c: Optional[float] = Field(None, description="Suhu °C")
    humidity_rh:   Optional[float] = Field(None, description="RH %")
    nh3_ppm:       Optional[float] = Field(None, description="Amonia ppm")
    extra_features: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None  # diabaikan; server-time
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

# ============ 6) Helpers ML ============
def _build_feature_row(req: DecideRequest) -> pd.DataFrame:
    row = {c: feature_defaults.get(c, 0.0) for c in feature_cols}
    if TEMP_COL and req.temperature_c is not None: row[TEMP_COL] = float(req.temperature_c)
    if RH_COL   and req.humidity_rh   is not None: row[RH_COL]   = float(req.humidity_rh)
    if NH3_COL  and req.nh3_ppm       is not None: row[NH3_COL]  = float(req.nh3_ppm)
    if req.extra_features:
        for k, v in req.extra_features.items():
            if k in row and v is not None:
                try: row[k] = float(v)
                except Exception: pass
    return pd.DataFrame([row], columns=feature_cols)

def _predict_grade(row_df: pd.DataFrame):
    pred_idx = int(best_model.predict(row_df)[0])
    grade = idx_to_class.get(pred_idx, str(pred_idx))
    probs = None
    try:
        proba = best_model.predict_proba(row_df)[0]
        probs = {idx_to_class.get(i, str(i)): float(round(p, 6)) for i, p in enumerate(proba)}
    except Exception:
        pass
    return grade, probs

def _detect_anomaly(sample_map: Dict[str, float]):
    flags = []
    t = sample_map.get(TEMP_COL); r = sample_map.get(RH_COL); n = sample_map.get(NH3_COL)

    if n is not None and n >= 20.0: flags.append("NH3_HIGH")
    if r is not None and (r < 0 or r > 100): flags.append("RH_OUT_OF_RANGE")
    if t is not None and (t < 10 or t > 45): flags.append("TEMP_OUT_OF_RANGE")

    if iso is None:
        return {"verdict": "unknown", "flags": flags}

    anom_features = [c for c in feature_cols if ("suhu" in c or "lembab" in c or "lembap" in c or "humid" in c or "rh" in c or "amoni" in c or "nh3" in c)]
    row = {c: feature_defaults.get(c, 0.0) for c in anom_features}
    if t is not None and TEMP_COL in row: row[TEMP_COL] = t
    if r is not None and RH_COL   in row: row[RH_COL]   = r
    if n is not None and NH3_COL  in row: row[NH3_COL]  = n
    df_row = pd.DataFrame([row], columns=anom_features)

    try:
        iso_pred = int(iso.predict(df_row)[0])   # 1 normal, -1 anomaly
        iso_score = float(iso.decision_function(df_row)[0])
        verdict = "anomaly" if (iso_pred == -1 or flags) else "normal"
    except Exception:
        iso_pred, iso_score, verdict = 1, 0.0, "normal" if not flags else "anomaly"

    return {"verdict": verdict, "iso_pred": iso_pred, "iso_score": iso_score, "flags": flags}

# ============ 7) DB setup (SQLAlchemy) ============
from sqlalchemy import create_engine, Column, String, Boolean, Float, JSON, DateTime, text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql import func

DATABASE_URL = os.getenv("DATABASE_URL")  # e.g. postgresql+psycopg://swift:swiftpass@localhost:5432/swiftlet
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
    p_jelek    = Column(Float)

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
    """Simpan hasil /decide ke DB (no-op bila DATABASE_URL tidak di-set)."""
    if not engine:
        return
    db = SessionLocal()
    try:
        db.add(Decision(**row))
        db.commit()
    finally:
        db.close()

# ============ 8) FastAPI app ============
app = FastAPI(title="Swiftlet ML Backend", version="1.3.0")

# CORS untuk integrasi BE utama / dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "has_anomaly_model": bool(iso is not None),
        "n_features": len(feature_cols),
        "version": "1.3.0"
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
    return {"service": "swiftlet-ml", "version": "1.3.0", "model_classes": idx_to_class}

# Reset state: per-node (node_id) atau global bila tidak diberikan
@app.post("/reset_state")
def reset_state(node_id: Optional[str] = Query(default=None)):
    if node_id:
        st = get_state(node_id)
        st.is_on = False; st.last_change_ts = 0.0; st.last_off_ts = 0.0
        return {"ok": True, "msg": f"Spray state reset for '{node_id}'"}
    for st in STATES.values():
        st.is_on = False; st.last_change_ts = 0.0; st.last_off_ts = 0.0
    return {"ok": True, "msg": "Spray state reset for ALL nodes"}

# Lihat state: per-node (wajib node_id)
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
        "last_off_ts": st.last_off_ts
    }

@app.post("/decide", response_model=DecideResponse)
def decide(req: DecideRequest):
    # bangun fitur utk model
    row_df = _build_feature_row(req)

    # nilai inti
    t = float(row_df[TEMP_COL].iloc[0]) if TEMP_COL else None
    r = float(row_df[RH_COL].iloc[0])   if RH_COL   else None
    n = float(row_df[NH3_COL].iloc[0])  if NH3_COL  else None

    # prediksi grade
    try:
        grade, probs = _predict_grade(row_df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal prediksi grade: {e}")

    # keputusan sprayer (per-node, waktu server)
    st = get_state(req.node_id)
    on, reason = decide_spray_with_state(st, t, r, n, ctrl, now_ts=None)

    # deteksi anomali
    anom = _detect_anomaly({TEMP_COL: t, RH_COL: r, NH3_COL: n})

    # simpan ke DB
    save_decision({
        "node_id": req.node_id,
        "temp_c": t, "rh": r, "nh3_ppm": n,
        "grade_pred": grade,
        "p_bagus": (probs or {}).get("bagus"),
        "p_sedang": (probs or {}).get("sedang"),
        "p_jelek":  (probs or {}).get("jelek"),
        "sprayer_on": on, "sprayer_reason": reason,
        "anom_verdict": anom.get("verdict"),
        "anom_iso_pred": anom.get("iso_pred"),
        "anom_iso_score": anom.get("iso_score"),
        "anom_flags": anom.get("flags"),
        "used_temp_on": ctrl.get("temp_on"),
        "used_temp_off": ctrl.get("temp_off"),
        "used_rh_on": ctrl.get("rh_on"),
        "used_rh_off": ctrl.get("rh_off"),
    })

    # debug per-node
    now_ts = time.time()
    debug = {
        "node_id": req.node_id,
        "is_on": st.is_on,
        "elapsed_since_change_s": round(now_ts - st.last_change_ts, 2),
        "min_on_s": st.min_on,
        "min_off_s": st.min_off,
        "cooldown_remaining_s": round(max(0.0, st.cooldown - (now_ts - st.last_off_ts)), 2),
        "last_change_ts": st.last_change_ts,
        "last_off_ts": st.last_off_ts
    }

    return DecideResponse(
        grade=grade,
        probabilities=probs,
        sprayer_on=on,
        sprayer_reason=reason,
        anomaly=anom,
        used_thresholds={
            "temp_on": ctrl.get("temp_on"), "temp_off": ctrl.get("temp_off"),
            "rh_on":   ctrl.get("rh_on"),   "rh_off":  ctrl.get("rh_off")
        },
        note="Server-time hysteresis aktif; fitur hilang diisi median training. Tambahkan extra_features untuk akurasi lebih baik.",
        debug=debug,
        request_id=req.request_id
    )

# ============================================
# Smart Swiftlet Farming — Harvest Grade ML v1.1
# Train & Test + Auto-Sprayer Control + Anomaly Detection
# Kaggle-ready, jalan lokal juga (VS Code)
# ============================================

import os, glob, re, json, warnings, pickle, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# LightGBM (opsional)
try:
    import lightgbm as lgb
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

# ---------------------------
# 0) Utils
# ---------------------------
def log(*args): print("[INFO]", *args)

def resolve_input_csv(preferred_name_regex=r".*lantai.*\.csv"):
    """Prioritas: lokal ./data/ → Kaggle /kaggle/input → raise"""
    local_try = os.path.join("data", "lantai1_data.csv")
    if os.path.exists(local_try):
        return local_try

    kaggle_dirs = glob.glob("/kaggle/input/*")
    for kd in kaggle_dirs:
        for f in glob.glob(os.path.join(kd, "**", "*.csv"), recursive=True):
            if re.search(preferred_name_regex, os.path.basename(f), flags=re.I):
                return f
    for kd in kaggle_dirs:
        any_csvs = glob.glob(os.path.join(kd, "**", "*.csv"), recursive=True)
        if any_csvs:
            return any_csvs[0]

    raise FileNotFoundError("CSV tidak ditemukan. Letakkan di ./data/lantai1_data.csv atau attach dataset di Kaggle.")

def find_col(candidates, cols):
    for pat in candidates:
        for c in cols:
            if re.search(pat, c, flags=re.I):
                return c
    return None

# ---------------------------
# 1) Load CSV & normalisasi kolom
# ---------------------------
csv_path = resolve_input_csv()
log(f"Using CSV: {csv_path}")

df = pd.read_csv(csv_path)
df.columns = [c.strip().lower() for c in df.columns]
log("Columns:", df.columns.tolist(), "| Shape:", df.shape)

# Mapping kolom penting
time_col = find_col([r"timestamp", r"waktu", r"date", r"datetime", r"tgl", r"tanggal"], df.columns)
temp_col = find_col([r"^suhu$", r"temp(erature)?", r"\bt_?c\b", r"t_?air"], df.columns)
rh_col   = find_col([r"kelembab(an|an)?", r"kelembap", r"humid", r"\brh\b"], df.columns)
nh3_col  = find_col([r"amoni(a|ak)", r"\bnh ?3\b", r"ammonia", r"amonia"], df.columns)
target_col = find_col([r"^grade(_panen)?$", r"label", r"target", r"kualitas", r"kelas"], df.columns)

print("[MAP] time_col =", time_col)
print("[MAP] temp_col =", temp_col)
print("[MAP] rh_col   =", rh_col)
print("[MAP] nh3_col  =", nh3_col)
print("[MAP] target   =", target_col)

# ---------------------------
# 2) Preprocess dasar
# ---------------------------
if time_col is not None:
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

num_cols = [c for c in [temp_col, rh_col, nh3_col] if c is not None]
if not num_cols:
    raise ValueError("Tidak ada kolom numerik kunci (suhu/kelembapan/amonia).")

for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=num_cols, how="all").copy()

# ---------------------------
# 3) Label pseudo jika target kosong
# ---------------------------
def derive_grade_rowwise(d, t_col, r_col, n_col,
                         t_min=26.0, t_max=30.0,
                         rh_min=84.0, rh_max=88.0,
                         nh3_max=20.0,
                         tol_t=1.0, tol_rh=3.0, tol_nh3=5.0):
    t = d[t_col] if t_col else np.nan
    r = d[r_col] if r_col else np.nan
    n = d[n_col] if n_col else np.nan
    good_t  = (not np.isnan(t))  and (t_min <= t <= t_max)
    good_rh = (not np.isnan(r))  and (rh_min <= r <= rh_max)
    good_n  = (not np.isnan(n))  and (n < nh3_max)
    if good_t and good_rh and good_n:
        return "bagus"
    near_t  = (not np.isnan(t)) and (t_min - tol_t <= t <= t_max + tol_t)
    near_rh = (not np.isnan(r)) and (rh_min - tol_rh <= r <= rh_max + tol_rh)
    near_n  = (not np.isnan(n)) and (n < nh3_max + tol_nh3)
    if sum([near_t, near_rh, near_n]) >= 2:
        return "sedang"
    return "jelek"

if target_col is None:
    if (temp_col is None) or (nh3_col is None) or (rh_col is None):
        raise ValueError("Target tidak ada dan fitur kunci tidak lengkap untuk pseudo label.")
    log("Target tidak ada → membuat 'grade' pseudo.")
    df["grade"] = df.apply(lambda row: derive_grade_rowwise(row, temp_col, rh_col, nh3_col), axis=1)
    target_col = "grade"

# ---------------------------
# 4) Fitur waktu (rolling)
# ---------------------------
def add_time_features(_df, _time_col, cols, windows=("1h","6h","24h")):
    if _time_col is None:
        return _df
    _df = _df.set_index(_time_col)
    for c in cols:
        if c not in _df.columns: 
            continue
        for w in windows:
            _df[f"{c}_rollmean_{w}"] = _df[c].rolling(w, min_periods=3).mean()
            _df[f"{c}_rollstd_{w}"]  = _df[c].rolling(w, min_periods=3).std()
    _df = _df.reset_index()
    return _df

df = add_time_features(df, time_col, num_cols)

# ---------------------------
# 5) Build X, y
# ---------------------------
y_raw = df[target_col].astype(str).str.strip().str.lower()
class_names = sorted(y_raw.unique().tolist())
class_to_idx = {c:i for i,c in enumerate(class_names)}
idx_to_class = {i:c for c,i in class_to_idx.items()}
y = y_raw.map(class_to_idx)

feature_cols = []
for c in df.columns:
    if c in [target_col, time_col]:
        continue
    if pd.api.types.is_numeric_dtype(df[c]):
        feature_cols.append(c)
feature_cols = [c for c in feature_cols if not df[c].isna().all()]

if not feature_cols:
    raise ValueError("Tidak ada fitur numerik valid setelah pembersihan.")

X = df[feature_cols].copy()
X = X.fillna(X.median(numeric_only=True))

print("[INFO] Classes map:", class_to_idx)
print("[INFO] Samples:", len(X), "| Features:", len(feature_cols))
print("[INFO] Class distribution:", y.value_counts().sort_index().to_dict())

# ---------------------------
# 6) Safeguards kelas & split
# ---------------------------
def describe_classes(name, y_series):
    vc = pd.Series(y_series).value_counts().sort_index().to_dict()
    print(f"[CLASSES] {name}: {vc} | n_unique={len(set(y_series))}")

def make_proxy_labels(df_src, t_col, r_col, n_col):
    t = df_src[t_col].astype(float)
    r = df_src[r_col].astype(float)
    n = df_src[n_col].astype(float)
    s_temp = -np.abs(t - 28.0)
    s_rh   = -np.abs(r - 86.0)
    s_nh3  = -(np.maximum(0, n - 15.0))
    score = (0.4*s_temp + 0.4*s_rh + 0.2*s_nh3)
    q = np.quantile(score.dropna(), [1/3, 2/3])
    labels = pd.Series(index=df_src.index, dtype=object)
    labels[score <= q[0]] = "jelek"
    labels[(score > q[0]) & (score <= q[1])] = "sedang"
    labels[score > q[1]] = "bagus"
    return labels

if len(set(y)) < 2:
    print("[WARN] Dataset hanya 1 kelas. Membuat proxy labels (terciles) ...")
    proxy = make_proxy_labels(df, temp_col, rh_col, nh3_col)
    y_raw = proxy.astype(str).str.strip().str.lower()
    class_names = sorted(y_raw.unique().tolist())
    class_to_idx = {c:i for i,c in enumerate(class_names)}
    idx_to_class = {i:c for c,i in class_to_idx.items()}
    y = y_raw.map(class_to_idx)

describe_classes("overall", y)

def time_or_stratified_split(Xs, ys, has_time):
    if has_time:
        n = len(Xs)
        split_idx = int(n * 0.75)
        X_tr, X_te = Xs.iloc[:split_idx], Xs.iloc[split_idx:]
        y_tr, y_te = ys.iloc[:split_idx], ys.iloc[split_idx:]
        if len(set(y_tr)) >= 2 and len(set(y_te)) >= 2:
            print("[SPLIT] Using time-based split (75/25).")
            return X_tr, X_te, y_tr, y_te
        print("[WARN] Time-based split <2 kelas. Fallback stratified.")
    stratify_opt = ys if len(set(ys)) > 1 else None
    return train_test_split(Xs, ys, test_size=0.25, random_state=42, stratify=stratify_opt)

X_train, X_test, y_train, y_test = time_or_stratified_split(X, y, time_col is not None)
describe_classes("train", y_train); describe_classes("test", y_test)

if len(set(y_train)) < 2:
    print("[WARN] Train 1 kelas. Undersampling mayoritas ...")
    train_df = X_train.copy(); train_df["__y__"] = y_train.values
    maj = train_df["__y__"].mode()[0]
    df_maj = train_df[train_df["__y__"] == maj]
    df_min = train_df[train_df["__y__"] != maj]
    k = max(50, 2*len(df_min)) if len(df_min) > 0 else min(len(df_maj), 200)
    df_maj_sub = df_maj.sample(n=min(k, len(df_maj)), random_state=42)
    resamp = pd.concat([df_maj_sub, df_min], ignore_index=True).sample(frac=1, random_state=42)
    X_train = resamp.drop(columns="__y__"); y_train = resamp["__y__"]

n_classes_train = len(set(y_train))
print(f"[INFO] n_classes_train={n_classes_train}")

# ---------------------------
# 7) Preprocessor & Models
# ---------------------------
preprocessor = ColumnTransformer(
    transformers=[("num", StandardScaler(with_mean=False), feature_cols)],
    remainder="drop"
)

models = {}
rf_clf = RandomForestClassifier(n_estimators=400, n_jobs=-1, class_weight="balanced", random_state=42)
models["random_forest"] = Pipeline([("prep", preprocessor), ("clf", rf_clf)])

if HAS_LGBM and n_classes_train >= 2:
    if n_classes_train == 2:
        lgb_params = dict(objective="binary")
    else:
        lgb_params = dict(objective="multiclass", num_class=n_classes_train)
    lgbm_clf = lgb.LGBMClassifier(
        n_estimators=800, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, random_state=42, **lgb_params
    )
    models["lightgbm"] = Pipeline([("prep", preprocessor), ("clf", lgbm_clf)])
else:
    print("[INFO] Skip LightGBM.")

# ---------------------------
# 8) Train, Evaluate, Pick Best
# ---------------------------
best_name, best_f1, best_model = None, -1.0, None
scores = {}
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1m = f1_score(y_test, pred, average="macro")
    scores[name] = {"accuracy": acc, "f1_macro": f1m}
    print(f"\n=== {name.upper()} ===")
    print("Accuracy :", acc)
    print("F1-macro :", f1m)
    print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
    target_names = [idx_to_class[i] for i in sorted(set(y))]
    print("\nClassification Report:")
    print(classification_report(y_test, pred, target_names=target_names))
    if f1m > best_f1:
        best_name, best_f1, best_model = name, f1m, pipe

print("\n[SUMMARY] Scores:", json.dumps(scores, indent=2))
print(f"[BEST] {best_name} with F1-macro={best_f1:.4f}")

# ---------------------------
# 9) Save artefacts (model grade)
# ---------------------------
os.makedirs(".", exist_ok=True)
with open("preprocessor.pkl", "wb") as f:
    pickle.dump(preprocessor, f)
with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
meta = {
    "feature_cols": feature_cols,
    "class_to_idx": {v:k for k,v in idx_to_class.items()},
    "idx_to_class": idx_to_class,
    "time_split": bool(time_col is not None),
    "source_csv": csv_path,
    "note": "Safeguards single-class; LightGBM objective by n_classes_train; pseudo labels if missing."
}
with open("config.json", "w") as f:
    json.dump(meta, f, indent=2)
print("[OK] Saved: preprocessor.pkl, best_model.pkl, config.json")

# ---------------------------
# 10) Quick inference demo
# ---------------------------
def predict_one(sample_dict):
    row = pd.DataFrame([sample_dict])
    for c in feature_cols:
        if c not in row.columns:
            row[c] = X[c].median() if c in X.columns else 0.0
    row = row[feature_cols].fillna(X.median(numeric_only=True))
    pred_idx = best_model.predict(row)[0]
    try:
        proba = best_model.predict_proba(row)[0].tolist()
    except Exception:
        proba = None
    return idx_to_class[int(pred_idx)], proba

example = {c: float(np.nan_to_num(X.median(numeric_only=True)[c])) for c in feature_cols}
pl, pp = predict_one(example)
print("\n[INFERENCE EXAMPLE]")
print("Input(example):", {k: round(v,3) for k,v in example.items()})
print("Predicted grade:", pl)
if pp is not None:
    print("Probabilities :", {idx_to_class[i]: round(p,3) for i,p in enumerate(pp)})

# ==============================================================
# 11) AUTO-SPRAYER (rule-based + hysteresis) & ANOMALY DETECTOR
# ==============================================================

from dataclasses import dataclass

@dataclass
class SprayState:
    is_on: bool = False
    last_change_ts: float = 0.0
    last_off_ts: float = 0.0
    cooldown: float = 60.0
    min_on: float = 20.0
    min_off: float = 40.0

def decide_spray(temp_c, rh, nh3_ppm, st: SprayState, cfg: dict, now_ts=None):
    now = time.time() if now_ts is None else now_ts
    if temp_c is None or rh is None or pd.isna(temp_c) or pd.isna(rh):
        return False, "sensor_invalid"

    t_on, t_off = cfg["temp_on"], cfg["temp_off"]
    rh_on, rh_off = cfg["rh_on"], cfg["rh_off"]

    turn_on_cond  = (temp_c >= t_on) or (rh <= rh_on)
    turn_off_cond = (temp_c <= t_off) and (rh >= rh_off)

    elapsed = now - st.last_change_ts
    if st.is_on:
        if elapsed < st.min_on:
            return True, "hold_min_on"
        if turn_off_cond:
            st.is_on = False; st.last_change_ts = now; st.last_off_ts = now
            return False, "off_by_condition"
        return True, "stay_on"
    else:
        if (now - st.last_off_ts) < st.cooldown or elapsed < st.min_off:
            return False, "hold_cooldown_or_min_off"
        if turn_on_cond:
            st.is_on = True; st.last_change_ts = now
            return True, "on_by_condition"
        return False, "stay_off"

# ---- IsolationForest + hard rules
from sklearn.ensemble import IsolationForest

anom_features = [c for c in X.columns if ("suhu" in c or "kelembab" in c or "amoni" in c or "nh3" in c)]
normal_mask = (df[nh3_col] < 15.0) if nh3_col is not None else pd.Series(True, index=df.index)
X_anom_train = X.loc[normal_mask, anom_features].fillna(X.median(numeric_only=True))

iso = IsolationForest(n_estimators=300, contamination=0.03, random_state=42).fit(X_anom_train)

def detect_anomaly(row_dict):
    row = pd.DataFrame([row_dict])
    for c in anom_features:
        if c not in row.columns:
            row[c] = X[c].median() if c in X.columns else 0.0
    row = row[anom_features].fillna(X.median(numeric_only=True))
    iso_pred = iso.predict(row)[0]            # 1 normal, -1 anomaly
    iso_score = iso.decision_function(row)[0]

    flags = []
    t = row_dict.get(temp_col); r = row_dict.get(rh_col); n = row_dict.get(nh3_col)
    if (n is not None) and (not pd.isna(n)) and n >= 20.0: flags.append("NH3_HIGH")
    if (r is not None) and (not pd.isna(r)) and (r < 0 or r > 100): flags.append("RH_OUT_OF_RANGE")
    if (t is not None) and (not pd.isna(t)) and (t < 10 or t > 45): flags.append("TEMP_OUT_OF_RANGE")

    verdict = "anomaly" if (iso_pred == -1 or len(flags)>0) else "normal"
    return {"verdict": verdict, "iso_pred": int(iso_pred), "iso_score": float(iso_score), "flags": flags}

# ---------------------------
# 12) Export artefak kontrol & detektor
# ---------------------------
control_cfg = {
    "temp_on": 30.5, "temp_off": 29.5,   # hysteresis suhu
    "rh_on": 83.0,   "rh_off": 85.0,     # hysteresis RH
}
with open("iso_anomaly.pkl", "wb") as f:
    pickle.dump(iso, f)
with open("control_config.json", "w") as f:
    json.dump(control_cfg, f, indent=2)
print("[OK] Saved: iso_anomaly.pkl, control_config.json")

# ---------------------------
# 13) Contoh keputusan sekali jalan
# ---------------------------
st = SprayState()
sample = example  # bisa diganti bacaan sensor live
spray_on, reason = decide_spray(sample.get(temp_col), sample.get(rh_col), sample.get(nh3_col),
                                st, cfg=control_cfg, now_ts=time.time())
anom = detect_anomaly(sample)
print("\n[ACTION SAMPLE]")
print({"spray_on": spray_on, "reason": reason, "anomaly": anom})

# ==============================================================
# 14) BACKTEST: simulasi sprayer & anomaly di seluruh dataset
# ==============================================================

def _infer_seconds_per_sample(df, tcol):
    """Estimasi interval sample (detik)."""
    if tcol is None: 
        return 60.0  # asumsi 1 menit
    ts = pd.to_datetime(df[tcol], errors="coerce").dropna().sort_values().values
    if len(ts) < 2: 
        return 60.0
    diffs = np.diff(ts.astype("datetime64[ns]").astype(np.int64) / 1e9)
    return float(np.median(diffs)) if len(diffs) else 60.0

def backtest_policy(df, time_col, temp_col, rh_col, nh3_col, control_cfg):
    # urutkan waktu kalau ada
    if time_col is not None:
        df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    # kolom minimal
    need = [temp_col, rh_col] + ([nh3_col] if nh3_col else [])
    df = df.dropna(subset=[c for c in need if c], how="any").copy()
    if df.empty:
        print("[BACKTEST] Data kosong setelah filter kolom wajib.")
        return None

    # state & counters
    st = SprayState()
    last_ts = None
    total_on_seconds = 0.0
    switches = 0
    nh3_alerts = 0
    rows = []

    seconds_per_sample = _infer_seconds_per_sample(df, time_col)
    t0 = None; tN = None

    for i, row in df.iterrows():
        t = row[temp_col]; r = row[rh_col]; n = row[nh3_col] if nh3_col else None
        # timestamp (unix)
        if time_col is not None:
            now_ts = pd.Timestamp(row[time_col]).timestamp()
        else:
            now_ts = (i * seconds_per_sample)  # simulasi waktu relatif
        if t0 is None: t0 = now_ts
        tN = now_ts

        # keputusan
        prev = st.is_on
        on, reason = decide_spray(t, r, n, st, control_cfg, now_ts=now_ts)
        if on != prev:
            switches += 1

        # durasi ON (integrasi interval dari sample-1 ke sample)
        if last_ts is not None and st.is_on:
            total_on_seconds += (now_ts - last_ts)
        last_ts = now_ts

        # deteksi anomali
        sample_dict = {temp_col: t, rh_col: r}
        if nh3_col: sample_dict[nh3_col] = n
        anom_res = detect_anomaly(sample_dict)
        if "NH3_HIGH" in anom_res["flags"]:
            nh3_alerts += 1

        rows.append({
            "ts": now_ts,
            "time": (pd.to_datetime(row[time_col]).isoformat() if time_col else i),
            "suhu": float(t),
            "rh": float(r),
            "nh3": float(n) if nh3_col else None,
            "spray_on": bool(on),
            "reason": reason,
            "anomaly": anom_res["verdict"],
            "iso_score": anom_res["iso_score"],
            "flags": "|".join(anom_res["flags"])
        })

    # ringkasan
    duration_hours = max((tN - t0) / 3600.0, 1e-6)
    on_ratio = total_on_seconds / ((tN - t0) if (tN and t0) else len(df)*seconds_per_sample)
    switches_per_hour = switches / duration_hours

    # metrik tambahan
    df_log = pd.DataFrame(rows)
    avg_temp_on  = df_log.loc[df_log["spray_on"], "suhu"].mean()
    avg_temp_off = df_log.loc[~df_log["spray_on"], "suhu"].mean()
    avg_rh_on    = df_log.loc[df_log["spray_on"], "rh"].mean()
    avg_rh_off   = df_log.loc[~df_log["spray_on"], "rh"].mean()

    # overshoot RH (spray ON padahal RH sudah tinggi)
    overshoot_events = ((df_log["spray_on"]) & (df_log["rh"] >= 90.0)).sum()

    # simpan log
    df_log.to_csv("decisions_log.csv", index=False)
    print("\n[BACKTEST SUMMARY]")
    print(f"- Durasi data ~ {duration_hours:.2f} jam")
    print(f"- Persentase ON: {on_ratio*100:.1f}%")
    print(f"- Switching per jam: {switches_per_hour:.2f}")
    print(f"- Rata2 SUHU saat ON/OFF: {avg_temp_on:.2f} / {avg_temp_off:.2f} °C")
    print(f"- Rata2 RH saat ON/OFF  : {avg_rh_on:.2f} / {avg_rh_off:.2f} %")
    print(f"- NH3 alerts (hard rule >=20 ppm): {nh3_alerts}")
    print(f"- Overshoot RH (ON saat RH >=90%): {overshoot_events}")
    print("Log keputusan tersimpan: decisions_log.csv")

    return {
        "hours": duration_hours,
        "on_ratio": on_ratio,
        "switches_per_hour": switches_per_hour,
        "nh3_alerts": nh3_alerts,
        "overshoot_events": overshoot_events
    }

bt_summary = backtest_policy(df, time_col, temp_col, rh_col, nh3_col, control_cfg)

# ==============================================================
# 15) AUTO-TUNING THRESHOLDS (SARAN)
# ==============================================================
def suggest_thresholds(df, temp_col, rh_col, nh3_col, y_raw=None):
    d = df.copy()
    mask = pd.Series(True, index=d.index)
    if nh3_col is not None:
        mask &= (d[nh3_col] < 15.0)
    if y_raw is not None:
        mask &= (~y_raw.str.contains("jelek", na=False))
    base = d.loc[mask, [temp_col, rh_col]].dropna()

    if base.empty:
        print("[TUNE] Basis data normal kosong → pakai default.")
        return {"temp_on": 30.5, "temp_off": 29.5, "rh_on": 83.0, "rh_off": 85.0}

    q = base.quantile([0.1, 0.5, 0.7, 0.8, 0.9])
    # Ide sederhana:
    # - temp_on = P90 suhu normal (agar ON hanya saat benar2 panas)
    # - temp_off = median suhu normal
    # - rh_on = P10 RH normal (ON kalau RH turun di bawah batas bawah normal)
    # - rh_off = median RH normal
    sugg = {
        "temp_on": float(q.loc[0.90, temp_col]),
        "temp_off": float(q.loc[0.50, temp_col]),
        "rh_on": float(q.loc[0.10, rh_col]),
        "rh_off": float(q.loc[0.50, rh_col]),
    }
    # jaga hysteresis (ON>OFF untuk suhu, OFF>ON untuk RH)
    if sugg["temp_on"] <= sugg["temp_off"]:
        sugg["temp_on"] = sugg["temp_off"] + 0.5
    if sugg["rh_off"] <= sugg["rh_on"]:
        sugg["rh_off"] = sugg["rh_on"] + 1.0

    print("\n[THRESHOLD SUGGESTION]")
    print(json.dumps(sugg, indent=2))
    return sugg

_ = suggest_thresholds(df, temp_col, rh_col, nh3_col, y_raw)

# Selesai.

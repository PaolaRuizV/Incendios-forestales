"""
core.py - Nucleo compartido del proyecto de prediccion de amenaza de incendios.

UNICA FUENTE DE VERDAD: tanto el notebook como la app de Streamlit importan este
archivo. Asi es imposible que difieran en features, ingenieria de variables,
modelos (LightGBM + XGBoost + CatBoost), construccion de objetivos o metrica.

Para usarlo en Colab: sube core.py junto al notebook y el notebook hace `import core`.
Para la app: app.py hace `import core`.
"""
import io
import os
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression
from lightgbm import LGBMClassifier
import lightgbm as lgb
from xgboost import XGBClassifier

try:
    from catboost import CatBoostClassifier, Pool
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# ---------------------------------------------------------------------------
# Configuracion (identica al notebook)
# ---------------------------------------------------------------------------
SEED = 42
N_FOLDS = 3
HORIZONS = [12, 24, 48]          # Horizontes de prediccion (horas). 72h retirado del analisis
EVAL_HORIZONS = [24, 48]        # Horizontes usados en la metrica principal
CORE_VERSION = "streamlit_sin_72h_fast"
USE_GPU = False                  # CPU: evita fallos silenciosos sin GPU

ID_COL = "event_id"
TARGET = "event"
TIME_COL = "time_to_hit_hours"

LABELS_TARGET = {
    0: "No alcanzo la zona de evacuacion",
    1: "Alcanzo / en riesgo de alcanzar la zona",
}

# Caracteristicas crudas que el usuario puede ajustar en el formulario de la app.
# El resto se completan con la mediana de entrenamiento y TODAS las derivadas se
# recalculan con engineer_features() despues de aplicar estos valores.
FORM_FEATURES = [
    ("dist_min_ci_0_5h", "Distancia minima a la zona de evacuacion (m)"),
    ("closing_speed_m_per_h", "Velocidad de acercamiento a la zona (m/h)"),
    ("area_first_ha", "Area inicial del incendio (ha)"),
    ("area_growth_rate_ha_per_h", "Tasa de crecimiento de area (ha/h)"),
    ("alignment_abs", "Alineacion del avance hacia la zona (0 a 1)"),
    ("num_perimeters_0_5h", "Numero de perimetros observados"),
    ("radial_growth_rate_m_per_h", "Velocidad de avance radial (m/h)"),
    ("event_start_hour", "Hora de inicio del evento (0-23)"),
]


# ---------------------------------------------------------------------------
# Ingenieria de variables (copia exacta del notebook)
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea caracteristicas basadas en fisica para prediccion de incendios."""
    out = df.copy()
    distance = out["dist_min_ci_0_5h"].clip(lower=1)
    speed = out["closing_speed_m_per_h"]

    out["time_to_contact"] = distance / speed.clip(lower=0.01)
    out["log_time_to_contact"] = np.log1p(out["time_to_contact"].clip(0, 5000))
    out["danger_vector"] = out["alignment_abs"] * speed
    out["tracking_urgency"] = out["num_perimeters_0_5h"] * speed
    out["fire_intensity"] = out["area_growth_rate_ha_per_h"] * out["num_perimeters_0_5h"]
    out["approach_momentum"] = speed * out["alignment_abs"] / np.log1p(distance)
    out["log_dist"] = np.log1p(distance)
    out["dist_zone_critical"] = (distance < 5000).astype(np.float32)
    out["dist_zone_mid"] = ((distance >= 5000) & (distance < 15000)).astype(np.float32)
    out["speed_per_km"] = speed / (distance / 1000).clip(lower=0.1)

    out = out.replace([np.inf, -np.inf], np.nan).fillna(0)
    return out


def get_feature_cols(df_fe: pd.DataFrame):
    return [c for c in df_fe.columns if c not in [ID_COL, TARGET, TIME_COL]]


# ---------------------------------------------------------------------------
# Objetivos de supervivencia con manejo de censura (copia exacta)
# ---------------------------------------------------------------------------
def build_survival_targets(time_values, event_values, horizons):
    targets, masks = {}, {}
    for H in horizons:
        unknown = (event_values == 0) & (time_values < H)
        y = ((event_values == 1) & (time_values <= H)).astype(np.float64)
        y[unknown] = np.nan
        targets[H] = y
        masks[H] = ~unknown
    return targets, masks


# ---------------------------------------------------------------------------
# Parametros de los modelos (identicos al notebook)
# ---------------------------------------------------------------------------
def get_model_params():
    lgb_params = dict(
        objective="binary", learning_rate=0.035, num_leaves=15, max_depth=3,
        min_child_samples=25, subsample=0.80, colsample_bytree=0.70,
        reg_alpha=0.1, reg_lambda=1.0, n_estimators=100, random_state=SEED,
        verbose=-1, n_jobs=1, force_col_wise=True,
    )
    xgb_params = dict(
        objective="binary:logistic", learning_rate=0.035, max_depth=3,
        min_child_weight=25, subsample=0.80, colsample_bytree=0.70,
        reg_alpha=0.1, reg_lambda=1.0, n_estimators=100, random_state=SEED,
        verbosity=0, n_jobs=1,
        tree_method="gpu_hist" if USE_GPU else "hist", eval_metric="logloss",
    )
    cat_params = dict(
        iterations=100, learning_rate=0.035, depth=3, l2_leaf_reg=3.0,
        random_seed=SEED, verbose=0,
        task_type="GPU" if USE_GPU else "CPU", eval_metric="Logloss",
    )
    if USE_GPU:
        lgb_params["device"] = "gpu"
    return lgb_params, xgb_params, cat_params


def _fit_horizon_models(Xtr, ytr, Xva=None, yva=None):
    """Entrena el ensamble (LGB + XGB + CatBoost) para un horizonte. Sin except: pass."""
    lgb_params, xgb_params, cat_params = get_model_params()
    models = []

    m_lgb = LGBMClassifier(**lgb_params)
    if Xva is not None:
        m_lgb.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    else:
        m_lgb.fit(Xtr, ytr)
    models.append(m_lgb)

    m_xgb = XGBClassifier(**xgb_params)
    if Xva is not None:
        m_xgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    else:
        m_xgb.fit(Xtr, ytr)
    models.append(m_xgb)

    if HAS_CATBOOST:
        m_cat = CatBoostClassifier(**cat_params)
        if Xva is not None:
            m_cat.fit(Pool(Xtr, ytr), eval_set=Pool(Xva, yva), early_stopping_rounds=50)
        else:
            m_cat.fit(Pool(Xtr, ytr))
        models.append(m_cat)

    return models


def _predict_ensemble(models, X):
    """Promedia predict_proba de los modelos del horizonte."""
    preds = np.zeros(len(X))
    for m in models:
        preds += m.predict_proba(X)[:, 1]
    return preds / len(models)


# ---------------------------------------------------------------------------
# Metrica principal: C-Index + Brier ponderado solo sobre 24h y 48h
# ---------------------------------------------------------------------------
def c_index(time, event, risk):
    n = 0
    concordant = 0.0
    for i in range(len(time)):
        for j in range(len(time)):
            if event[i] == 1 and time[i] < time[j]:
                n += 1
                if risk[i] > risk[j]:
                    concordant += 1
                elif risk[i] == risk[j]:
                    concordant += 0.5
    return concordant / n if n > 0 else 0.5


def brier_at(time, event, prob, H):
    valid = ~((event == 0) & (time < H))
    if valid.sum() == 0:
        return 0.0
    y = ((event == 1) & (time <= H)).astype(float)[valid]
    p = np.asarray(prob)[valid]
    return float(np.mean((p - y) ** 2))


def hybrid_score(time, event, p24, p48, risk=None):
    """
    Hibrido = 0.3*C-Index + 0.7*(1 - Brier Ponderado).
    Se calcula solo sobre 24h y 48h. El horizonte 72h no se usa.
    """
    w24, w48 = 0.3 / 0.7, 0.4 / 0.7
    if risk is None:
        risk = w24 * p24 + w48 * p48
    ci = c_index(time, event, risk)
    b24 = brier_at(time, event, p24, 24)
    b48 = brier_at(time, event, p48, 48)
    weighted_brier = w24 * b24 + w48 * b48
    hybrid = 0.3 * ci + 0.7 * (1 - weighted_brier)
    return hybrid, ci, weighted_brier


def enforce_monotonicity(probs):
    out = np.clip(probs.copy(), 0, 1)
    for i in range(1, out.shape[1]):
        out[:, i] = np.maximum(out[:, i], out[:, i - 1])
    return out


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------
def _read_csv_flexible(source):
    return pd.read_csv(source, sep=None, engine="python")


def load_data(uploaded_file=None, candidates=("data/train.csv", "train.csv")):
    if uploaded_file is not None:
        raw = uploaded_file.read()
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        df = _read_csv_flexible(io.StringIO(text))
    else:
        df = None
        for path in candidates:
            if os.path.exists(path):
                df = _read_csv_flexible(path)
                break
        if df is None:
            raise FileNotFoundError("No se encontro train.csv.")
    if TARGET not in df.columns:
        raise ValueError(f"Falta la columna objetivo '{TARGET}'. Sube el conjunto de "
                         f"entrenamiento. Columnas: {list(df.columns)}")
    return df


def read_prediction_file(uploaded_file):
    name = getattr(uploaded_file, "name", "").lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    raw = uploaded_file.read()
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    return _read_csv_flexible(io.StringIO(text))


# ---------------------------------------------------------------------------
# Entrenamiento con validacion cruzada (FUENTE UNICA para metricas Y predicciones)
# Replica exactamente el notebook: promedia los modelos de los N_FOLDS pliegues.
# ---------------------------------------------------------------------------
def fit_cv(df: pd.DataFrame):
    """
    Entrena el ensamble con validacion cruzada estratificada optimizada para web. Guarda TODOS los
    modelos de todos los pliegues por horizonte, de modo que la prediccion (en la
    app o para predicciones.csv) sea el promedio sobre pliegues y modelos,
    identico a `test_preds` del notebook.
    """
    train_fe = engineer_features(df)
    feature_cols = get_feature_cols(train_fe)
    X = train_fe[feature_cols].values.astype(np.float32)
    time_arr = train_fe[TIME_COL].values
    event_arr = train_fe[TARGET].values.astype(int)
    targets, masks = build_survival_targets(time_arr, event_arr, HORIZONS)

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    strat = ((event_arr == 1) & (time_arr <= 24)).astype(int)

    # Modelos almacenados por horizonte (lista de todos los modelos de todos los pliegues)
    horizon_models = {H: [] for H in HORIZONS}
    horizon_constant = {H: None for H in HORIZONS}
    oof = np.full((len(df), len(HORIZONS)), np.nan)
    fold_scores = []

    # Posiciones de 24h y 48h dentro de HORIZONS.
    # Esto evita errores si se retiró 72h y ya no existe la columna índice 3.
    h_idx = {H: i for i, H in enumerate(HORIZONS)}
    idx24 = h_idx[24]
    idx48 = h_idx[48]

    for tr_idx, va_idx in cv.split(X, strat):
        for h_i, H in enumerate(HORIZONS):
            y = targets[H]
            ytr_full, yva_full = y[tr_idx], y[va_idx]
            tr_ok, va_ok = ~np.isnan(ytr_full), ~np.isnan(yva_full)

            if tr_ok.sum() < 5 or va_ok.sum() < 3:
                oof[va_idx, h_i] = np.nanmean(y[~np.isnan(y)])
                continue

            Xtr, ytr = X[tr_idx][tr_ok], ytr_full[tr_ok]
            Xva, yva = X[va_idx][va_ok], yva_full[va_ok]

            if len(np.unique(ytr)) < 2:
                # Horizonte con una sola clase -> predictor constante
                horizon_constant[H] = float(np.unique(ytr)[0])
                oof[va_idx, h_i] = horizon_constant[H]
                continue

            models = _fit_horizon_models(Xtr, ytr, Xva, yva)
            horizon_models[H].extend(models)
            oof[va_idx, h_i] = _predict_ensemble(models, X[va_idx])

        oof[va_idx] = enforce_monotonicity(oof[va_idx])
        s, _, _ = hybrid_score(time_arr[va_idx], event_arr[va_idx],
                               oof[va_idx, idx24], oof[va_idx, idx48])
        fold_scores.append(s)

    score, ci, wb = hybrid_score(time_arr, event_arr, oof[:, idx24], oof[:, idx48])
    metrics = {
        "hybrid": score, "c_index": ci, "weighted_brier": wb,
        "fold_scores": fold_scores,
        "brier_12h": brier_at(time_arr, event_arr, oof[:, 0], 12),
        "brier_24h": brier_at(time_arr, event_arr, oof[:, idx24], 24),
        "brier_48h": brier_at(time_arr, event_arr, oof[:, idx48], 48),
        "oof": oof, "time_arr": time_arr, "event_arr": event_arr,
        "horizon_valid": {H: int(masks[H].sum()) for H in HORIZONS},
        "horizon_pos": {H: int(np.nansum(targets[H])) for H in HORIZONS},
    }

    return {
        "horizon_models": horizon_models,
        "horizon_constant": horizon_constant,
        "feature_cols": feature_cols,
        "medians": train_fe[feature_cols].median(numeric_only=True),
        "raw_medians": df.median(numeric_only=True),
        "train_fe": train_fe,
        "metrics": metrics,
    }


def _predict_rows(bundle, df_fe_rows):
    feature_cols = bundle["feature_cols"]
    X = df_fe_rows[feature_cols].values.astype(np.float32)
    out = np.zeros((len(X), len(HORIZONS)))
    for h_i, H in enumerate(HORIZONS):
        models = bundle["horizon_models"][H]
        if not models:  # horizonte constante si no hay dos clases suficientes
            out[:, h_i] = bundle["horizon_constant"][H] if bundle["horizon_constant"][H] is not None else 1.0
        else:
            preds = np.zeros(len(X))
            for m in models:
                preds += m.predict_proba(X)[:, 1]
            out[:, h_i] = preds / len(models)
    return enforce_monotonicity(out)


def predict_batch(model_bundle, df_new: pd.DataFrame):
    """Predice prob_12h/24h/48h para un lote tipo test.csv."""
    df_fe = engineer_features(df_new)
    # Completar columnas faltantes con la mediana cruda antes de la ingenieria
    for col in model_bundle["raw_medians"].index:
        if col not in df_fe.columns:
            df_fe[col] = float(model_bundle["raw_medians"][col])
    probs = _predict_rows(model_bundle, df_fe)

    out = pd.DataFrame()
    if ID_COL in df_new.columns:
        out[ID_COL] = df_new[ID_COL].values
    for h_i, H in enumerate(HORIZONS):
        out[f"prob_{H}h"] = np.round(probs[:, h_i], 6)
    return out


def predict_single(model_bundle, form_values: dict):
    """
    Construye una fila cruda (medianas + valores del formulario), RECALCULA las
    variables derivadas con engineer_features y predice los 3 horizontes.
    Devuelve (probs_dict, fila_enviada_al_modelo).
    """
    raw = {c: float(model_bundle["raw_medians"].get(c, 0.0))
           for c in model_bundle["raw_medians"].index}
    for feat, val in form_values.items():
        raw[feat] = float(val)
    df_raw = pd.DataFrame([raw])
    df_fe = engineer_features(df_raw)
    probs = _predict_rows(model_bundle, df_fe)[0]
    probs_dict = {H: float(probs[h_i]) for h_i, H in enumerate(HORIZONS)}
    fila = df_fe[model_bundle["feature_cols"]].T
    return probs_dict, fila


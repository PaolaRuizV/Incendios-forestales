"""
core.py
Logica de datos y modelos para el dashboard de Prediccion de Amenaza de
Incendios Forestales. Separado de la interfaz de Streamlit para poder probarlo
independientemente.

Correcciones principales:
1) El formulario ahora expone mas variables para que exista mayor variacion.
2) Se recalculan variables derivadas cuando cambia una variable base.
3) La lectura de archivos acepta bytes, objetos UploadedFile o rutas locales.
4) Se mantiene excluida 'dist_min_ci_0_5h' para evitar fuga de datos.
"""

import io
import os
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Esquema de datos
# ---------------------------------------------------------------------------
ID_COL = "event_id"
TARGET = "event"                 # 1 = el incendio alcanzo una zona de evacuacion
TIME_COL = "time_to_hit_hours"   # horas hasta el impacto, solo para analisis

# Caracteristicas del incendio.
FEATURES = [
    "num_perimeters_0_5h", "dt_first_last_0_5h", "low_temporal_resolution_0_5h",
    "area_first_ha", "area_growth_abs_0_5h", "area_growth_rel_0_5h",
    "area_growth_rate_ha_per_h", "log1p_area_first", "log1p_growth",
    "log_area_ratio_0_5h", "relative_growth_0_5h", "radial_growth_m",
    "radial_growth_rate_m_per_h", "centroid_displacement_m", "centroid_speed_m_per_h",
    "spread_bearing_deg", "spread_bearing_sin", "spread_bearing_cos",
    "dist_min_ci_0_5h", "dist_std_ci_0_5h", "dist_change_ci_0_5h",
    "dist_slope_ci_0_5h", "closing_speed_m_per_h", "closing_speed_abs_m_per_h",
    "projected_advance_m", "dist_accel_m_per_h2", "dist_fit_r2_0_5h",
    "alignment_cos", "alignment_abs", "cross_track_component", "along_track_speed",
    "event_start_hour", "event_start_dayofweek", "event_start_month",
]

# Variables con sospecha de fuga de datos.
# IMPORTANTE:
# 'dist_min_ci_0_5h' puede separar muy fuerte las clases. Si la usas, las metricas
# pueden salir artificialmente altas. Por eso se excluye del entrenamiento.
# Si el profesor permite usarla, cambia a: LEAKAGE_SUSPECTS = []
LEAKAGE_SUSPECTS = ["dist_min_ci_0_5h"]

# Variables que se muestran en el formulario individual.
# Se agregaron mas campos para que la prediccion cambie con mayor claridad.
FORM_FEATURES = [
    ("area_first_ha", "Area inicial del incendio (ha)"),
    ("area_growth_abs_0_5h", "Crecimiento absoluto de area 0-5 h (ha)"),
    ("area_growth_rate_ha_per_h", "Tasa de crecimiento de area (ha/h)"),
    ("radial_growth_m", "Crecimiento radial acumulado (m)"),
    ("radial_growth_rate_m_per_h", "Velocidad de avance radial (m/h)"),
    ("centroid_displacement_m", "Desplazamiento del centroide (m)"),
    ("centroid_speed_m_per_h", "Velocidad del centroide (m/h)"),
    ("projected_advance_m", "Avance proyectado hacia la zona (m)"),
    ("closing_speed_m_per_h", "Velocidad de acercamiento a la zona (m/h)"),
    ("dist_change_ci_0_5h", "Cambio de distancia a la zona 0-5 h (m)"),
    ("dist_slope_ci_0_5h", "Pendiente de distancia a la zona (m/h)"),
    ("dist_accel_m_per_h2", "Aceleracion de distancia (m/h2)"),
    ("dist_fit_r2_0_5h", "Calidad del ajuste de distancia R2"),
    ("alignment_cos", "Alineacion del avance hacia la zona (-1 a 1)"),
    ("cross_track_component", "Componente transversal del avance"),
    ("along_track_speed", "Velocidad en direccion de avance"),
    ("num_perimeters_0_5h", "Cantidad de perimetros 0-5 h"),
    ("dt_first_last_0_5h", "Tiempo entre primer y ultimo perimetro (h)"),
    ("low_temporal_resolution_0_5h", "Baja resolucion temporal 0-5 h"),
    ("spread_bearing_sin", "Direccion del avance - seno"),
    ("spread_bearing_cos", "Direccion del avance - coseno"),
    ("event_start_hour", "Hora de inicio del evento (0-23)"),
    ("event_start_dayofweek", "Dia de la semana (0-6)"),
    ("event_start_month", "Mes de inicio (1-12)"),
]

LOCAL_CANDIDATES = [
    "data/train.csv",
    "train.csv",
]

LABELS_TARGET = {
    0: "No alcanzo la zona de evacuacion",
    1: "Alcanzo / en riesgo de alcanzar la zona",
}


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------
def _read_csv_flexible(source) -> pd.DataFrame:
    """Lee un CSV detectando separador: coma, punto y coma o tabulacion."""
    return pd.read_csv(source, sep=None, engine="python")


def _source_to_dataframe(source) -> pd.DataFrame:
    """
    Convierte diferentes tipos de entrada a DataFrame.
    Acepta:
    - bytes, por ejemplo uploaded_file.getvalue()
    - objetos con .read(), por ejemplo UploadedFile de Streamlit
    - rutas locales como string
    """
    if isinstance(source, (bytes, bytearray)):
        text = bytes(source).decode("utf-8")
        return _read_csv_flexible(io.StringIO(text))

    if hasattr(source, "read"):
        raw = source.read()
        text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
        return _read_csv_flexible(io.StringIO(text))

    return _read_csv_flexible(source)


def load_data(uploaded_file=None) -> pd.DataFrame:
    """
    Orden de busqueda del dataset de entrenamiento:
    1) archivo subido por el usuario en Streamlit
    2) data/train.csv
    3) train.csv
    """
    if uploaded_file is not None:
        df = _source_to_dataframe(uploaded_file)
    else:
        df = None
        for path in LOCAL_CANDIDATES:
            if os.path.exists(path):
                df = _read_csv_flexible(path)
                break

        if df is None:
            raise FileNotFoundError(
                "No se encontro train.csv. Subelo en la barra lateral o "
                "incluyelo en la carpeta data/ del repositorio."
            )

    if TARGET not in df.columns:
        raise ValueError(
            f"El archivo cargado no contiene la columna objetivo '{TARGET}'. "
            "Asegurate de subir el conjunto de ENTRENAMIENTO, no el test.csv. "
            f"Columnas detectadas: {list(df.columns)}"
        )

    return df


def read_prediction_file(uploaded_file) -> pd.DataFrame:
    """Lee un CSV, TXT o Excel de casos a predecir, como test.csv."""
    name = getattr(uploaded_file, "name", "").lower()

    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)

    if hasattr(uploaded_file, "read"):
        raw = uploaded_file.read()
    else:
        raw = uploaded_file

    text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
    return _read_csv_flexible(io.StringIO(text))


# ---------------------------------------------------------------------------
# Preprocesamiento y modelos
# ---------------------------------------------------------------------------
def build_preprocessor() -> Pipeline:
    """Imputacion por mediana + estandarizacion."""
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def get_model_definitions():
    """Define los cinco modelos que se comparan en el dashboard."""
    return {
        "Regresion Logistica": LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
        ),
        "Arbol de Decision": DecisionTreeClassifier(
            max_depth=5,
            random_state=42,
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=42,
            class_weight="balanced",
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=7,
        ),
    }


def train_and_evaluate(df: pd.DataFrame):
    """
    Entrena 5 modelos y devuelve:
    - pipelines entrenados
    - tabla de metricas
    - nombre del mejor modelo
    - medianas del entrenamiento
    - columnas usadas
    - conjunto de prueba
    """
    present = [
        c for c in FEATURES
        if c in df.columns and c not in LEAKAGE_SUSPECTS
    ]

    if not present:
        raise ValueError("No se encontraron columnas de caracteristicas validas.")

    X = df[present].apply(pd.to_numeric, errors="coerce")
    y = df[TARGET].astype(int)

    # Stratify mantiene la proporcion de clases en train y test.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    medians = X_train.median(numeric_only=True)
    model_defs = get_model_definitions()

    pipelines = {}
    rows = []

    for name, estimator in model_defs.items():
        pipe = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        try:
            y_proba = pipe.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
        except Exception:
            auc = np.nan

        rows.append({
            "Modelo": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred, zero_division=0),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": auc,
        })

        pipelines[name] = pipe

    metrics_df = (
        pd.DataFrame(rows)
        .sort_values("F1-Score", ascending=False)
        .reset_index(drop=True)
    )

    best_name = metrics_df.iloc[0]["Modelo"]
    return pipelines, metrics_df, best_name, medians, present, (X_test, y_test)


# ---------------------------------------------------------------------------
# Prediccion
# ---------------------------------------------------------------------------
def _safe_float(value, default=0.0):
    """Convierte a float evitando errores por valores vacios o no numericos."""
    try:
        if value is None or pd.isna(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _recalculate_derived_features(row: dict) -> dict:
    """
    Recalcula variables derivadas para que el formulario sea coherente.
    Sin esto, el usuario cambiaba una variable base, pero sus variables derivadas
    quedaban congeladas con la mediana y la prediccion variaba poco.
    """

    # Transformaciones logaritmicas.
    if "area_first_ha" in row and "log1p_area_first" in row:
        row["log1p_area_first"] = float(np.log1p(max(row["area_first_ha"], 0)))

    if "area_growth_abs_0_5h" in row and "log1p_growth" in row:
        row["log1p_growth"] = float(np.log1p(max(row["area_growth_abs_0_5h"], 0)))

    # Si existen area inicial y crecimiento absoluto, recalcula crecimiento relativo.
    if (
        "area_first_ha" in row
        and "area_growth_abs_0_5h" in row
        and "area_growth_rel_0_5h" in row
    ):
        base = max(abs(row["area_first_ha"]), 1e-6)
        row["area_growth_rel_0_5h"] = row["area_growth_abs_0_5h"] / base

    if (
        "area_growth_rel_0_5h" in row
        and "relative_growth_0_5h" in row
    ):
        row["relative_growth_0_5h"] = row["area_growth_rel_0_5h"]

    if (
        "area_growth_rel_0_5h" in row
        and "log_area_ratio_0_5h" in row
    ):
        # log(1 + crecimiento relativo), acotado para evitar log de negativo.
        ratio = max(1.0 + row["area_growth_rel_0_5h"], 1e-6)
        row["log_area_ratio_0_5h"] = float(np.log(ratio))

    # Valores absolutos derivados.
    if "alignment_cos" in row and "alignment_abs" in row:
        row["alignment_abs"] = abs(row["alignment_cos"])

    if "closing_speed_m_per_h" in row and "closing_speed_abs_m_per_h" in row:
        row["closing_speed_abs_m_per_h"] = abs(row["closing_speed_m_per_h"])

    # Tasas derivadas si el tiempo esta disponible.
    if (
        "area_growth_abs_0_5h" in row
        and "dt_first_last_0_5h" in row
        and "area_growth_rate_ha_per_h" in row
        and row["dt_first_last_0_5h"] > 0
    ):
        row["area_growth_rate_ha_per_h"] = (
            row["area_growth_abs_0_5h"] / row["dt_first_last_0_5h"]
        )

    if (
        "radial_growth_m" in row
        and "dt_first_last_0_5h" in row
        and "radial_growth_rate_m_per_h" in row
        and row["dt_first_last_0_5h"] > 0
    ):
        row["radial_growth_rate_m_per_h"] = (
            row["radial_growth_m"] / row["dt_first_last_0_5h"]
        )

    if (
        "centroid_displacement_m" in row
        and "dt_first_last_0_5h" in row
        and "centroid_speed_m_per_h" in row
        and row["dt_first_last_0_5h"] > 0
    ):
        row["centroid_speed_m_per_h"] = (
            row["centroid_displacement_m"] / row["dt_first_last_0_5h"]
        )

    return row


def _row_from_form(form_values: dict, medians: pd.Series, present_features):
    """
    Construye una fila completa para prediccion individual.
    Empieza con medianas y reemplaza con los valores ingresados por el usuario.
    """
    row = {
        feat: _safe_float(medians.get(feat, 0.0), 0.0)
        for feat in present_features
    }

    for feat, val in form_values.items():
        if feat in row:
            row[feat] = _safe_float(val, row[feat])

    row = _recalculate_derived_features(row)
    return pd.DataFrame([row])[present_features]


def predict_single(pipeline, form_values: dict, medians: pd.Series, present_features):
    """Predice un caso individual ingresado desde el formulario."""
    X_new = _row_from_form(form_values, medians, present_features)
    pred = int(pipeline.predict(X_new)[0])
    proba = pipeline.predict_proba(X_new)[0]
    return pred, float(proba[1]), float(proba[0])


def predict_batch(pipeline, df_new: pd.DataFrame, medians: pd.Series, present_features):
    """
    Predice un archivo por lote.
    Si faltan columnas, las completa con la mediana del entrenamiento.
    """
    X = pd.DataFrame(index=df_new.index)

    for feat in present_features:
        if feat in df_new.columns:
            X[feat] = pd.to_numeric(df_new[feat], errors="coerce")
        else:
            X[feat] = float(medians.get(feat, 0.0))

    X = X[present_features]

    proba = pipeline.predict_proba(X)[:, 1]
    pred = pipeline.predict(X).astype(int)

    out = pd.DataFrame()
    if ID_COL in df_new.columns:
        out[ID_COL] = df_new[ID_COL].values

    out["prob_alcanza_zona"] = np.round(proba, 4)
    out["prediccion"] = pred
    out["diagnostico"] = pd.Series(pred).map(LABELS_TARGET).values

    return out


def feature_importance(pipeline, present_features) -> pd.DataFrame | None:
    """Devuelve importancia de caracteristicas si el modelo la soporta."""
    model = pipeline.named_steps.get("model")

    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_).ravel()
    else:
        return None

    return (
        pd.DataFrame({"Caracteristica": present_features, "Importancia": imp})
        .sort_values("Importancia", ascending=False)
        .reset_index(drop=True)
    )

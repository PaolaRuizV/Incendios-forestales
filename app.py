"""
app.py - Versión web Streamlit del proyecto.

Predice probabilidades de amenaza de incendios forestales para 12h, 24h y 48h.
El núcleo del modelo está en core.py para mantener la misma lógica del notebook.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import core

# -----------------------------------------------------------------------------
# Configuración general
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Amenaza de incendios forestales",
    page_icon="🔥",
    layout="wide",
)

COLOR_MAP = {
    0: "#2E86AB",
    1: "#E63946",
    "No alcanzo la zona de evacuacion": "#2E86AB",
    "Alcanzo / en riesgo de alcanzar la zona": "#E63946",
}

HORIZON_LABELS = {12: "12 horas", 24: "24 horas", 48: "48 horas"}


# -----------------------------------------------------------------------------
# Utilidades de carga y entrenamiento
# -----------------------------------------------------------------------------
def _read_csv_flexible_from_bytes(raw: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw), sep=None, engine="python")


@st.cache_data(show_spinner="Cargando dataset de entrenamiento...")
def load_train_data(uploaded_bytes: bytes | None) -> pd.DataFrame:
    """Carga train.csv subido o el incluido en data/train.csv."""
    if uploaded_bytes is not None:
        df = _read_csv_flexible_from_bytes(uploaded_bytes)
    else:
        df = core.load_data()

    if core.TARGET not in df.columns:
        raise ValueError(
            f"El archivo de entrenamiento debe contener la columna objetivo '{core.TARGET}'. "
            "Si subiste test.csv por error, súbelo en la pestaña de predicción por lote."
        )
    return df


@st.cache_resource(show_spinner="Entrenando ensamble con validación cruzada...")
def train_model(df: pd.DataFrame):
    """Entrena LightGBM + XGBoost + CatBoost con la lógica de core.py."""
    return core.fit_cv(df)


def read_prediction_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    return pd.read_csv(uploaded_file, sep=None, engine="python")


def probability_level(prob: float) -> tuple[str, str]:
    if prob >= 0.66:
        return "Alto", "🔴"
    if prob >= 0.33:
        return "Medio", "🟡"
    return "Bajo", "🟢"


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.title("🔥 Incendios forestales")
st.sidebar.caption("Modelo predictivo")

st.sidebar.markdown("---")
st.sidebar.subheader("Fuente de datos")
train_file = st.sidebar.file_uploader(
    "Sube train.csv opcionalmente",
    type=["csv", "txt"],
    help="Si no subes un archivo, se usa data/train.csv incluido en el proyecto.",
)

if st.sidebar.button("Limpiar caché y reentrenar"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("Configuración del modelo")
st.sidebar.write(f"**Horizontes:** {', '.join(str(h) + 'h' for h in core.HORIZONS)}")
st.sidebar.write(f"**Pliegues CV:** {core.N_FOLDS}")
st.sidebar.write(f"**Versión core:** {getattr(core, 'CORE_VERSION', 'sin versión')}")

if 72 in core.HORIZONS:
    st.sidebar.error("Advertencia: 72h sigue activo en core.HORIZONS.")
else:
    st.sidebar.success("72h excluido correctamente")

# -----------------------------------------------------------------------------
# Carga de datos y entrenamiento
# -----------------------------------------------------------------------------
try:
    uploaded_bytes = train_file.getvalue() if train_file is not None else None
    train_raw = load_train_data(uploaded_bytes)
except Exception as exc:
    st.error(f"No se pudo cargar el dataset de entrenamiento: {exc}")
    st.stop()

try:
    bundle = train_model(train_raw)
    metrics = bundle["metrics"]
except Exception as exc:
    st.error("No se pudo entrenar el modelo.")
    st.exception(exc)
    st.stop()

train_fe = bundle["train_fe"]

# -----------------------------------------------------------------------------
# Encabezado
# -----------------------------------------------------------------------------
st.title("🔥 Predicción de amenaza de incendios forestales")
st.markdown(
    "Sistema inteligente de apoyo a la toma de decisiones que estima la "
    "probabilidad de que un incendio amenace una zona de evacuación en "
    "**12h, 24h y 48h**."
)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Registros de entrenamiento", f"{len(train_raw):,}")
kpi2.metric("Incendios con evento", f"{int(train_raw[core.TARGET].sum()):,}")
kpi3.metric("Puntaje híbrido", f"{metrics['hybrid']:.4f}")
kpi4.metric("Brier ponderado", f"{metrics['weighted_brier']:.4f}")

# -----------------------------------------------------------------------------
# Pestañas
# -----------------------------------------------------------------------------
tab_resumen, tab_eda, tab_individual, tab_lote, tab_tecnico = st.tabs(
    [
        "📌 Resumen del modelo",
        "📊 Análisis de datos",
        "🔮 Predicción individual",
        "📁 Predicción por lote",
        "⚙️ Detalle técnico",
    ]
)

# -----------------------------------------------------------------------------
# Resumen del modelo
# -----------------------------------------------------------------------------
with tab_resumen:
    st.header("📌 Resumen del modelo")
    st.write(
        "El modelo usa un ensamble de **LightGBM + XGBoost + CatBoost** con "
        "validación cruzada estratificada de 5 pliegues. La métrica principal "
        "combina **C-Index** y **Brier Score ponderado**"
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Puntaje híbrido", f"{metrics['hybrid']:.4f}")
    c2.metric("C-Index", f"{metrics['c_index']:.4f}")
    c3.metric("Brier ponderado", f"{metrics['weighted_brier']:.4f}")

    st.subheader("Resultados por pliegue")
    fold_df = pd.DataFrame({
        "Pliegue": [f"Fold {i + 1}" for i in range(len(metrics["fold_scores"]))],
        "Puntaje híbrido": metrics["fold_scores"],
    })
    fig_folds = px.bar(
        fold_df,
        x="Pliegue",
        y="Puntaje híbrido",
        text="Puntaje híbrido",
        title="Puntaje híbrido por pliegue de validación cruzada",
    )
    fig_folds.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_folds.update_yaxes(range=[0, 1])
    st.plotly_chart(fig_folds, use_container_width=True)

    st.subheader("Brier Score por horizonte")
    brier_rows = []
    for h in core.HORIZONS:
        key = f"brier_{h}h"
        if key in metrics:
            brier_rows.append({"Horizonte": f"{h}h", "Brier Score": metrics[key]})
    brier_df = pd.DataFrame(brier_rows)
    st.dataframe(brier_df, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# EDA
# -----------------------------------------------------------------------------
with tab_eda:
    st.header("📊 Análisis exploratorio de datos")

    df_plot = train_raw.copy()
    df_plot["Resultado"] = df_plot[core.TARGET].map(core.LABELS_TARGET)

    g1, g2 = st.columns(2)
    with g1:
        counts = df_plot["Resultado"].value_counts().reset_index()
        counts.columns = ["Resultado", "Casos"]
        fig_counts = px.bar(
            counts,
            x="Resultado",
            y="Casos",
            color="Resultado",
            color_discrete_map=COLOR_MAP,
            title="Distribución de la variable objetivo",
        )
        st.plotly_chart(fig_counts, use_container_width=True)

    with g2:
        fig_time = px.histogram(
            df_plot,
            x=core.TIME_COL,
            color="Resultado",
            nbins=25,
            barmode="overlay",
            color_discrete_map=COLOR_MAP,
            title="Distribución del tiempo hasta impacto",
            labels={core.TIME_COL: "Tiempo hasta impacto (horas)"},
        )
        st.plotly_chart(fig_time, use_container_width=True)

    g3, g4 = st.columns(2)
    with g3:
        fig_scatter = px.scatter(
            df_plot,
            x="dist_min_ci_0_5h",
            y="closing_speed_m_per_h",
            color="Resultado",
            color_discrete_map=COLOR_MAP,
            title="Distancia mínima vs. velocidad de acercamiento",
            labels={
                "dist_min_ci_0_5h": "Distancia mínima a zona (m)",
                "closing_speed_m_per_h": "Velocidad de acercamiento (m/h)",
            },
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with g4:
        fig_area = px.box(
            df_plot,
            x="Resultado",
            y="area_first_ha",
            color="Resultado",
            color_discrete_map=COLOR_MAP,
            title="Área inicial por resultado",
            labels={"area_first_ha": "Área inicial (ha)"},
        )
        st.plotly_chart(fig_area, use_container_width=True)

    st.subheader("Correlación de variables con el evento")
    numeric_cols = train_fe.select_dtypes(include=[np.number]).columns.tolist()
    corr = train_fe[numeric_cols].corr(numeric_only=True)[core.TARGET].drop(core.TARGET)
    corr_df = corr.abs().sort_values(ascending=False).head(15).reset_index()
    corr_df.columns = ["Característica", "Correlación absoluta con evento"]
    fig_corr = px.bar(
        corr_df.iloc[::-1],
        x="Correlación absoluta con evento",
        y="Característica",
        orientation="h",
        title="Top 15 variables más asociadas al evento",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    with st.expander("Ver tabla de entrenamiento"):
        st.dataframe(train_raw, use_container_width=True)

# -----------------------------------------------------------------------------
# Predicción individual
# -----------------------------------------------------------------------------
with tab_individual:
    st.header("🔮 Predicción individual")
    st.write(
        "Ingresa las características principales del incendio. Las demás variables se "
        "completan con la mediana del conjunto de entrenamiento y luego se recalculan "
        "las variables derivadas."
    )

    with st.form("individual_form"):
        values = {}
        cols = st.columns(2)
        for i, (feat, label) in enumerate(core.FORM_FEATURES):
            serie = pd.to_numeric(train_raw[feat], errors="coerce").dropna()
            lo = float(serie.min())
            hi = float(serie.max())
            med = float(serie.median())

            if feat in ["num_perimeters_0_5h", "event_start_hour"]:
                step = 1.0
            elif feat == "alignment_abs":
                step = 0.01
                lo, hi = max(0.0, lo), min(1.0, hi)
            else:
                step = float((hi - lo) / 200) if hi > lo else 0.01
                step = max(step, 0.01)

            with cols[i % 2]:
                values[feat] = st.slider(
                    label,
                    min_value=lo,
                    max_value=hi,
                    value=min(max(med, lo), hi),
                    step=step,
                )

        submitted = st.form_submit_button("Calcular probabilidad", use_container_width=True)

    if submitted:
        try:
            probs_dict, model_row = core.predict_single(bundle, values)
            prob48 = probs_dict.get(48, list(probs_dict.values())[-1])
            level, icon = probability_level(prob48)

            st.markdown("---")
            st.subheader(f"{icon} Nivel de riesgo estimado: {level}")

            cols_prob = st.columns(len(core.HORIZONS))
            for col, h in zip(cols_prob, core.HORIZONS):
                col.metric(f"Probabilidad {h}h", f"{probs_dict[h] * 100:.1f}%")

            prob_df = pd.DataFrame({
                "Horizonte": [f"{h}h" for h in core.HORIZONS],
                "Probabilidad": [probs_dict[h] for h in core.HORIZONS],
            })
            fig_prob = px.line(
                prob_df,
                x="Horizonte",
                y="Probabilidad",
                markers=True,
                title="Evolución de la probabilidad por horizonte",
            )
            fig_prob.update_yaxes(range=[0, 1], tickformat=".0%")
            st.plotly_chart(fig_prob, use_container_width=True)

            st.info(
                "Interpretación: el resultado es una estimación probabilística de apoyo "
                "a la decisión. No reemplaza la evaluación de especialistas ni información "
                "operativa en tiempo real."
            )

            with st.expander("Ver fila enviada al modelo"):
                st.dataframe(model_row, use_container_width=True)
        except Exception as exc:
            st.error(f"No se pudo calcular la predicción: {exc}")

# -----------------------------------------------------------------------------
# Predicción por lote
# -----------------------------------------------------------------------------
with tab_lote:
    st.header("📁 Predicción por lote")
    st.write(
        "Sube un archivo con formato similar a `test.csv`. Debe contener las "
        "variables de entrada del incendio, pero no necesita la columna `event`."
    )

    batch_file = st.file_uploader(
        "Archivo para predecir",
        type=["csv", "txt", "xlsx"],
        key="batch_file",
    )

    use_sample = st.checkbox("Usar data/test.csv incluido como ejemplo", value=batch_file is None)

    df_new = None
    try:
        if batch_file is not None:
            df_new = read_prediction_file(batch_file)
        elif use_sample and Path("data/test.csv").exists():
            df_new = pd.read_csv("data/test.csv")
    except Exception as exc:
        st.error(f"No se pudo leer el archivo de predicción: {exc}")

    if df_new is not None:
        try:
            result = core.predict_batch(bundle, df_new)
            result = result.drop(columns=["prob_72h"], errors="ignore")

            st.success(f"Predicciones generadas para {len(result):,} registros.")
            c1, c2, c3 = st.columns(3)
            c1.metric("Promedio 12h", f"{result['prob_12h'].mean() * 100:.1f}%")
            c2.metric("Promedio 24h", f"{result['prob_24h'].mean() * 100:.1f}%")
            c3.metric("Promedio 48h", f"{result['prob_48h'].mean() * 100:.1f}%")

            st.dataframe(result, use_container_width=True)

            csv_bytes = result.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Descargar predicciones.csv",
                data=csv_bytes,
                file_name="predicciones.csv",
                mime="text/csv",
                use_container_width=True,
            )

            fig_dist = px.histogram(
                result,
                x="prob_48h",
                nbins=20,
                title="Distribución de probabilidades a 48h",
                labels={"prob_48h": "Probabilidad 48h"},
            )
            fig_dist.update_xaxes(tickformat=".0%")
            st.plotly_chart(fig_dist, use_container_width=True)

        except Exception as exc:
            st.error("No se pudieron generar predicciones. Verifica que el archivo tenga las columnas necesarias.")
            st.exception(exc)
    else:
        st.info("Sube un archivo o activa el uso de `data/test.csv` incluido.")

# -----------------------------------------------------------------------------
# Detalle técnico
# -----------------------------------------------------------------------------
with tab_tecnico:
    st.header("⚙️ Detalle técnico")

    st.subheader("Horizontes considerados")
    st.code(f"core.HORIZONS = {core.HORIZONS}")


    st.subheader("Validez y positivos por horizonte")
    horizon_df = pd.DataFrame({
        "Horizonte": [f"{h}h" for h in core.HORIZONS],
        "Casos válidos": [metrics["horizon_valid"].get(h, 0) for h in core.HORIZONS],
        "Casos positivos": [metrics["horizon_pos"].get(h, 0) for h in core.HORIZONS],
    })
    st.dataframe(horizon_df, use_container_width=True, hide_index=True)

    st.subheader("Variables derivadas principales")
    derived = pd.DataFrame({
        "Variable": [
            "time_to_contact",
            "log_time_to_contact",
            "danger_vector",
            "tracking_urgency",
            "fire_intensity",
            "approach_momentum",
            "log_dist",
            "dist_zone_critical",
            "dist_zone_mid",
            "speed_per_km",
        ],
        "Descripción": [
            "Tiempo estimado para contactar la zona según distancia y velocidad.",
            "Transformación logarítmica del tiempo de contacto.",
            "Combinación entre alineación del avance y velocidad de acercamiento.",
            "Urgencia de seguimiento según número de perímetros y velocidad.",
            "Intensidad aproximada del crecimiento del incendio.",
            "Momentum del avance hacia la zona ajustado por distancia.",
            "Distancia mínima transformada con logaritmo.",
            "Indicador de distancia crítica menor a 5 km.",
            "Indicador de distancia media entre 5 km y 15 km.",
            "Velocidad normalizada por kilómetro de distancia.",
        ],
    })
    st.dataframe(derived, use_container_width=True, hide_index=True)

    st.subheader("Columnas usadas por el modelo")
    st.write(f"Total de características: {len(bundle['feature_cols'])}")
    st.code("\n".join(bundle["feature_cols"]))

st.markdown("---")
st.caption(
    "Dashboard académico desarrollado con Streamlit. Modelo probabilístico para apoyo a la toma de decisiones; "
    "no reemplaza evaluación técnica especializada ni datos operativos en tiempo real."
)

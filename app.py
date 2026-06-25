"""
app.py - Dashboard interactivo de Prediccion de Amenaza de Incendios Forestales

Panel A: Analisis de Datos, con filtros y KPIs.
Panel B: Analisis Predictivo, con formulario individual y prediccion por lote.

Para correr localmente:
    streamlit run app.py

Para desplegar en Streamlit Community Cloud:
    1) Sube app.py, core.py, requirements.txt y data/train.csv al repositorio.
    2) En Streamlit Community Cloud selecciona app.py como archivo principal.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

import core

st.set_page_config(
    page_title="Incendios Forestales - Dashboard",
    page_icon="🔥",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Cargando dataset de incendios...")
def get_data(uploaded_bytes=None):
    """Carga el dataset de entrenamiento. Se usan bytes para evitar errores de cache."""
    return core.load_data(uploaded_bytes)


@st.cache_resource(show_spinner="Entrenando y comparando 5 modelos...")
def get_models(df: pd.DataFrame):
    """Entrena modelos y guarda el resultado en cache."""
    return core.train_and_evaluate(df)


COLOR_MAP = {
    core.LABELS_TARGET[0]: "#2E86AB",   # azul = no alcanzo
    core.LABELS_TARGET[1]: "#E63946",   # rojo = alcanzo / en riesgo
}


# ---------------------------------------------------------------------------
# Sidebar: fuente de datos
# ---------------------------------------------------------------------------
st.sidebar.title("🔥 Incendios Forestales")
st.sidebar.markdown("---")
st.sidebar.subheader("Fuente de datos")

uploaded = st.sidebar.file_uploader(
    "Sube train.csv (opcional). Si no, se usa el incluido en data/.",
    type=["csv", "txt"],
)

# Boton util cuando cambias train.csv o haces pruebas.
if st.sidebar.button("🔄 Reiniciar cache / reentrenar"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

try:
    uploaded_bytes = uploaded.getvalue() if uploaded is not None else None
    df = get_data(uploaded_bytes)
except Exception as e:
    st.error(
        "No se pudo cargar el dataset de entrenamiento. Sube train.csv en la "
        f"barra lateral. Detalle: {e}"
    )
    st.stop()

try:
    pipelines, metrics_df, best_model_name, medians, present_features, test_data = get_models(df)
    X_test, y_test = test_data
    best_pipeline = pipelines[best_model_name]
except Exception as e:
    st.error(f"No se pudo entrenar el modelo. Detalle: {e}")
    st.stop()


tab_a, tab_b = st.tabs(
    ["📊 Panel A - Analisis de Datos", "🔮 Panel B - Analisis Predictivo"]
)


# ===========================================================================
# PANEL A - ANALISIS DE DATOS
# ===========================================================================
with tab_a:
    st.header("📊 Panel A - Analisis de Datos")
    st.caption("Explora el dataset de incendios forestales de forma interactiva.")

    has_time = core.TIME_COL in df.columns

    with st.sidebar:
        st.markdown("---")
        st.subheader("Filtros - Panel A")

        diag_opt = st.selectbox(
            "Resultado",
            ["Todos", core.LABELS_TARGET[1], core.LABELS_TARGET[0]],
        )

        if "area_first_ha" not in df.columns:
            st.error("El dataset no contiene la columna area_first_ha.")
            st.stop()

        area_min = float(df["area_first_ha"].min())
        area_max = float(df["area_first_ha"].max())
        rango_area = st.slider(
            "Area inicial (ha)",
            area_min,
            area_max,
            (area_min, area_max),
        )

        if has_time:
            t = df[core.TIME_COL].dropna()
            t_min, t_max = float(t.min()), float(t.max())
            rango_t = st.slider(
                "Tiempo hasta impacto (horas)",
                t_min,
                t_max,
                (t_min, t_max),
            )

    df_f = df[
        (df["area_first_ha"] >= rango_area[0])
        & (df["area_first_ha"] <= rango_area[1])
    ].copy()

    if diag_opt != "Todos":
        cod = 1 if diag_opt == core.LABELS_TARGET[1] else 0
        df_f = df_f[df_f[core.TARGET] == cod]

    if has_time:
        df_f = df_f[
            (df_f[core.TIME_COL] >= rango_t[0])
            & (df_f[core.TIME_COL] <= rango_t[1])
        ]

    if len(df_f) == 0:
        st.warning("No hay registros que cumplan con los filtros seleccionados.")
        st.stop()

    df_plot = df_f.copy()
    df_plot["Resultado"] = df_plot[core.TARGET].map(core.LABELS_TARGET)

    # --- KPIs ---
    c1, c2, c3 = st.columns(3)
    c1.metric("Registros filtrados", f"{len(df_f)} / {len(df)}")
    c2.metric("% que alcanzo la zona", f"{(df_f[core.TARGET].mean() * 100):.1f}%")

    if has_time:
        hit_times = df_f.loc[df_f[core.TARGET] == 1, core.TIME_COL]
        prom = hit_times.mean() if len(hit_times) else float("nan")
        c3.metric("Tiempo prom. hasta impacto", f"{prom:.1f} h" if prom == prom else "-")
    else:
        c3.metric("Area inicial promedio", f"{df_f['area_first_ha'].mean():.1f} ha")

    st.markdown("---")

    g1, g2 = st.columns(2)

    with g1:
        x_var = core.TIME_COL if has_time else "area_first_ha"
        x_lab = "Tiempo hasta impacto (h)" if has_time else "Area inicial (ha)"
        fig1 = px.histogram(
            df_plot,
            x=x_var,
            color="Resultado",
            nbins=20,
            barmode="overlay",
            title=f"Distribucion de {x_lab.lower()} por resultado",
            labels={x_var: x_lab},
            color_discrete_map=COLOR_MAP,
        )
        st.plotly_chart(fig1, use_container_width=True)

    with g2:
        if {"dist_min_ci_0_5h", "closing_speed_m_per_h"}.issubset(df_plot.columns):
            fig2 = px.scatter(
                df_plot,
                x="dist_min_ci_0_5h",
                y="closing_speed_m_per_h",
                color="Resultado",
                title="Distancia a la zona vs. velocidad de acercamiento",
                labels={
                    "dist_min_ci_0_5h": "Distancia minima a la zona (m)",
                    "closing_speed_m_per_h": "Velocidad de acercamiento (m/h)",
                },
                color_discrete_map=COLOR_MAP,
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No hay columnas suficientes para el grafico de dispersion.")

    g3, g4 = st.columns(2)

    with g3:
        counts = df_plot["Resultado"].value_counts().reset_index()
        counts.columns = ["Resultado", "Casos"]
        fig3 = px.bar(
            counts,
            x="Resultado",
            y="Casos",
            color="Resultado",
            title="Distribucion de la variable objetivo",
            color_discrete_map=COLOR_MAP,
        )
        st.plotly_chart(fig3, use_container_width=True)

    with g4:
        top_feats = [
            "area_first_ha",
            "area_growth_rate_ha_per_h",
            "radial_growth_rate_m_per_h",
            "dist_min_ci_0_5h",
            "closing_speed_m_per_h",
            "centroid_speed_m_per_h",
            "alignment_cos",
        ]
        top_feats = [c for c in top_feats if c in df_f.columns]

        if len(top_feats) >= 2:
            corr = df_f[top_feats + [core.TARGET]].corr(numeric_only=True)
            fig4 = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="Correlaciones entre caracteristicas clave y el objetivo",
            )
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("No hay columnas suficientes para calcular correlaciones.")

    with st.expander("Ver tabla de datos filtrados"):
        st.dataframe(df_f, use_container_width=True)


# ===========================================================================
# PANEL B - ANALISIS PREDICTIVO
# ===========================================================================
with tab_b:
    st.header("🔮 Panel B - Analisis Predictivo")

    best_f1 = metrics_df.set_index("Modelo").loc[best_model_name, "F1-Score"]
    st.caption(
        f"Modelo activo: **{best_model_name}** "
        f"(F1-Score = {best_f1:.3f} en el conjunto de prueba)"
    )

    modo = st.radio(
        "Modo de prediccion",
        ["Caso individual (formulario)", "Por lote (subir archivo)"],
        horizontal=True,
    )

    # -----------------------------------------------------------------------
    # Caso individual
    # -----------------------------------------------------------------------
    if modo == "Caso individual (formulario)":
        with st.form("formulario_prediccion"):
            st.subheader("Ingresa las caracteristicas del incendio")
            st.caption(
                "Ahora el formulario incluye mas variables y recalcula variables "
                "derivadas para que los resultados tengan mayor variacion."
            )

            form_values = {}
            cols = st.columns(2)

            for i, (feat, label) in enumerate(core.FORM_FEATURES):
                if feat not in df.columns or feat not in present_features:
                    continue

                serie = pd.to_numeric(df[feat], errors="coerce").dropna()
                if serie.empty:
                    continue

                lo = float(serie.min())
                hi = float(serie.max())
                med = float(serie.median())

                # Para columnas casi binarias o enteras pequenas se usa paso 1.
                unique_count = serie.nunique()
                is_integer_like = (serie.dropna() % 1 == 0).all()

                if is_integer_like and unique_count <= 30:
                    step = 1.0
                else:
                    step = (hi - lo) / 200 if hi > lo else 0.1

                with cols[i % 2]:
                    form_values[feat] = st.slider(
                        label,
                        min_value=lo,
                        max_value=hi,
                        value=med,
                        step=step,
                    )

            submitted = st.form_submit_button("🔍 Predecir", use_container_width=True)

        if submitted:
            pred, prob_pos, prob_neg = core.predict_single(
                best_pipeline,
                form_values,
                medians,
                present_features,
            )

            st.markdown("---")
            r1, r2 = st.columns([1, 1])

            with r1:
                if pred == 1:
                    st.error(f"### ⚠️ {core.LABELS_TARGET[1]}")
                    st.progress(prob_pos)
                    st.metric("Probabilidad de alcanzar la zona", f"{prob_pos * 100:.1f}%")
                else:
                    st.success(f"### ✅ {core.LABELS_TARGET[0]}")
                    st.progress(prob_neg)
                    st.metric("Probabilidad de NO alcanzar la zona", f"{prob_neg * 100:.1f}%")

            with r2:
                st.info(
                    "**Como interpretar este resultado**\n\n"
                    + (
                        "Segun las caracteristicas ingresadas, el modelo estima "
                        "un patron similar al de incendios que SI alcanzaron una "
                        "zona de evacuacion. Es una estimacion estadistica de apoyo "
                        "a la decision, no una prediccion definitiva."
                        if pred == 1
                        else
                        "Segun las caracteristicas ingresadas, el modelo estima "
                        "un patron mas similar al de incendios que NO alcanzaron "
                        "una zona de evacuacion. Esto no elimina el riesgo, porque "
                        "las condiciones pueden cambiar."
                    )
                )

            with st.expander("Ver valores usados en la prediccion"):
                st.dataframe(pd.DataFrame([form_values]), use_container_width=True)

    # -----------------------------------------------------------------------
    # Prediccion por lote
    # -----------------------------------------------------------------------
    else:
        st.subheader("Prediccion por lote")
        st.caption(
            "Sube un archivo con el mismo formato que test.csv. Puede tener event_id "
            "y las caracteristicas del incendio, pero no necesita la columna event."
        )

        pred_file = st.file_uploader(
            "Archivo de casos a predecir",
            type=["csv", "txt", "xlsx"],
            key="batch",
        )

        if pred_file is not None:
            try:
                df_new = core.read_prediction_file(pred_file)
                resultado = core.predict_batch(
                    best_pipeline,
                    df_new,
                    medians,
                    present_features,
                )

                st.success(f"Se generaron predicciones para {len(resultado)} incendios.")

                k1, k2 = st.columns(2)
                k1.metric("En riesgo (prediccion = 1)", int(resultado["prediccion"].sum()))
                k2.metric(
                    "Prob. promedio de alcanzar zona",
                    f"{resultado['prob_alcanza_zona'].mean() * 100:.1f}%",
                )

                st.dataframe(resultado, use_container_width=True)

                st.download_button(
                    "⬇️ Descargar predicciones (CSV)",
                    resultado.to_csv(index=False).encode("utf-8"),
                    file_name="predicciones.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            except Exception as e:
                st.error(f"No se pudo procesar el archivo. Detalle: {e}")

    # -----------------------------------------------------------------------
    # Comparacion de modelos e importancia
    # -----------------------------------------------------------------------
    with st.expander("📈 Ver comparacion de los 5 modelos entrenados"):
        st.dataframe(
            metrics_df.style.format({
                "Accuracy": "{:.4f}",
                "F1-Score": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "ROC-AUC": "{:.4f}",
            }),
            use_container_width=True,
        )

        st.caption(
            f"Se selecciono **{best_model_name}** como modelo final por su mayor "
            "F1-Score en el conjunto de prueba. Los modelos usan class_weight "
            "balanceado cuando el algoritmo lo permite. Se mantiene excluida "
            "la variable 'dist_min_ci_0_5h' por posible fuga de datos; por eso "
            "puede aparecer en graficos, pero no en el entrenamiento."
        )

    imp = core.feature_importance(best_pipeline, present_features)
    if imp is not None:
        with st.expander("🌲 Ver importancia de caracteristicas del modelo final"):
            fig_imp = px.bar(
                imp.head(15).iloc[::-1],
                x="Importancia",
                y="Caracteristica",
                orientation="h",
                title="Top 15 caracteristicas mas influyentes",
            )
            st.plotly_chart(fig_imp, use_container_width=True)


st.markdown("---")
st.caption(
    "Dashboard desarrollado con Streamlit - Proyecto de prediccion de amenaza de "
    "incendios forestales - GrupoSixpack (uso academico)"
)

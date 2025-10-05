# app.py
# Streamlit app para cargar ventas/servicios mensuales del Taller de Moto,
# explorar datos y pronosticar la DEMANDA por servicio.
# Autor: ChatGPT (Naty project)

import io
import json
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Modelado
# statsmodels es opcional; si no está, hacemos un fallback simple
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAVE_STATSMODELS = True
except Exception:
    ExponentialSmoothing = None
    HAVE_STATSMODELS = False
# ARIMA opcional (pmdarima puede fallar en algunos entornos sin wheels)
try:
    from pmdarima import auto_arima
    HAVE_PMDARIMA = True
except Exception:
    HAVE_PMDARIMA = False

# Visualización
import plotly.express as px

# =============================
# Configuración básica UI
# =============================
st.set_page_config(page_title="Pronóstico de Demanda – Taller de Moto", layout="wide")
st.title("🔧📈 Pronóstico de Demanda – Taller de Moto")

# === Health check / diagnóstico rápido ===
with st.expander("🩺 Diagnóstico del entorno", expanded=False):
    import sys, platform
    st.write("Python:", sys.version)
    st.write("Platform:", platform.platform())
    try:
        import pandas as _pd; st.write("pandas", _pd.__version__)
    except Exception as e:
        st.write("pandas no disponible", e)
    try:
        import numpy as _np; st.write("numpy", _np.__version__)
    except Exception as e:
        st.write("numpy no disponible", e)
    try:
        import plotly as _plotly; st.write("plotly", _plotly.__version__)
    except Exception as e:
        st.write("plotly no disponible", e)
    st.write("statsmodels disponible:", 'Sí' if 'HAVE_STATSMODELS' in globals() and HAVE_STATSMODELS else 'No')

# Captura global de errores de la app
class SafeBlock:
    def __init__(self, label):
        self.label = label
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        if exc is not None:
            st.error(f"Ocurrió un error en: {self.label}")
            st.exception(exc)
        return True  # evita que caiga la app completa

st.markdown(
    """
Esta app permite **subir un archivo** (Excel/CSV) con la siguiente estructura mínima:

- **fecha** (idealmente mensual; puede ser diaria y se agregará a mes)  
- **servicio** (nombre del servicio)  
- **cantidad** (número de servicios realizados)  
- **precio** (precio unitario promedio o de venta por registro)

Luego podrás **explorar** y **pronosticar** por servicio usando modelos como **Holt‑Winters (ETS)** y **Auto-ARIMA**, con selección automática del mejor.
    """
)

# =============================
# Taxonomía de servicios (aportada por el usuario)
# =============================
SERVICIO_TIPO = {
    "Lavado y detallado de motos": "Lavado",
    "Lavado aspirado de carros": "Lavado",
    "Encerado": "Lavado",
    "Detallado interior": "Lavado",
    "Detallado de partes negras": "Lavado",
    "Pulido de parabrisas y ventanas": "Lavado",
    "Tratamiento antiempaño": "Lavado",
    "Pulido de focos": "Lavado",
    "Cambio y tensión de cadena": "Reparación",
    "Balanceo de llantas": "Reparación",
    "Mantenimiento de suspensión": "Reparación",
    "Revisión y ajuste de frenos": "Reparación",
    "Cambio de aceite": "Reparación",
    "Ajuste de carburador": "Reparación",
    "Limpieza mecánica general y engrase": "Reparación",
    "Revisión eléctrica": "Reparación",
    "Reparación/Cambio de motor": "Reparación",
    "Cambio de cloch": "Reparación",
    "Soldadura y enderezado": "Reparación",
    "Cambio de mufla": "Reparación",
    "Revisión técnica previa a Dekra": "Reparación",
    "Reparación/Cambio de transmisión": "Reparación",
    "Rectificación/Reemplazo de tubones": "Reparación",
    "Instalación de accesorios": "Reparación",
}

# =============================
# Utilidades
# =============================

def _try_parse_date(s: pd.Series) -> pd.Series:
    """Intenta parsear fechas en diferentes formatos (YYYY-MM, dd/mm/yyyy, etc.)."""
    # Primero intentos comunes
    parsed = pd.to_datetime(s, errors="coerce", infer_datetime_format=True, utc=False)
    # Si vienen como 'ene-2024' en español, intentar un parse manual
    if parsed.isna().any():
        # Reemplazar meses en español por número
        meses = {
            "ene": "01", "feb": "02", "mar": "03", "abr": "04", "may": "05", "jun": "06",
            "jul": "07", "ago": "08", "sep": "09", "oct": "10", "nov": "11", "dic": "12",
        }
        def repl(x: str):
            if not isinstance(x, str):
                return x
            y = x.strip().lower()
            for k, v in meses.items():
                y = y.replace(f"{k}-", f"{v}-")
            return y
        parsed2 = pd.to_datetime(s.apply(repl), errors="coerce", dayfirst=True)
        parsed = parsed.fillna(parsed2)
    return parsed


def coerce_schema(df: pd.DataFrame, cols: dict) -> pd.DataFrame:
    """Normaliza el esquema a columnas estándar: fecha, servicio, cantidad, precio.
    - Detecta granularidad diaria y agrega a MES.
    - Completa meses faltantes por servicio con 0 (cantidad) y NaN (precio) y luego
      usa forward-fill del precio.
    """
    df = df.copy()
    # Renombrar
    df = df.rename(columns={
        cols["fecha"]: "fecha",
        cols["servicio"]: "servicio",
        cols["cantidad"]: "cantidad",
        cols["precio"]: "precio",
    })

    # Parseo de fecha
    df["fecha"] = _try_parse_date(df["fecha"]) 
    if df["fecha"].isna().any():
        raise ValueError("No se pudieron parsear algunas fechas. Revisá el formato de la columna 'fecha'.")

    # Tipos
    df["servicio"] = df["servicio"].astype(str).str.strip()
    df["cantidad"] = pd.to_numeric(df["cantidad"], errors="coerce").fillna(0.0)
    df["precio"] = pd.to_numeric(df["precio"], errors="coerce")

    # Agregar a mensual si la granularidad es diaria
    # Regla: si hay más de 20 fechas únicas por servicio-año, asumimos diario y agregamos a mes
    df["año_mes"] = df["fecha"].dt.to_period("M").dt.to_timestamp()
    tmp = df.groupby(["servicio", "año_mes"], as_index=False).agg({
        "cantidad": "sum",
        "precio": "mean",  # precio promedio mensual
    })
    tmp = tmp.rename(columns={"año_mes": "fecha"})

    # Completar meses faltantes por servicio
    full = []
    for svc, g in tmp.groupby("servicio"):
        # Rango completo de meses para ese servicio
        idx = pd.date_range(g["fecha"].min(), g["fecha"].max(), freq="MS")
        g2 = g.set_index("fecha").reindex(idx).rename_axis("fecha").reset_index()
        g2["servicio"] = svc
        g2["cantidad"] = g2["cantidad"].fillna(0.0)
        # Propagar precio hacia adelante/atrás
        g2["precio"] = g2["precio"].ffill().bfill()
        full.append(g2)
    dfm = pd.concat(full, ignore_index=True)

    # Añadir tipo de servicio si aplica
    dfm["tipo_servicio"] = dfm["servicio"].map(SERVICIO_TIPO)

    # KPIs derivados
    dfm["ingreso"] = dfm["cantidad"] * dfm["precio"].fillna(0)

    return dfm[["fecha", "servicio", "tipo_servicio", "cantidad", "precio", "ingreso"]]


# =============================
# Modelos y backtesting
# =============================

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom = np.where(denom == 0, 1.0, denom)
    return 100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denom)


def fit_ets(series: pd.Series, seasonal_periods: int = 12):
    """Intenta ETS (Holt-Winters). Si statsmodels no está disponible, usa Holt lineal simple como fallback."""
    if HAVE_STATSMODELS:
        model = ExponentialSmoothing(
            series,
            trend="add",
            seasonal="add",
            seasonal_periods=seasonal_periods,
            initialization_method="estimated",
        )
        res = model.fit(optimized=True)
        return res
    else:
        # Fallback: implementamos Holt lineal básico
        alpha, beta = 0.3, 0.1
        y = series
        level = y.iloc[0]
        trend = (y.iloc[1] - y.iloc[0]) if len(y) > 1 else 0.0
        for t in range(1, len(y)):
            prev_level = level
            level = alpha * y.iloc[t] + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
        class _Res:
            def __init__(self, l, b):
                self._l = l; self._b = b
                self.resid = np.array([])  # no resid disponible
            def forecast(self, h):
                return pd.Series([self._l + (i+1)*self._b for i in range(h)],
                                 index=pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS"))
        return _Res(level, trend)


def fit_auto_arima(series: pd.Series, seasonal_periods: int = 12):
    if not HAVE_PMDARIMA:
        raise RuntimeError("pmdarima no disponible en este entorno")
    model = auto_arima(
        series,
        seasonal=True,
        m=seasonal_periods,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
        trace=False,
    )
    return model


def time_series_cv(series: pd.Series, horizon: int = 3, initial: int = 24, step: int = 1) -> dict:
    """Backtesting simple expandiendo ventana.
    Retorna métricas para ETS y (si está disponible) ARIMA y elige el mejor por sMAPE.
    """
    y = series.dropna()
    if len(y) < max(initial, horizon) + 3:
        return {"best": None, "ets": None, "arima": None}

    ets_preds, arima_preds, y_trues = [], [], []

    for end in range(initial, len(y) - horizon + 1, step):
        train = y.iloc[:end]
        test = y.iloc[end:end + horizon]

        # ETS
        try:
            ets_fit = fit_ets(train)
            f_ets = ets_fit.forecast(horizon)
        except Exception:
            f_ets = pd.Series([np.nan] * horizon, index=test.index)

        # ARIMA (solo si disponible)
        if HAVE_PMDARIMA:
            try:
                arima_fit = fit_auto_arima(train)
                f_ari = pd.Series(arima_fit.predict(n_periods=horizon), index=test.index)
            except Exception:
                f_ari = pd.Series([np.nan] * horizon, index=test.index)
        else:
            f_ari = pd.Series([np.nan] * horizon, index=test.index)

        ets_preds.append(f_ets.values)
        arima_preds.append(f_ari.values)
        y_trues.append(test.values)

    ets_preds = np.concatenate(ets_preds)
    arima_preds = np.concatenate(arima_preds)
    y_trues = np.concatenate(y_trues)

    mets = smape(y_trues, ets_preds)
    mari = smape(y_trues, arima_preds) if HAVE_PMDARIMA else np.inf

    best = "ETS" if mets <= mari else "ARIMA"
    return {
        "best": best,
        "ets": {"sMAPE": mets},
        "arima": ({"sMAPE": mari} if HAVE_PMDARIMA else None),
    }"best": None, "ets": None, "arima": None}

    ets_preds, arima_preds, y_trues = [], [], []

    for end in range(initial, len(y) - horizon + 1, step):
        train = y.iloc[:end]
        test = y.iloc[end:end + horizon]

        # ETS
        try:
            ets_fit = fit_ets(train)
            f_ets = ets_fit.forecast(horizon)
        except Exception:
            f_ets = pd.Series([np.nan] * horizon, index=test.index)

        # ARIMA
        try:
            arima_fit = fit_auto_arima(train)
            f_ari = pd.Series(arima_fit.predict(n_periods=horizon), index=test.index)
        except Exception:
            f_ari = pd.Series([np.nan] * horizon, index=test.index)

        ets_preds.append(f_ets.values)
        arima_preds.append(f_ari.values)
        y_trues.append(test.values)

    ets_preds = np.concatenate(ets_preds)
    arima_preds = np.concatenate(arima_preds)
    y_trues = np.concatenate(y_trues)

    mets = smape(y_trues, ets_preds)
    mari = smape(y_trues, arima_preds)

    best = "ETS" if mets <= mari else "ARIMA"
    return {
        "best": best,
        "ets": {"sMAPE": mets},
        "arima": {"sMAPE": mari},
    }


def fit_and_forecast(series: pd.Series, horizon: int = 6, seasonal_periods: int = 12) -> Tuple[pd.Series, pd.DataFrame]:
    metrics = time_series_cv(series, horizon=min(horizon, 6), initial=min(24, max(12, len(series)//2)))
    best = metrics.get("best") if metrics else None

    if best == "ARIMA" and HAVE_PMDARIMA:
        model = fit_auto_arima(series, seasonal_periods)
        fc = model.predict(n_periods=horizon)
        pred_idx = pd.period_range(series.index[-1].to_period("M").to_timestamp(), periods=horizon+1, freq="MS")[1:]
        df_fc = pd.DataFrame({"fecha": pred_idx, "yhat": fc})
    else:
        # ETS por defecto o fallback Holt
        fit = fit_ets(series, seasonal_periods)
        fc = fit.forecast(horizon)
        if hasattr(fit, 'resid') and isinstance(fit.resid, (np.ndarray, list)) and len(getattr(fit,'resid',[]))>0:
            ci_low = fc - 1.96 * np.std(fit.resid, ddof=1)
            ci_hi = fc + 1.96 * np.std(fit.resid, ddof=1)
            df_fc = pd.DataFrame({
                "fecha": fc.index,
                "yhat": fc.values,
                "yhat_lower": ci_low.values,
                "yhat_upper": ci_hi.values,
            })
        else:
            df_fc = pd.DataFrame({
                "fecha": fc.index,
                "yhat": fc.values,
            })
    return metrics, df_fc


# =============================
# SIDEBAR – Carga y opciones
# =============================
st.sidebar.header("1) Cargar archivo")
file = st.sidebar.file_uploader("Subí un Excel (.xlsx) o CSV", type=["xlsx", "xls", "csv"]) 

if file is not None:
    if file.name.lower().endswith(".csv"):
        raw = pd.read_csv(file)
    else:
        # Si Excel con varias hojas, usar la primera por defecto
        xls = pd.ExcelFile(file)
        sheet = st.sidebar.selectbox("Hoja de Excel", xls.sheet_names, index=0)
        raw = pd.read_excel(xls, sheet_name=sheet)

    st.success(f"Archivo cargado: {file.name} – {len(raw):,} filas")
    with st.expander("👀 Vista previa del archivo (primeras 200 filas)", expanded=False):
        st.dataframe(raw.head(200))

    st.sidebar.header("2) Mapear columnas")
    cols = list(raw.columns)
    fecha_col = st.sidebar.selectbox("Columna de fecha", cols)
    servicio_col = st.sidebar.selectbox("Columna de servicio", cols)
    cantidad_col = st.sidebar.selectbox("Columna de cantidad", cols)
    precio_col = st.sidebar.selectbox("Columna de precio", cols)

    try:
        dfm = coerce_schema(raw, {
            "fecha": fecha_col, "servicio": servicio_col,
            "cantidad": cantidad_col, "precio": precio_col,
        })
        st.session_state["dfm"] = dfm
        st.sidebar.success("Esquema normalizado ✅ (agregado mensual por servicio)")
    except Exception as e:
        st.sidebar.error(f"Error al normalizar: {e}")
        st.stop()

    # =============================
    # EDA
    # =============================
    st.header("Exploración de datos")
    col1, col2, col3, col4 = st.columns(4)
    total_serv = int(dfm["cantidad"].sum())
    total_ing = float(dfm["ingreso"].sum())
    meses = dfm["fecha"].nunique()
    servicios = dfm["servicio"].nunique()

    col1.metric("Servicios realizados (total)", f"{total_serv:,}")
    col2.metric("Ingresos (total)", f"₡{total_ing:,.0f}")
    col3.metric("Meses con datos", f"{meses}")
    col4.metric("Servicios distintos", f"{servicios}")

    # Filtros
    c1, c2 = st.columns([2,1])
    svc_sel = c1.multiselect(
        "Filtrar servicios (opcional)",
        sorted(dfm["servicio"].unique()),
        default=[],
    )
    tipo_sel = c2.multiselect(
        "Filtrar por tipo de servicio", ["Lavado", "Reparación"], default=[]
    )

    df_f = dfm.copy()
    if svc_sel:
        df_f = df_f[df_f["servicio"].isin(svc_sel)]
    if tipo_sel:
        df_f = df_f[df_f["tipo_servicio"].isin(tipo_sel)]

    # Serie total por mes
    fig_total = px.line(
        df_f.groupby("fecha", as_index=False)["cantidad"].sum(),
        x="fecha", y="cantidad", markers=True,
        title="Demanda total mensual (todas las series filtradas)"
    )
    st.plotly_chart(fig_total, use_container_width=True)

    # Top servicios
    topn = st.slider("Top servicios por demanda (acumulado)", 3, 20, 8)
    top_df = (
        df_f.groupby("servicio", as_index=False)["cantidad"].sum()
        .sort_values("cantidad", ascending=False)
        .head(topn)
    )
    fig_bar = px.bar(top_df, x="servicio", y="cantidad", title="Top servicios por demanda acumulada")
    st.plotly_chart(fig_bar, use_container_width=True)

    # =============================
    # Pronóstico
    # =============================
    st.header("Pronóstico de demanda por servicio")

    left, right = st.columns([2,1])
    servicios_disp = sorted(dfm["servicio"].unique())
    servicios_fore = left.multiselect("Elegí uno o varios servicios", servicios_disp)
    horizonte = right.selectbox("Horizonte (meses)", [3, 6, 9, 12], index=1)

    if servicios_fore:
        results = []
        tabs = st.tabs([f"🔮 {s}" for s in servicios_fore])
        for tab, svc in zip(tabs, servicios_fore):
            with tab:
                g = dfm[dfm["servicio"] == svc].sort_values("fecha")
                y = g.set_index("fecha")["cantidad"]
                # Fiteo y forecast
                metrics, df_fc = fit_and_forecast(y, horizon=horizonte)

                # Gráfico
                hist = g[["fecha", "cantidad"]].rename(columns={"cantidad": "y"})
                hist["serie"] = "histórico"
                fc_plot = df_fc.rename(columns={"yhat": "y"}).copy()
                fc_plot["serie"] = "pronóstico"
                plot_df = pd.concat([
                    hist[["fecha", "y", "serie"]],
                    fc_plot[["fecha", "y", "serie"]],
                ], ignore_index=True)

                fig = px.line(plot_df, x="fecha", y="y", color="serie", markers=True,
                              title=f"{svc} – histórico vs. pronóstico ({horizonte} m)")
                st.plotly_chart(fig, use_container_width=True)

                if "yhat_lower" in df_fc.columns:
                    band = px.area(
                        df_fc, x="fecha", y=["yhat_lower", "yhat_upper"],
                        title="Intervalos de confianza (aprox. 95%)"
                    )
                    st.plotly_chart(band, use_container_width=True)

                # Métricas
                st.subheader("Métricas de backtesting (sMAPE ↓)")
                if metrics and metrics.get("best"):
                    rows = [["ETS", metrics["ets"]["sMAPE"]]]
                    if metrics.get("arima"):
                        rows.append(["ARIMA", metrics["arima"]["sMAPE"]])
                        rows.append(["Mejor", min(metrics["ets"]["sMAPE"], metrics["arima"]["sMAPE"])])
                    else:
                        rows.append(["Mejor", metrics["ets"]["sMAPE"]])
                    met_df = pd.DataFrame(rows, columns=["Modelo", "sMAPE"]) 
                    st.dataframe(met_df)
                    if not HAVE_PMDARIMA:
                        st.info("ARIMA no disponible en este entorno. Se usó ETS y backtesting con ETS.")
                else:
                    st.info("Serie demasiado corta para backtesting robusto; se utilizó un modelo por defecto.")

                # Tabla de pronóstico y descarga
                out = df_fc.copy()
                out["servicio"] = svc
                out = out[["fecha", "servicio"] + [c for c in out.columns if c not in ["fecha", "servicio"]]]
                st.dataframe(out)

                results.append(out)

        # Consolidado y descarga
        all_fc = pd.concat(results, ignore_index=True)
        st.subheader("Descargar pronóstico consolidado")
        def to_excel_bytes(df: pd.DataFrame) -> bytes:
            bio = io.BytesIO()
            with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
                df.to_excel(writer, index=False, sheet_name="pronostico")
            return bio.getvalue()
        b = to_excel_bytes(all_fc)
        st.download_button(
            label="💾 Descargar Excel (pronóstico)",
            data=b,
            file_name=f"pronostico_taller_moto_{datetime.now().date()}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

else:
    st.info("Subí un archivo Excel/CSV para comenzar.")

st.caption("Hecho con ❤️ en Streamlit • Modelos: ETS (statsmodels) y Auto-ARIMA (pmdarima)")

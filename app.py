# app.py — Pronóstico Taller Moto (Cloud-safe, sin librerías pesadas)
import io
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# =============================
# Config & título
# =============================
st.set_page_config(page_title="Pronóstico Taller Moto", layout="wide")
st.title("🔧📈 Pronóstico de Demanda – Taller de Moto (Cloud-safe)")

st.markdown("""
Subí un **Excel/CSV** con columnas:
- **fecha** (diaria o mensual; se agrega a mes)
- **servicio** (categoría)
- **cantidad** (# servicios realizados)
- **precio** (unitario o promedio)

La app: limpia → agrega a **mensual** → EDA → **pronóstico** por servicio/**tipo**/**total**
usando métodos livianos (Naive, Seasonal Naive, MA, WMA, SES, Holt, LinearTrend y **Croston**).
""")

# =============================
# Catálogo Servicio → Tipo
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
# Utilidades: fechas / schema
# =============================
def _try_parse_date(s: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(s, errors="coerce", infer_datetime_format=True, utc=False)
    if parsed.isna().any():
        meses = {"ene":"01","feb":"02","mar":"03","abr":"04","may":"05","jun":"06",
                 "jul":"07","ago":"08","sep":"09","oct":"10","nov":"11","dic":"12"}
        def repl(x):
            if not isinstance(x, str):
                return x
            y = x.strip().lower()
            for k, v in meses.items():
                y = y.replace(f"{k}-", f"{v}-")
            return y
        parsed2 = pd.to_datetime(s.apply(repl), errors="coerce", dayfirst=True)
        parsed = parsed.fillna(parsed2)
    return parsed

def coerce_schema(df: pd.DataFrame, cols: Dict[str, str]) -> pd.DataFrame:
    df = df.rename(columns={
        cols["fecha"]: "fecha",
        cols["servicio"]: "servicio",
        cols["cantidad"]: "cantidad",
        cols["precio"]: "precio",
    }).copy()

    df["fecha"] = _try_parse_date(df["fecha"])
    if df["fecha"].isna().any():
        raise ValueError("No se pudieron parsear algunas fechas (columna 'fecha').")

    df["servicio"] = df["servicio"].astype(str).str.strip()
    df["cantidad"] = pd.to_numeric(df["cantidad"], errors="coerce").fillna(0.0)
    df["precio"]   = pd.to_numeric(df["precio"], errors="coerce")

    # Agregar a mensual
    df["fecha_mes"] = df["fecha"].dt.to_period("M").dt.to_timestamp()
    g = (df.groupby(["servicio", "fecha_mes"], as_index=False)
           .agg(cantidad=("cantidad","sum"), precio=("precio","mean"))
           .rename(columns={"fecha_mes":"fecha"}))

    # Completar meses faltantes por servicio
    full = []
    for svc, gi in g.groupby("servicio"):
        idx = pd.date_range(gi["fecha"].min(), gi["fecha"].max(), freq="MS")
        gi2 = gi.set_index("fecha").reindex(idx).rename_axis("fecha").reset_index()
        gi2["servicio"] = svc
        gi2["cantidad"] = gi2["cantidad"].fillna(0.0)
        gi2["precio"]   = gi2["precio"].ffill().bfill()
        full.append(gi2)

    dfm = pd.concat(full, ignore_index=True)
    dfm["tipo_servicio"] = dfm["servicio"].map(SERVICIO_TIPO)
    dfm["ingreso"] = dfm["cantidad"] * dfm["precio"].fillna(0)
    return dfm[["fecha","servicio","tipo_servicio","cantidad","precio","ingreso"]]

# =============================
# Forecast (métodos livianos) + Croston
# =============================
def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom = np.where(denom == 0, 1.0, denom)
    return 100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denom)

def _naive(y: pd.Series, h: int) -> np.ndarray:
    return np.array([y.iloc[-1]] * h)

def _seasonal_naive(y: pd.Series, h: int, m: int = 12) -> np.ndarray:
    m_eff = min(m, max(1, len(y)))
    if len(y) < 2:
        return _naive(y, h)
    vals = []
    for i in range(1, h+1):
        idx = -m_eff + (i-1) % m_eff
        vals.append(y.iloc[idx])
    return np.array(vals)

def _moving_average(y: pd.Series, h: int, w: int = 3) -> np.ndarray:
    ma = y.rolling(window=min(w, len(y))).mean().iloc[-1]
    return np.array([ma] * h)

def _wma(y: pd.Series, h: int, weights=(0.6, 0.3, 0.1)) -> np.ndarray:
    w = np.array(weights[:min(len(weights), len(y))], dtype=float)
    w = w / w.sum()
    last_vals = y.iloc[-len(w):].values
    val = float(np.dot(last_vals[::-1], w))  # más peso a lo reciente
    return np.array([val] * h)

def _ses(y: pd.Series, h: int, alpha: float = 0.3) -> np.ndarray:
    level = y.iloc[0]
    for t in range(1, len(y)):
        level = alpha * y.iloc[t] + (1 - alpha) * level
    return np.array([level] * h)

def _holt(y: pd.Series, h: int, alpha: float = 0.3, beta: float = 0.1) -> np.ndarray:
    level = y.iloc[0]
    trend = y.iloc[1] - y.iloc[0] if len(y) > 1 else 0.0
    for t in range(1, len(y)):
        prev_level = level
        level = alpha * y.iloc[t] + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
    return np.array([level + (i+1)*trend for i in range(h)])

def _linear_trend(y: pd.Series, h: int) -> np.ndarray:
    if len(y) < 2:
        return _naive(y, h)
    x = np.arange(len(y))
    a, b = np.polyfit(x, y.values.astype(float), 1)  # y = a*x + b
    x_fc = np.arange(len(y), len(y) + h)
    return a * x_fc + b

def _croston(y: pd.Series, h: int, alpha: float = 0.3) -> np.ndarray:
    # Demanda intermitente (muchos ceros)
    y = y.fillna(0).astype(float)
    if len(y) == 0:
        return np.array([0.0]*h)
    z = None  # demanda media en llegada
    p = None  # intervalo medio entre demandas (>0)
    q = 0     # contador de intervalos
    for val in y:
        if val > 0:
            z = val if z is None else alpha*val + (1-alpha)*z
            p = 1  if p is None else alpha*q + (1-alpha)*p
            q = 1
        else:
            q = q + 1 if q else 1
    if z is None:  # nunca hubo demanda
        return np.array([0.0]*h)
    if p is None or p == 0:
        p = 1.0
    rate = z / p
    return np.array([rate]*h)

def time_series_cv(series: pd.Series, horizon: int = 3,
                   initial: int = 18, step: int = 1, m: int = 12) -> Dict:
    y = series.dropna()
    if len(y) < max(initial, horizon) + 3:
        return {"best": None, "metrics": {}}

    methods = {
        "Naive":        lambda tr, H: _naive(tr, H),
        "SeasonalNaive":lambda tr, H: _seasonal_naive(tr, H, m=m),
        "MovingAvg":    lambda tr, H: _moving_average(tr, H, w=3),
        "WMA":          lambda tr, H: _wma(tr, H, weights=(0.6, 0.3, 0.1)),
        "SES":          lambda tr, H: _ses(tr, H, alpha=0.3),
        "Holt":         lambda tr, H: _holt(tr, H, alpha=0.3, beta=0.1),
        "LinearTrend":  lambda tr, H: _linear_trend(tr, H),
        "Croston":      lambda tr, H: _croston(tr, H),
    }

    errors = {k: [] for k in methods}
    for end in range(initial, len(y) - horizon + 1, step):
        train = y.iloc[:end]
        test  = y.iloc[end:end+horizon]
        for name, fn in methods.items():
            try:
                pred = fn(train, horizon)
            except Exception:
                pred = np.array([np.nan] * horizon)
            errors[name].append(smape(test.values, pred))

    avg = {k: float(np.nanmean(v)) if len(v) else np.inf for k, v in errors.items()}
    best = min(avg, key=avg.get) if len(avg) else None
    return {"best": best, "metrics": avg}

def fit_and_forecast(series: pd.Series, horizon: int = 6, seasonal_periods: int = 12):
    metrics = time_series_cv(series, horizon=min(horizon, 6),
                             initial=min(18, max(12, len(series)//2)),
                             m=seasonal_periods)
    best = metrics.get("best") if metrics else None
    method_map = {
        "Naive": _naive,
        "SeasonalNaive": lambda y, H: _seasonal_naive(y, H, m=seasonal_periods),
        "MovingAvg": _moving_average,
        "WMA": _wma,
        "SES": _ses,
        "Holt": _holt,
        "LinearTrend": _linear_trend,
        "Croston": _croston,
    }
    fn = method_map.get(best, lambda y,H: _seasonal_naive(y, H, m=seasonal_periods))
    fc = fn(series, horizon)

    pred_idx = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    df_fc = pd.DataFrame({"fecha": pred_idx, "yhat": fc})
    return metrics, df_fc

# =============================
# Sidebar: carga & mapeo
# =============================
st.sidebar.header("1) Cargar archivo")
file = st.sidebar.file_uploader("Subí un Excel (.xlsx/.xls) o CSV", type=["xlsx", "xls", "csv"])

if file is None:
    st.info("Subí un archivo para comenzar.")
    st.stop()

if file.name.lower().endswith(".csv"):
    raw = pd.read_csv(file)
else:
    xls = pd.ExcelFile(file)
    sheet = st.sidebar.selectbox("Hoja de Excel", xls.sheet_names, index=0)
    raw = pd.read_excel(xls, sheet_name=sheet)

st.success(f"Archivo cargado: {file.name} – {len(raw):,} filas")
with st.expander("👀 Vista previa (primeras 200 filas)", expanded=False):
    st.dataframe(raw.head(200))

st.sidebar.header("2) Mapear columnas")
cols = list(raw.columns)
fecha_col    = st.sidebar.selectbox("Columna de fecha", cols)
servicio_col = st.sidebar.selectbox("Columna de servicio", cols)
cantidad_col = st.sidebar.selectbox("Columna de cantidad", cols)
precio_col   = st.sidebar.selectbox("Columna de precio", cols)

try:
    dfm = coerce_schema(raw, {"fecha": fecha_col, "servicio": servicio_col,
                              "cantidad": cantidad_col, "precio": precio_col})
    st.sidebar.success("Esquema normalizado ✅ (agregado mensual por servicio)")
except Exception as e:
    st.sidebar.error(f"Error al normalizar: {e}")
    st.stop()

# =============================
# EDA
# =============================
st.header("Exploración de datos")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Servicios realizados (total)", f"{int(dfm['cantidad'].sum()):,}")
c2.metric("Ingresos (total)", f"₡{float(dfm['ingreso'].sum()):,.0f}")
c3.metric("Meses con datos", f"{dfm['fecha'].nunique()}")
c4.metric("Servicios distintos", f"{dfm['servicio'].nunique()}")

f1, f2 = st.columns([2,1])
svc_sel  = f1.multiselect("Filtrar servicios (opcional)", sorted(dfm["servicio"].unique()))
tipo_sel = f2.multiselect("Filtrar por tipo", ["Lavado","Reparación"])

df_f = dfm.copy()
if svc_sel:  df_f = df_f[df_f["servicio"].isin(svc_sel)]
if tipo_sel: df_f = df_f[df_f["tipo_servicio"].isin(tipo_sel)]

fig_total = px.line(df_f.groupby("fecha", as_index=False)["cantidad"].sum(),
                    x="fecha", y="cantidad", markers=True,
                    title="Demanda total mensual (filtro aplicado)")
st.plotly_chart(fig_total, use_container_width=True)

topn = st.slider("Top servicios por demanda (acumulado)", 3, 20, 8)
top_df = (df_f.groupby("servicio", as_index=False)["cantidad"].sum()
          .sort_values("cantidad", ascending=False).head(topn))
fig_bar = px.bar(top_df, x="servicio", y="cantidad", title="Top servicios por demanda acumulada")
st.plotly_chart(fig_bar, use_container_width=True)

# =============================
# Pronóstico (niveles)
# =============================
st.header("Pronóstico de demanda")
nivel = st.radio("Nivel de pronóstico", ["Por servicio", "Por tipo (Lavado/Reparación)", "Total (todos)"], index=0)

def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "pronostico") -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return bio.getvalue()

if nivel == "Por servicio":
    left, right = st.columns([2,1])
    servicios_disp = sorted(dfm["servicio"].unique())
    select_all = left.checkbox("Seleccionar todos los servicios", value=False)
    if select_all:
        servicios_fore = servicios_disp
    else:
        servicios_fore = left.multiselect("Elegí uno o varios servicios", servicios_disp)
    horizonte = right.selectbox("Horizonte (meses)", [3, 6, 9, 12], index=1)

    if servicios_fore:
        results = []
        tabs = st.tabs([f"🔮 {s}" for s in servicios_fore])
        for tab, svc in zip(tabs, servicios_fore):
            with tab:
                g = dfm[dfm["servicio"] == svc].sort_values("fecha")
                y = g.set_index("fecha")["cantidad"]
                metrics, df_fc = fit_and_forecast(y, horizon=horizonte, seasonal_periods=12)

                hist = g[["fecha","cantidad"]].rename(columns={"cantidad":"y"})
                hist["serie"] = "histórico"
                fc_plot = df_fc.rename(columns={"yhat":"y"}).copy()
                fc_plot["serie"] = "pronóstico"
                plot_df = pd.concat([hist[["fecha","y","serie"]], fc_plot[["fecha","y","serie"]]], ignore_index=True)

                fig = px.line(plot_df, x="fecha", y="y", color="serie", markers=True,
                              title=f"{svc} – histórico vs pronóstico ({horizonte} m)")
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Métricas (sMAPE ↓)")
                if metrics and metrics.get("metrics"):
                    met_df = pd.DataFrame(sorted(metrics["metrics"].items(), key=lambda x: x[1]), columns=["Modelo","sMAPE"])
                    st.dataframe(met_df, use_container_width=True)
                    st.success(f"Mejor método: {metrics['best']}")
                else:
                    st.info("Serie corta para backtesting; se usó SeasonalNaive por defecto.")

                out = df_fc.copy()
                out["servicio"] = svc
                out = out[["fecha","servicio","yhat"]]
                st.dataframe(out, use_container_width=True)
                results.append(out)

        all_fc = pd.concat(results, ignore_index=True)
        st.subheader("Descargar pronóstico consolidado")
        st.download_button(
            "💾 Descargar Excel (pronóstico servicios)",
            data=to_excel_bytes(all_fc, "pronostico"),
            file_name=f"pronostico_taller_moto_{datetime.now().date()}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

elif nivel == "Por tipo (Lavado/Reparación)":
    left, right = st.columns([2,1])
    tipos_disp = ["Lavado", "Reparación"]
    tipos_sel = left.multiselect("Elegí tipo(s) de servicio", tipos_disp, default=tipos_disp)
    horizonte = right.selectbox("Horizonte (meses)", [3, 6, 9, 12], index=1)

    if tipos_sel:
        results = []
        tabs = st.tabs([f"🧩 {t}" for t in tipos_sel])
        for tab, t in zip(tabs, tipos_sel):
            with tab:
                g = (dfm[dfm["tipo_servicio"] == t].groupby("fecha", as_index=False)["cantidad"].sum())
                y = g.set_index("fecha")["cantidad"]
                metrics, df_fc = fit_and_forecast(y, horizon=horizonte, seasonal_periods=12)

                hist = g.rename(columns={"cantidad":"y"}); hist["serie"] = "histórico"
                fc_plot = df_fc.rename(columns={"yhat":"y"}); fc_plot["serie"] = "pronóstico"
                plot_df = pd.concat([hist[["fecha","y","serie"]], fc_plot[["fecha","y","serie"]]])
                fig = px.line(plot_df, x="fecha", y="y", color="serie", markers=True,
                              title=f"{t} – histórico vs pronóstico ({horizonte} m)")
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Métricas (sMAPE ↓)")
                if metrics and metrics.get("metrics"):
                    met_df = pd.DataFrame(sorted(metrics["metrics"].items(), key=lambda x: x[1]), columns=["Modelo","sMAPE"])
                    st.dataframe(met_df, use_container_width=True)
                    st.success(f"Mejor método: {metrics['best']}")
                else:
                    st.info("Serie corta para backtesting.")

                out = df_fc.copy(); out["tipo_servicio"] = t
                out = out[["fecha","tipo_servicio","yhat"]]
                st.dataframe(out, use_container_width=True)
                results.append(out)

        all_fc = pd.concat(results, ignore_index=True)
        st.subheader("Descargar pronóstico por tipo")
        st.download_button(
            "💾 Descargar Excel (por tipo)",
            data=to_excel_bytes(all_fc, "pronostico_por_tipo"),
            file_name=f"pronostico_por_tipo_{datetime.now().date()}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

else:  # Total (todos)
    right = st.columns([1])[0]
    horizonte = right.selectbox("Horizonte (meses)", [3, 6, 9, 12], index=1)

    g = dfm.groupby("fecha", as_index=False)["cantidad"].sum()
    y = g.set_index("fecha")["cantidad"]
    metrics, df_fc = fit_and_forecast(y, horizon=horizonte, seasonal_periods=12)

    hist = g.rename(columns={"cantidad":"y"}); hist["serie"] = "histórico"
    fc_plot = df_fc.rename(columns={"yhat":"y"}); fc_plot["serie"] = "pronóstico"
    plot_df = pd.concat([hist[["fecha","y","serie"]], fc_plot[["fecha","y","serie"]]])
    fig = px.line(plot_df, x="fecha", y="y", color="serie", markers=True,
                  title=f"TOTAL – histórico vs pronóstico ({horizonte} m)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Métricas (sMAPE ↓)")
    if metrics and metrics.get("metrics"):
        met_df = pd.DataFrame(sorted(metrics["metrics"].items(), key=lambda x: x[1]), columns=["Modelo","sMAPE"])
        st.dataframe(met_df, use_container_width=True)
        st.success(f"Mejor método: {metrics['best']}")
    else:
        st.info("Serie corta para backtesting.")

    out = df_fc.copy(); out["nivel"] = "TOTAL"
    st.dataframe(out, use_container_width=True)
    st.subheader("Descargar pronóstico total")
    st.download_button(
        "💾 Descargar Excel (total)",
        data=to_excel_bytes(out, "pronostico_total"),
        file_name=f"pronostico_total_{datetime.now().date()}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

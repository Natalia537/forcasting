# app.py — Pronóstico Taller: 5 métodos (parámetros fijos α=β=γ=0.10), fórmulas correctas
import io
from datetime import datetime
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ==========================
# Configuración general
# ==========================
st.set_page_config(page_title="Pronóstico Taller", layout="wide")
st.title("🔧📈 Pronóstico de Demanda – 5 métodos (α=β=γ=0.10)")

st.markdown("""
Carga un **Excel/CSV** con columnas: **fecha**, **servicio**, **cantidad**, **precio**.  
La app agrega a **mensual**, y pronostica con **parámetros fijos**:
**Estático (Naive), Exponencial Simple (SES), Holt, Holt–Winters multiplicativo (Winter), Promedio móvil**.
Muestra **gráficas y tablas por método**, **sMAPE** por backtesting y exporta a **Excel** (una hoja por método).
""")

# ==========================
# Catálogo Servicio → Tipo
# ==========================
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

# ==========================
# Utilidades de fechas / esquema
# ==========================
def _try_parse_date(s: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(s, errors="coerce", infer_datetime_format=True, utc=False)
    if parsed.isna().any():
        meses = {"ene":"01","feb":"02","mar":"03","abr":"04","may":"05","jun":"06",
                 "jul":"07","ago":"08","sep":"09","oct":"10","nov":"11","dic":"12"}
        def repl(x):
            if not isinstance(x, str): return x
            y = x.lower().strip()
            for k,v in meses.items(): y = y.replace(f"{k}-", f"{v}-")
            return y
        parsed2 = pd.to_datetime(s.apply(repl), errors="coerce", dayfirst=True)
        parsed = parsed.fillna(parsed2)
    return parsed

def coerce_schema(df: pd.DataFrame, cols: Dict[str,str]) -> pd.DataFrame:
    df = df.rename(columns={
        cols["fecha"]:"fecha",
        cols["servicio"]:"servicio",
        cols["cantidad"]:"cantidad",
        cols["precio"]:"precio"
    }).copy()
    df["fecha"] = _try_parse_date(df["fecha"])
    if df["fecha"].isna().any():
        raise ValueError("No se pudieron parsear algunas fechas.")
    df["servicio"] = df["servicio"].astype(str).str.strip()
    df["cantidad"] = pd.to_numeric(df["cantidad"], errors="coerce").fillna(0.0)
    df["precio"]   = pd.to_numeric(df["precio"], errors="coerce")

    # Agregado mensual
    df["fecha_mes"] = df["fecha"].dt.to_period("M").dt.to_timestamp()
    g = (df.groupby(["servicio","fecha_mes"], as_index=False)
           .agg(cantidad=("cantidad","sum"), precio=("precio","mean"))
           .rename(columns={"fecha_mes":"fecha"}))

    # Completar meses faltantes por servicio
    outs = []
    for svc, gi in g.groupby("servicio"):
        idx = pd.date_range(gi["fecha"].min(), gi["fecha"].max(), freq="MS")
        gi2 = gi.set_index("fecha").reindex(idx).rename_axis("fecha").reset_index()
        gi2["servicio"] = svc
        gi2["cantidad"] = gi2["cantidad"].fillna(0.0)
        gi2["precio"]   = gi2["precio"].ffill().bfill()
        outs.append(gi2)

    dfm = pd.concat(outs, ignore_index=True)
    dfm["tipo_servicio"] = dfm["servicio"].map(SERVICIO_TIPO)
    dfm["ingreso"] = dfm["cantidad"] * dfm["precio"].fillna(0)
    return dfm[["fecha","servicio","tipo_servicio","cantidad","precio","ingreso"]]

# ==========================
# Métrica
# ==========================
def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom == 0, 1.0, denom)
    return float(100.0 * np.mean(2.0 * np.abs(y_true - y_pred) / denom))

# ==========================
# 5 Métodos (fórmulas correctas; α=β=γ=0.10)
# ==========================
ALPHA = 0.10
BETA  = 0.10
GAMMA = 0.10

def fc_naive(y: pd.Series, h: int) -> np.ndarray:
    return np.repeat(y.iloc[-1], h)

def fc_ses(y: pd.Series, h: int, alpha: float = ALPHA) -> np.ndarray:
    # Exponencial simple: L_t = αY_t + (1-α)L_{t-1}; pronóstico = L_t
    y = y.astype(float)
    L = y.iloc[0]
    for t in range(1, len(y)):
        L = alpha * y.iloc[t] + (1 - alpha) * L
    return np.repeat(L, h)

def fc_holt(y: pd.Series, h: int, alpha: float = ALPHA, beta: float = BETA) -> np.ndarray:
    # Holt (aditivo): L_t = αY_t + (1-α)(L_{t-1}+B_{t-1}); B_t = β(L_t-L_{t-1}) + (1-β)B_{t-1}
    y = y.astype(float)
    if len(y) < 2:
        return fc_naive(y, h)
    L = y.iloc[0]
    B = y.iloc[1] - y.iloc[0]
    for t in range(1, len(y)):
        L_prev = L
        L = alpha * y.iloc[t] + (1 - alpha) * (L + B)
        B = beta * (L - L_prev) + (1 - beta) * B
    return np.array([L + (i+1)*B for i in range(h)], dtype=float)

def _init_seasonal_multiplicative(y: pd.Series, m: int = 12) -> Tuple[float, float, np.ndarray]:
    # Inicialización clásica de HW multiplicativo (ratios a medias estacionales)
    n = len(y)
    k = n // m
    if k >= 2:
        seasons = y.values[:k*m].reshape(k, m)
        season_means = seasons.mean(axis=1, keepdims=True)
        ratios = seasons / season_means
        S = ratios.mean(axis=0)
    else:
        first = y.iloc[:m].values
        S = first / first.mean()
    S = S * (m / S.sum())        # Normalizar a promedio 1.0
    L0 = (y.iloc[:m] / S).mean() # Nivel inicial
    if k >= 2:
        L1 = (y.iloc[m:2*m] / S).mean()
        B0 = (L1 - L0) / m
    else:
        x = np.arange(m)
        a, b = np.polyfit(x, (y.iloc[:m] / S).values, 1)
        B0 = a
    return float(L0), float(B0), S.astype(float)

def fc_hw_mul(y: pd.Series, h: int, alpha: float = ALPHA, beta: float = BETA,
              gamma: float = GAMMA, m: int = 12) -> np.ndarray:
    # Holt–Winters multiplicativo (convención 1 paso adelante con S_{t-m})
    y = y.astype(float)
    if len(y) < m + 2:
        return fc_naive(y, h) if len(y) < m else fc_seasonal_naive(y, h, m)
    L, B, S = _init_seasonal_multiplicative(y, m=m)
    S_list = list(S)   # S_t se actualiza en la misma posición t (tiene memoria m)

    for t in range(len(y)):
        s_idx = t - m
        S_tm = S_list[s_idx] if s_idx >= 0 else S_list[s_idx % m]
        # Pronóstico de 1 paso para el tiempo t (usando estado t-1)
        # F_t = (L + B) * S_{t-m}
        Yt = y.iloc[t]
        L_new = alpha * (Yt / S_tm) + (1 - alpha) * (L + B)
        B_new = beta  * (L_new - L) + (1 - beta) * B
        S_new = gamma * (Yt / L_new) + (1 - gamma) * S_tm
        L, B = L_new, B_new
        if s_idx >= 0:
            S_list[t] = S_new
        else:
            S_list.append(S_new)

    # Pronóstico futuro k pasos: (L + kB) * S_{t-m+k}
    idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    fc = []
    for k_ahead in range(1, h+1):
        s_idx = len(y) - m + k_ahead
        S_k = S_list[s_idx] if s_idx < len(S_list) else S_list[s_idx % m]
        fc.append((L + k_ahead * B) * S_k)
    return np.array(fc, dtype=float)

def fc_moving_average(y: pd.Series, h: int, window: int = 3) -> np.ndarray:
    # Promedio móvil simple de ventana w: pronóstico constante = media de los últimos w
    w = max(1, min(window, len(y)))
    ma = y.rolling(window=w).mean().iloc[-1]
    return np.repeat(ma, h)

# ==========================
# Backtesting (origen rodante) con parámetros fijos
# ==========================
def backtest_fixed(y: pd.Series, method: str, horizon: int, m: int = 12, ma_window: int = 3) -> float:
    errs: List[float] = []
    # usar como "entrenamiento" al menos 12 meses o mitad de la serie (lo que sea mayor)
    initial = min(max(12, len(y)//2), len(y)-horizon-1)
    if initial <= 0:
        return np.inf
    for end in range(initial, len(y) - horizon + 1):
        tr = y.iloc[:end]
        te = y.iloc[end:end+horizon]
        if method == "Naive":
            pred = fc_naive(tr, horizon)
        elif method == "SES":
            pred = fc_ses(tr, horizon, alpha=ALPHA)
        elif method == "Holt":
            pred = fc_holt(tr, horizon, alpha=ALPHA, beta=BETA)
        elif method == "Winter":
            pred = fc_hw_mul(tr, horizon, alpha=ALPHA, beta=BETA, gamma=GAMMA, m=m)
        elif method == "MA":
            pred = fc_moving_average(tr, horizon, window=ma_window)
        else:
            pred = np.repeat(np.nan, horizon)
        errs.append(smape(te.values, pred))
    return float(np.nanmean(errs)) if errs else np.inf

def forecast_all_fixed(y: pd.Series, horizon: int, m: int = 12, ma_window: int = 3) -> Tuple[pd.DataFrame, Dict[str,float], Dict[str,pd.DataFrame]]:
    methods = ["Naive","SES","Holt","Winter","MA"]
    metrics: Dict[str,float] = {}
    dfs: Dict[str,pd.DataFrame] = {}
    idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    df_all = pd.DataFrame({"fecha": idx})

    for name in methods:
        metrics[name] = backtest_fixed(y, name, horizon, m, ma_window)
        if name == "Naive":
            yhat = fc_naive(y, horizon)
        elif name == "SES":
            yhat = fc_ses(y, horizon, alpha=ALPHA)
        elif name == "Holt":
            yhat = fc_holt(y, horizon, alpha=ALPHA, beta=BETA)
        elif name == "Winter":
            yhat = fc_hw_mul(y, horizon, alpha=ALPHA, beta=BETA, gamma=GAMMA, m=m)
        else:  # MA
            yhat = fc_moving_average(y, horizon, window=ma_window)
        df_m = pd.DataFrame({"fecha": idx, "yhat": yhat})
        dfs[name] = df_m.copy()
        df_all[name] = yhat
    return df_all, metrics, dfs

# ==========================
# Sidebar: carga & mapeo
# ==========================
st.sidebar.header("1) Cargar archivo")
file = st.sidebar.file_uploader("Excel (.xlsx/.xls) o CSV", type=["xlsx","xls","csv"])
if not file:
    st.info("Subí un archivo para comenzar.")
    st.stop()

raw = pd.read_excel(file) if file.name.lower().endswith(("xlsx","xls")) else pd.read_csv(file)
with st.expander("👀 Vista previa", expanded=False):
    st.dataframe(raw.head(200), use_container_width=True)

st.sidebar.header("2) Mapear columnas")
cols = raw.columns.tolist()
fecha_col    = st.sidebar.selectbox("Fecha", cols)
servicio_col = st.sidebar.selectbox("Servicio", cols)
cantidad_col = st.sidebar.selectbox("Cantidad", cols)
precio_col   = st.sidebar.selectbox("Precio", cols)

try:
    dfm = coerce_schema(raw, {"fecha":fecha_col,"servicio":servicio_col,"cantidad":cantidad_col,"precio":precio_col})
    st.sidebar.success("Esquema normalizado ✅ (mensual por servicio)")
except Exception as e:
    st.sidebar.error(f"Error: {e}")
    st.stop()

# ==========================
# EDA breve
# ==========================
st.header("Exploración")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Servicios (total)", f"{int(dfm['cantidad'].sum()):,}")
c2.metric("Ingresos (total)", f"₡{float(dfm['ingreso'].sum()):,.0f}")
c3.metric("Meses", dfm["fecha"].nunique())
c4.metric("Servicios únicos", dfm["servicio"].nunique())

fig_total = px.line(dfm.groupby("fecha", as_index=False)["cantidad"].sum(),
                    x="fecha", y="cantidad", markers=True, title="Demanda total mensual")
st.plotly_chart(fig_total, use_container_width=True)

# ==========================
# Pronóstico
# ==========================
st.header("Pronóstico (α=β=γ=0.10)")
nivel = st.radio("Nivel", ["Por servicio","Por tipo (Lavado/Reparación)","Total"], index=0)
horizonte = st.selectbox("Horizonte (meses)", [3,6,9,12], index=1)
ma_window = st.slider("Ventana de Promedio Móvil (MA)", min_value=2, max_value=12, value=3, step=1)
SEASON_M = 12

def export_all_sheets(dfs: Dict[str,pd.DataFrame], metrics: Dict[str,float], fname: str) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        met = pd.DataFrame(sorted(metrics.items(), key=lambda x: x[1]), columns=["Modelo","sMAPE"])
        met.to_excel(w, index=False, sheet_name="metricas")
        for name, d in dfs.items():
            sheet = name[:31]
            pd.DataFrame({"sMAPE":[metrics.get(name, np.nan)]}).to_excel(w, index=False, sheet_name=sheet, startrow=0)
            d.to_excel(w, index=False, sheet_name=sheet, startrow=2)
    return buf.getvalue()

def run_block(title: str, g: pd.DataFrame):
    st.subheader(title)
    y = g.set_index("fecha")["cantidad"].astype(float)

    df_all, metrics, dfs = forecast_all_fixed(y, horizon=horizonte, m=SEASON_M, ma_window=ma_window)

    # Mejor por sMAPE
    best = min(metrics, key=metrics.get)
    hist = g[["fecha","cantidad"]].rename(columns={"cantidad":"y"}).assign(serie="histórico")
    best_df = dfs[best].rename(columns={"yhat":"y"}).assign(serie=f"pronóstico ({best})")
    fig = px.line(pd.concat([hist, best_df]), x="fecha", y="y", color="serie",
                  title=f"{title} – mejor: {best} (sMAPE={metrics[best]:.2f})", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    # Comparación de los 5 métodos
    plot_m = df_all.melt(id_vars=["fecha"], var_name="modelo", value_name="yhat")
    fig_all = px.line(plot_m, x="fecha", y="yhat", color="modelo",
                      title=f"{title} – comparación de métodos (α=β=γ=0.10)")
    fig_all.add_scatter(x=hist["fecha"], y=hist["y"], mode="lines+markers", name="histórico", line=dict(dash="dot"))
    st.plotly_chart(fig_all, use_container_width=True)

    # Métricas
    st.subheader("Métricas (sMAPE ↓)")
    st.dataframe(pd.DataFrame(sorted(metrics.items(), key=lambda x: x[1]), columns=["Modelo","sMAPE"]),
                 use_container_width=True)

    # Tabs por método (gráfica + tabla)
    st.subheader("Tablas y gráficas por método")
    tabs = st.tabs(list(dfs.keys()))
    for tab, (name, d) in zip(tabs, dfs.items()):
        with tab:
            st.write(f"**{name}**  • sMAPE={metrics[name]:.3f}")
            # Gráfica individual
            fig_i = px.line(pd.concat([hist, d.rename(columns={"yhat":"y"}).assign(serie=name)]),
                            x="fecha", y="y", color="serie", title=f"{title} – {name}", markers=True)
            st.plotly_chart(fig_i, use_container_width=True)
            # Tabla Yhat
            st.dataframe(d, use_container_width=True)

    # Exportar Excel
    st.subheader("Descarga")
    st.download_button(
        "💾 Descargar Excel (una hoja por método + métricas)",
        data=export_all_sheets(dfs, metrics, "pronosticos.xlsx"),
        file_name=f"pronosticos_{title.replace(' ','_')}_{datetime.now().date()}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# ---- Por servicio ----
if nivel == "Por servicio":
    servicios = sorted(dfm["servicio"].unique())
    sel_all = st.checkbox("Seleccionar todos", value=False)
    svcs = servicios if sel_all else st.multiselect("Elegí servicio(s)", servicios)
    if not svcs: st.stop()
    for s in svcs:
        g = dfm[dfm["servicio"]==s].sort_values("fecha")
        run_block(f"🔧 {s}", g)

# ---- Por tipo ----
elif nivel == "Por tipo (Lavado/Reparación)":
    tipos = st.multiselect("Elegí tipo(s)", ["Lavado","Reparación"], default=["Lavado","Reparación"])
    if not tipos: st.stop()
    for t in tipos:
        g = (dfm[dfm["tipo_servicio"]==t].groupby("fecha", as_index=False)["cantidad"].sum()
             .assign(servicio=t, tipo_servicio=t))
        run_block(f"🧩 {t}", g)

# ---- Total ----
else:
    g = dfm.groupby("fecha", as_index=False)["cantidad"].sum().assign(servicio="TOTAL", tipo_servicio="TOTAL")
    run_block("📦 TOTAL", g)

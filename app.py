# app.py — Pronóstico Taller (5 métodos con tuning y backtesting; sin librerías pesadas)
import io
from datetime import datetime
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# =============== Config ===============
st.set_page_config(page_title="Pronóstico Taller", layout="wide")
st.title("🔧📈 Pronóstico de Demanda – 5 métodos (tuning & backtesting)")

st.markdown("""
Subí un archivo **Excel/CSV** con columnas:
- **fecha**, **servicio**, **cantidad**, **precio**.

La app normaliza a **mensual**, evalúa 5 métodos con **backtesting** y **elige el mejor** por sMAPE:
**Naive, SeasonalNaive, Holt, Holt-Winters multiplicativo y Croston**.
Podés ver **curvas, tablas y métricas**; además descargar un **Excel** con una hoja por método.
""")

# =============== Helpers de fechas / schema ===============
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

def _try_parse_date(s: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(s, errors="coerce", infer_datetime_format=True, utc=False)
    if parsed.isna().any():
        meses = {"ene":"01","feb":"02","mar":"03","abr":"04","may":"05","jun":"06",
                 "jul":"07","ago":"08","sep":"09","oct":"10","nov":"11","dic":"12"}
        def repl(x):
            if not isinstance(x, str): return x
            y = x.lower().strip()
            for k,v in meses.items():
                y = y.replace(f"{k}-", f"{v}-")
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

    # mensual
    df["fecha_mes"] = df["fecha"].dt.to_period("M").dt.to_timestamp()
    g = (df.groupby(["servicio","fecha_mes"], as_index=False)
           .agg(cantidad=("cantidad","sum"), precio=("precio","mean"))
           .rename(columns={"fecha_mes":"fecha"}))

    # reindex por servicio (meses faltantes)
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

# =============== Métricas ===============
def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom == 0, 1.0, denom)
    return float(100.0*np.mean(2.0*np.abs(y_true - y_pred)/denom))

# =============== Métodos base (5) ===============
def fc_naive(y: pd.Series, h: int) -> np.ndarray:
    return np.repeat(y.iloc[-1], h)

def fc_seasonal_naive(y: pd.Series, h: int, m: int=12) -> np.ndarray:
    if len(y) < m:
        return fc_naive(y, h)
    last_season = y.iloc[-m:].values
    reps = int(np.ceil(h/m))
    return np.tile(last_season, reps)[:h]

def fc_holt(y: pd.Series, h: int, alpha: float, beta: float) -> np.ndarray:
    # Holt (nivel + tendencia) – formulación clásica aditiva
    y = y.astype(float)
    if len(y) < 2:
        return fc_naive(y, h)
    l = y.iloc[0]
    b = y.iloc[1] - y.iloc[0]
    for t in range(1, len(y)):
        l_prev = l
        l = alpha * y.iloc[t] + (1-alpha) * (l + b)
        b = beta * (l - l_prev) + (1-beta)*b
    return np.array([l + (i+1)*b for i in range(h)])

def _init_seasonal_multiplicative(y: pd.Series, m: int=12) -> Tuple[float, float, np.ndarray]:
    # inicialización clásica para HW multiplicativo (promedios por temporada + ratios)
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
    S = S * (m / S.sum())  # normalizar a promedio 1.0
    L0 = (y.iloc[:m] / S).mean()
    if k >= 2:
        L1 = (y.iloc[m:2*m] / S).mean()
        B0 = (L1 - L0) / m
    else:
        x = np.arange(m)
        a, b = np.polyfit(x, (y.iloc[:m] / S).values, 1)
        B0 = a
    return float(L0), float(B0), S.astype(float)

def fc_holt_winters_mul(y: pd.Series, h: int, alpha: float, beta: float, gamma: float, m: int=12) -> np.ndarray:
    # HW multiplicativo: Y_t ≈ (L_{t-1}+B_{t-1}) * S_{t-m}
    # Updates:
    # L_t = α*(Y_t / S_{t-m}) + (1-α)*(L_{t-1}+B_{t-1})
    # B_t = β*(L_t - L_{t-1}) + (1-β)*B_{t-1}
    # S_t = γ*(Y_t / L_t) + (1-γ)*S_{t-m}
    y = y.astype(float)
    if len(y) < m+2:
        return fc_seasonal_naive(y, h, m=m)
    L, B, S = _init_seasonal_multiplicative(y, m=m)
    L_hist = [L]; B_hist = [B]; S_list = list(S)  # S_list va creciendo

    for t in range(len(y)):
        # índice estacional que toca
        s_idx = t - m
        S_tm = S_list[s_idx] if s_idx >= 0 else S_list[s_idx % m]
        # pronóstico dentro de la muestra (1 paso adelante)
        # Ft = (L + B) * S_{t-m}
        Ft = (L + B) * S_tm

        # actualizar con la observación Y_t
        Yt = y.iloc[t]
        L_new = alpha * (Yt / S_tm) + (1-alpha) * (L + B)
        B_new = beta * (L_new - L) + (1-beta) * B
        S_new = gamma * (Yt / L_new) + (1-gamma) * S_tm

        L, B = L_new, B_new
        if s_idx >= 0:
            S_list[t] = S_new
        else:
            # antes de completar el primer año, extendemos
            S_list.append(S_new)

        L_hist.append(L); B_hist.append(B)

    # pronóstico futuro
    idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    yhat = []
    for k in range(1, h+1):
        s_idx = len(y) - m + k
        S_k = S_list[s_idx] if s_idx < len(S_list) else S_list[s_idx % m]
        yhat.append((L + k*B) * S_k)
    return np.array(yhat, dtype=float)

def fc_croston(y: pd.Series, h: int, alpha: float=0.3) -> np.ndarray:
    # Croston clásico para demanda intermitente
    y = y.fillna(0).astype(float)
    z, p, q = None, None, 0
    for v in y:
        if v > 0:
            z = v if z is None else alpha*v + (1-alpha)*z
            p = 1 if p is None else alpha*q + (1-alpha)*p
            q = 1
        else:
            q = (q or 0) + 1
    if z is None:
        return np.zeros(h)
    if not p:
        p = 1.0
    rate = z/p
    return np.repeat(rate, h)

# =============== Backtesting & tuning ===============
def rolling_origin_backtest(y: pd.Series, horizon: int, initial: int, m: int,
                            method_name: str, param: dict) -> float:
    """Devuelve sMAPE promedio para el método con params dados."""
    errors: List[float] = []
    for end in range(initial, len(y) - horizon + 1):
        train = y.iloc[:end]
        test  = y.iloc[end:end+horizon]
        try:
            if method_name == "Naive":
                pred = fc_naive(train, horizon)
            elif method_name == "SeasonalNaive":
                pred = fc_seasonal_naive(train, horizon, m=m)
            elif method_name == "Holt":
                pred = fc_holt(train, horizon, alpha=param["alpha"], beta=param["beta"])
            elif method_name == "HoltWintersM":
                pred = fc_holt_winters_mul(train, horizon, alpha=param["alpha"], beta=param["beta"],
                                           gamma=param["gamma"], m=m)
            elif method_name == "Croston":
                pred = fc_croston(train, horizon, alpha=param["alpha"])
            else:
                pred = np.repeat(np.nan, horizon)
        except Exception:
            pred = np.repeat(np.nan, horizon)
        errors.append(smape(test.values, pred))
    return float(np.nanmean(errors)) if errors else np.inf

def tune_method(y: pd.Series, method_name: str, horizon: int, m: int=12) -> Tuple[dict, float]:
    """Búsqueda de parámetros por rejilla simple; retorna (mejores_params, sMAPE)."""
    y = y.astype(float)
    initial = min( max(12, len(y)//2), len(y)-horizon-1 )
    if initial <= 0:
        return ({}, np.inf)

    if method_name == "Naive":
        return ({}, rolling_origin_backtest(y, horizon, initial, m, "Naive", {}))
    if method_name == "SeasonalNaive":
        return ({"m": m}, rolling_origin_backtest(y, horizon, initial, m, "SeasonalNaive", {"m":m}))
    if method_name == "Croston":
        best, best_err = None, np.inf
        for a in [0.1, 0.2, 0.3, 0.5]:
            err = rolling_origin_backtest(y, horizon, initial, m, "Croston", {"alpha":a})
            if err < best_err: best, best_err = {"alpha":a}, err
        return (best, best_err)
    if method_name == "Holt":
        best, best_err = None, np.inf
        for a in [0.1, 0.2, 0.3, 0.5, 0.8]:
            for b in [0.05, 0.1, 0.2, 0.3]:
                err = rolling_origin_backtest(y, horizon, initial, m, "Holt", {"alpha":a, "beta":b})
                if err < best_err: best, best_err = {"alpha":a,"beta":b}, err
        return (best, best_err)
    if method_name == "HoltWintersM":
        best, best_err = None, np.inf
        for a in [0.1, 0.2, 0.3]:
            for b in [0.05, 0.1, 0.2]:
                for g in [0.05, 0.1, 0.2, 0.3]:
                    err = rolling_origin_backtest(y, horizon, initial, m, "HoltWintersM",
                                                  {"alpha":a, "beta":b, "gamma":g})
                    if err < best_err: best, best_err = {"alpha":a,"beta":b,"gamma":g}, err
        return (best, best_err)
    return ({}, np.inf)

def forecast_with_best(y: pd.Series, horizon: int, m: int=12) -> Tuple[pd.DataFrame, Dict[str,float], Dict[str,pd.DataFrame]]:
    """Entrena cada método, selecciona mejores parámetros por sMAPE y produce pronóstico de TODOS.
       Retorna: (df_merged_all, metrics_dict, dict_method_to_df)"""
    methods = ["Naive","SeasonalNaive","Holt","HoltWintersM","Croston"]
    metrics: Dict[str,float] = {}
    dfs: Dict[str,pd.DataFrame] = {}
    idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    df_all = pd.DataFrame({"fecha": idx})

    for name in methods:
        params, err = tune_method(y, name, horizon, m)
        metrics[name] = err
        # forecast final con mejores params
        if name == "Naive":
            yhat = fc_naive(y, horizon)
        elif name == "SeasonalNaive":
            yhat = fc_seasonal_naive(y, horizon, m=m)
        elif name == "Holt":
            yhat = fc_holt(y, horizon, alpha=params["alpha"], beta=params["beta"])
        elif name == "HoltWintersM":
            yhat = fc_holt_winters_mul(y, horizon, alpha=params["alpha"], beta=params["beta"], gamma=params["gamma"], m=m)
        elif name == "Croston":
            yhat = fc_croston(y, horizon, alpha=params["alpha"])
        else:
            yhat = np.repeat(np.nan, horizon)

        df_m = pd.DataFrame({"fecha": idx, "yhat": yhat})
        dfs[name] = df_m.copy()
        df_all[name] = yhat

    return df_all, metrics, dfs

# =============== UI: carga & mapeo ===============
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

# =============== EDA breve ===============
st.header("Exploración")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Servicios (total)", f"{int(dfm['cantidad'].sum()):,}")
c2.metric("Ingresos (total)", f"₡{float(dfm['ingreso'].sum()):,.0f}")
c3.metric("Meses", dfm["fecha"].nunique())
c4.metric("Servicios únicos", dfm["servicio"].nunique())

fig_total = px.line(dfm.groupby("fecha", as_index=False)["cantidad"].sum(),
                    x="fecha", y="cantidad", markers=True, title="Demanda total mensual")
st.plotly_chart(fig_total, use_container_width=True)

# =============== Pronóstico ===============
st.header("Pronóstico de demanda (5 métodos)")
nivel = st.radio("Nivel", ["Por servicio","Por tipo (Lavado/Reparación)","Total"], index=0)
horizonte = st.selectbox("Horizonte (meses)", [3,6,9,12], index=1)
m = 12  # estacionalidad mensual

def export_all_sheets(dfs: Dict[str,pd.DataFrame], metrics: Dict[str,float], fname: str) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        # ranking de métricas
        met = pd.DataFrame(sorted(metrics.items(), key=lambda x: x[1]), columns=["Modelo","sMAPE"])
        met.to_excel(w, index=False, sheet_name="metricas")
        # una hoja por método
        for name, d in dfs.items():
            sheet = name[:31]
            # título con sMAPE
            pd.DataFrame({"sMAPE":[metrics.get(name, np.nan)]}).to_excel(w, index=False, sheet_name=sheet, startrow=0)
            d.to_excel(w, index=False, sheet_name=sheet, startrow=2)
    return buf.getvalue()

def run_block(title: str, g: pd.DataFrame):
    st.subheader(title)
    y = g.set_index("fecha")["cantidad"].astype(float)
    if len(y) < 8:
        st.warning("Serie muy corta para tuning robusto; resultados orientativos.")
    df_all, metrics, dfs = forecast_with_best(y, horizon=horizonte, m=m)

    # gráfico: histórico + mejor modelo
    best = min(metrics, key=metrics.get)
    hist = g[["fecha","cantidad"]].rename(columns={"cantidad":"y"}).assign(serie="histórico")
    best_df = dfs[best].rename(columns={"yhat":"y"}).assign(serie=f"pronóstico ({best})")
    fig = px.line(pd.concat([hist, best_df]), x="fecha", y="y", color="serie",
                  title=f"{title} – mejor: {best} (sMAPE={metrics[best]:.2f})", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    # comparación: todos los métodos
    plot_m = df_all.melt(id_vars=["fecha"], var_name="modelo", value_name="yhat")
    fig_all = px.line(plot_m, x="fecha", y="yhat", color="modelo",
                      title=f"{title} – comparación de los 5 métodos")
    fig_all.add_scatter(x=hist["fecha"], y=hist["y"], mode="lines+markers", name="histórico", line=dict(dash="dot"))
    st.plotly_chart(fig_all, use_container_width=True)

    # métricas y tablas
    st.subheader("Métricas (sMAPE ↓)")
    st.dataframe(pd.DataFrame(sorted(metrics.items(), key=lambda x: x[1]), columns=["Modelo","sMAPE"]),
                 use_container_width=True)

    st.subheader("Tablas de Y por método")
    tabs = st.tabs(list(dfs.keys()))
    for tab, (name, d) in zip(tabs, dfs.items()):
        with tab:
            st.write(f"**{name}**  • sMAPE={metrics[name]:.3f}")
            st.dataframe(d, use_container_width=True)

    # exportar excel
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
    sel_all = st.checkbox("Seleccionar todos los servicios", value=False)
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


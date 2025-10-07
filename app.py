# app.py — Pronóstico Taller Moto: compara todos los métodos, muestra errores y exporta Excel por método
import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime

# =============================
# Config & título
# =============================
st.set_page_config(page_title="Pronóstico Taller Moto", layout="wide")
st.title("🔧📈 Pronóstico de Demanda – Taller de Moto (Cloud-safe)")

st.markdown("""
Subí un **Excel/CSV** con columnas: **fecha**, **servicio**, **cantidad**, **precio**.  
La app agrega a **mensual**, hace EDA y pronostica por **servicio / tipo / total** con métodos livianos:
**Naive, SeasonalNaive, MovingAvg, WMA, SES, Holt, LinearTrend, Croston**.  
Ahora ves **cada método** con su **gráfica**, su **tabla de Y** y su **error (sMAPE)**; además podés **descargar un Excel** con **una hoja por método**.
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
# Limpieza y schema
# =============================
def _try_parse_date(s: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
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

def coerce_schema(df: pd.DataFrame, cols: dict) -> pd.DataFrame:
    df = df.rename(columns={
        cols["fecha"]: "fecha",
        cols["servicio"]: "servicio",
        cols["cantidad"]: "cantidad",
        cols["precio"]: "precio"
    }).copy()

    df["fecha"] = _try_parse_date(df["fecha"])
    if df["fecha"].isna().any():
        raise ValueError("Hay fechas inválidas en la columna seleccionada.")
    df["servicio"] = df["servicio"].astype(str).str.strip()
    df["cantidad"] = pd.to_numeric(df["cantidad"], errors="coerce").fillna(0.0)
    df["precio"]   = pd.to_numeric(df["precio"], errors="coerce")

    df["fecha_mes"] = df["fecha"].dt.to_period("M").dt.to_timestamp()
    g = df.groupby(["servicio","fecha_mes"], as_index=False).agg({"cantidad":"sum","precio":"mean"})
    g = g.rename(columns={"fecha_mes":"fecha"})

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
# Modelos de pronóstico
# =============================
def smape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = np.abs(y_true)+np.abs(y_pred)
    denom = np.where(denom==0,1,denom)
    return float(100*np.mean(2*np.abs(y_true-y_pred)/denom))

def _naive(y,h): 
    return np.repeat(y.iloc[-1], h)

def _seasonal_naive(y,h,m=12):
    m_eff = min(m, max(1, len(y)))
    if len(y) < 2: 
        return _naive(y,h)
    vals = []
    for i in range(1, h+1):
        idx = -m_eff + (i-1) % m_eff
        vals.append(y.iloc[idx])
    return np.array(vals)

def _moving_average(y,h,w=3):
    ma = y.rolling(window=min(w,len(y))).mean().iloc[-1]
    return np.repeat(ma, h)

def _wma(y,h,weights=(0.6,0.3,0.1)):
    w = np.array(weights[:min(len(weights), len(y))], dtype=float)
    w = w / w.sum()
    last_vals = y.iloc[-len(w):].values
    val = float(np.dot(last_vals[::-1], w))
    return np.repeat(val, h)

def _ses(y,h,alpha=0.3):
    level=y.iloc[0]
    for t in range(1,len(y)):
        level = alpha*y.iloc[t] + (1-alpha)*level
    return np.repeat(level, h)

def _holt(y,h,a=0.3,b=0.1):
    level=y.iloc[0]
    trend=y.iloc[1]-y.iloc[0] if len(y)>1 else 0.0
    for t in range(1,len(y)):
        prev=level
        level = a*y.iloc[t] + (1-a)*(level+trend)
        trend = b*(level-prev) + (1-b)*trend
    return np.array([level + (i+1)*trend for i in range(h)])

def _linear_trend(y,h):
    if len(y)<2: return _naive(y,h)
    x=np.arange(len(y)); a,b=np.polyfit(x, y.values.astype(float), 1)
    return a*np.arange(len(y), len(y)+h) + b

def _croston(y,h,a=0.3):
    y = y.fillna(0).astype(float)
    z = None  # demanda media
    p = None  # intervalo medio
    q = 0     # contador intervalo actual
    for v in y:
        if v>0:
            z = v if z is None else a*v + (1-a)*z
            p = 1 if p is None else a*q + (1-a)*p
            q = 1
        else:
            q = (q or 0) + 1
    if z is None: return np.zeros(h)
    if not p: p = 1.0
    rate = z/p
    return np.repeat(rate, h)

# Catálogo de candidatos
def candidate_methods(seasonal_periods=12):
    return {
        "Naive": _naive,
        "SeasonalNaive": lambda y,H: _seasonal_naive(y,H,m=seasonal_periods),
        "MovingAvg": _moving_average,
        "WMA": _wma,
        "SES": _ses,
        "Holt": _holt,
        "LinearTrend": _linear_trend,
        "Croston": _croston,
    }

# Backtesting + torneo
def time_series_cv(series, h=3, initial=18, step=1, m=12):
    y=series.dropna()
    if len(y)<max(initial,h)+2:
        return {"best": None, "metrics": {}}
    methods = candidate_methods(m)
    errs={k:[] for k in methods}
    for end in range(initial, len(y)-h+1, step):
        tr=y.iloc[:end]; ts=y.iloc[end:end+h]
        for name,fn in methods.items():
            try:
                pred = fn(tr, h)
            except Exception:
                pred = np.repeat(np.nan, h)
            errs[name].append(smape(ts.values, pred))
    avg = {k: np.nanmean(v) for k,v in errs.items()}
    best = min(avg, key=avg.get)
    return {"best": best, "metrics": avg}

# Pronóstico de TODOS los métodos (con captura de errores)
def forecast_all(series, horizon, seasonal_periods=12):
    methods = candidate_methods(seasonal_periods)
    idx = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    df_all = pd.DataFrame({"fecha": idx})
    errors = {}
    error_msgs = {}

    # backtesting (mismas condiciones que el best)
    mets = time_series_cv(series, h=min(horizon, 6),
                          initial=min(18, max(12, len(series)//2)),
                          m=seasonal_periods)
    errors = mets.get("metrics", {}) if mets else {}
    best   = mets.get("best") if mets else None

    for name, fn in methods.items():
        try:
            df_all[name] = fn(series, horizon)
        except Exception as e:
            df_all[name] = np.nan
            error_msgs[name] = str(e)

    return best, errors, error_msgs, df_all

# =============================
# Sidebar: carga y mapeo
# =============================
st.sidebar.header("1) Cargar archivo")
file = st.sidebar.file_uploader("Excel (.xlsx/.xls) o CSV", type=["xlsx","xls","csv"])
if not file:
    st.info("Subí un archivo para comenzar.")
    st.stop()

raw = pd.read_excel(file) if file.name.lower().endswith(("xlsx","xls")) else pd.read_csv(file)
st.sidebar.header("2) Mapear columnas")
cols = raw.columns.tolist()
fecha_col    = st.sidebar.selectbox("Fecha", cols)
servicio_col = st.sidebar.selectbox("Servicio", cols)
cantidad_col = st.sidebar.selectbox("Cantidad", cols)
precio_col   = st.sidebar.selectbox("Precio", cols)

try:
    dfm = coerce_schema(raw, {"fecha":fecha_col,"servicio":servicio_col,"cantidad":cantidad_col,"precio":precio_col})
    st.sidebar.success("Esquema normalizado ✅")
except Exception as e:
    st.sidebar.error(f"Error: {e}")
    st.stop()

# =============================
# EDA rápida
# =============================
st.header("Exploración")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Servicios (total)", f"{int(dfm['cantidad'].sum()):,}")
c2.metric("Ingresos (total)", f"₡{float(dfm['ingreso'].sum()):,.0f}")
c3.metric("Meses", dfm["fecha"].nunique())
c4.metric("Servicios únicos", dfm["servicio"].nunique())

fig_total = px.line(dfm.groupby("fecha", as_index=False)["cantidad"].sum(),
                    x="fecha", y="cantidad", markers=True, title="Demanda total mensual")
st.plotly_chart(fig_total, use_container_width=True)

# =============================
# Pronóstico
# =============================
st.header("Pronóstico de demanda")
nivel = st.radio("Nivel", ["Por servicio", "Por tipo (Lavado/Reparación)", "Total"], index=0)
horizonte = st.selectbox("Horizonte (meses)", [3,6,9,12], index=1)

def export_excel_all(df_methods_dict, metrics, filename):
    """df_methods_dict: {metodo: DataFrame(fecha, yhat)}, metrics: {metodo: sMAPE}"""
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        # Ranking de métricas
        met_df = pd.DataFrame(sorted(metrics.items(), key=lambda x: x[1]), columns=["Modelo","sMAPE"])
        met_df.to_excel(writer, index=False, sheet_name="metricas")
        # Una hoja por método
        for name, dfmth in df_methods_dict.items():
            # Agregar sMAPE en la hoja (encabezado)
            # Escribimos una fila con el valor y luego la tabla
            sheet = name[:31] if len(name)>31 else name  # Excel sheet name limit
            start_row = 0
            tmp = pd.DataFrame({"sMAPE":[metrics.get(name, np.nan)]})
            tmp.to_excel(writer, index=False, sheet_name=sheet, startrow=start_row)
            dfmth.to_excel(writer, index=False, sheet_name=sheet, startrow=start_row+2)
    return bio.getvalue()

def show_all_methods_ui(y, title_prefix=""):
    """Pronostica con todos los métodos, muestra: best, métricas, errores y pestañas con gráfica+tabla por método.
       Retorna dict {metodo: df(fecha,yhat)} y dict metrics."""
    best, errs, error_msgs, df_all = forecast_all(y, horizonte, seasonal_periods=12)

    # Métricas (tabla)
    if errs:
        st.subheader("Métricas (sMAPE ↓)")
        met_df = pd.DataFrame(sorted(errs.items(), key=lambda x: x[1]), columns=["Modelo","sMAPE"])
        st.dataframe(met_df, use_container_width=True)
        st.success(f"Mejor modelo: {best}")
    else:
        st.info("Serie corta para backtesting.")

    # Tabs por método
    dfs_por_metodo = {}
    tabs = st.tabs(list(df_all.columns[df_all.columns!='fecha']))
    hist_df = y.reset_index().rename(columns={y.name:"y"})
    for tab, metodo in zip(tabs, [c for c in df_all.columns if c!='fecha']):
        with tab:
            if metodo in error_msgs:
                st.error(f"Error en {metodo}: {error_msgs[metodo]}")
            this = df_all[["fecha", metodo]].rename(columns={metodo:"yhat"})
            dfs_por_metodo[metodo] = this.copy()
            # Gráfica
            plot_df = pd.concat([
                hist_df.rename(columns={"y":"valor"}).assign(serie="histórico"),
                this.rename(columns={"yhat":"valor"}).assign(serie=f"pronóstico ({metodo})")
            ], ignore_index=True)
            fig = px.line(plot_df, x="fecha", y="valor", color="serie",
                          title=f"{title_prefix} – {metodo}")
            st.plotly_chart(fig, use_container_width=True)
            # Tabla de Y de este método
            st.dataframe(this, use_container_width=True)
            # sMAPE del método
            sm = errs.get(metodo, np.nan)
            st.caption(f"sMAPE {metodo}: {sm:.3f}" if pd.notna(sm) else f"sMAPE {metodo}: N/D")

    return dfs_por_metodo, errs

# ---- POR SERVICIO ----
if nivel == "Por servicio":
    servicios = sorted(dfm["servicio"].unique())
    sel_all = st.checkbox("Seleccionar todos los servicios", value=False)
    seleccion = servicios if sel_all else st.multiselect("Elegí servicio(s)", servicios)
    if not seleccion:
        st.stop()

    excel_buffers = []  # para juntar descargas por cada servicio
    for s in seleccion:
        st.subheader(f"🔧 {s}")
        g = dfm[dfm["servicio"]==s].sort_values("fecha")
        y = g.set_index("fecha")["cantidad"]
        dfs_por_metodo, metrics = show_all_methods_ui(y, title_prefix=s)
        # botón de descarga por servicio (una hoja por método)
        data = export_excel_all(dfs_por_metodo, metrics, f"pronostico_{s}.xlsx")
        st.download_button(
            f"💾 Descargar Excel (métodos) – {s}",
            data=data,
            file_name=f"pronostico_{s}_{datetime.now().date()}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

# ---- POR TIPO ----
elif nivel == "Por tipo (Lavado/Reparación)":
    tipos = st.multiselect("Elegí tipo(s)", ["Lavado","Reparación"], default=["Lavado","Reparación"])
    if not tipos: st.stop()
    for t in tipos:
        st.subheader(f"🧩 {t}")
        g = dfm[dfm["tipo_servicio"]==t].groupby("fecha", as_index=False)["cantidad"].sum()
        y = g.set_index("fecha")["cantidad"]
        dfs_por_metodo, metrics = show_all_methods_ui(y, title_prefix=t)
        data = export_excel_all(dfs_por_metodo, metrics, f"pronostico_{t}.xlsx")
        st.download_button(
            f"💾 Descargar Excel (métodos) – {t}",
            data=data,
            file_name=f"pronostico_{t}_{datetime.now().date()}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

# ---- TOTAL ----
else:
    st.subheader("📦 TOTAL")
    g = dfm.groupby("fecha", as_index=False)["cantidad"].sum()
    y = g.set_index("fecha")["cantidad"]
    dfs_por_metodo, metrics = show_all_methods_ui(y, title_prefix="TOTAL")
    data = export_excel_all(dfs_por_metodo, metrics, "pronostico_total.xlsx")
    st.download_button(
        "💾 Descargar Excel (métodos) – TOTAL",
        data=data,
        file_name=f"pronostico_total_{datetime.now().date()}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

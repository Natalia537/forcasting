# app.py — Pronóstico Taller Moto (con comparación de modelos)
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
Subí un **Excel/CSV** con columnas:
- **fecha**
- **servicio**
- **cantidad**
- **precio**

La app limpia → agrega a mensual → hace análisis → pronósticos por servicio / tipo / total,
usando métodos livianos (Naive, Seasonal Naive, MA, WMA, SES, Holt, LinearTrend y Croston).
Ahora también podés comparar **todos los modelos** gráficamente y en tabla.
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
    df["cantidad"] = pd.to_numeric(df["cantidad"], errors="coerce").fillna(0)
    df["precio"] = pd.to_numeric(df["precio"], errors="coerce").fillna(0)

    df["fecha_mes"] = df["fecha"].dt.to_period("M").dt.to_timestamp()
    g = df.groupby(["servicio","fecha_mes"], as_index=False).agg({"cantidad":"sum","precio":"mean"})
    g = g.rename(columns={"fecha_mes":"fecha"})

    out = []
    for svc, gi in g.groupby("servicio"):
        idx = pd.date_range(gi["fecha"].min(), gi["fecha"].max(), freq="MS")
        gi2 = gi.set_index("fecha").reindex(idx).rename_axis("fecha").reset_index()
        gi2["servicio"] = svc
        gi2["cantidad"] = gi2["cantidad"].fillna(0)
        gi2["precio"] = gi2["precio"].ffill().bfill()
        out.append(gi2)

    dfm = pd.concat(out)
    dfm["tipo_servicio"] = dfm["servicio"].map(SERVICIO_TIPO)
    dfm["ingreso"] = dfm["cantidad"] * dfm["precio"]
    return dfm

# =============================
# Modelos de pronóstico
# =============================
def smape(y_true, y_pred):
    denom = np.abs(y_true)+np.abs(y_pred)
    denom = np.where(denom==0,1,denom)
    return 100*np.mean(2*np.abs(y_true-y_pred)/denom)

def _naive(y,h): return np.repeat(y.iloc[-1],h)
def _seasonal_naive(y,h,m=12):
    if len(y)<m: return _naive(y,h)
    return np.tile(y.iloc[-m:],int(np.ceil(h/m)))[:h]
def _moving_average(y,h,w=3): return np.repeat(y.rolling(min(w,len(y))).mean().iloc[-1],h)
def _wma(y,h,weights=(0.6,0.3,0.1)):
    w=np.array(weights[:min(len(weights),len(y))]);w=w/w.sum()
    val=np.dot(y.iloc[-len(w):][::-1],w)
    return np.repeat(val,h)
def _ses(y,h,alpha=0.3):
    lvl=y.iloc[0]
    for t in range(1,len(y)): lvl=alpha*y.iloc[t]+(1-alpha)*lvl
    return np.repeat(lvl,h)
def _holt(y,h,a=0.3,b=0.1):
    l=y.iloc[0];t=y.iloc[1]-y.iloc[0] if len(y)>1 else 0
    for i in range(1,len(y)):
        prev=l;l=a*y.iloc[i]+(1-a)*(l+t);t=b*(l-prev)+(1-b)*t
    return np.array([l+(i+1)*t for i in range(h)])
def _linear_trend(y,h):
    if len(y)<2: return _naive(y,h)
    x=np.arange(len(y));a,b=np.polyfit(x,y,1)
    return a*np.arange(len(y),len(y)+h)+b
def _croston(y,h,a=0.3):
    y=y.fillna(0).astype(float);z=p=None;q=0
    for v in y:
        if v>0:
            z=v if z is None else a*v+(1-a)*z
            p=1 if p is None else a*q+(1-a)*p
            q=1
        else:
            q=(q or 0)+1
    if z is None:return np.zeros(h)
    if not p:p=1
    rate=z/p
    return np.repeat(rate,h)

def time_series_cv(series,h=3,initial=18,step=1,m=12):
    y=series.dropna()
    if len(y)<max(initial,h)+2:return{"best":None,"metrics":{}}
    methods={
        "Naive":_naive,"SeasonalNaive":lambda tr,H:_seasonal_naive(tr,H,m),
        "MovingAvg":_moving_average,"WMA":_wma,"SES":_ses,"Holt":_holt,
        "LinearTrend":_linear_trend,"Croston":_croston
    }
    errs={k:[] for k in methods}
    for end in range(initial,len(y)-h+1,step):
        tr=y.iloc[:end];ts=y.iloc[end:end+h]
        for name,fn in methods.items():
            try:pred=fn(tr,h)
            except:pred=np.repeat(np.nan,h)
            errs[name].append(smape(ts,pred))
    avg={k:np.nanmean(v) for k,v in errs.items()}
    best=min(avg,key=avg.get)
    return{"best":best,"metrics":avg}

def fit_and_forecast(series,h=6,m=12):
    res=time_series_cv(series,h,min(18,max(12,len(series)//2)))
    best=res["best"];metrics=res["metrics"]
    method={
        "Naive":_naive,"SeasonalNaive":lambda y,H:_seasonal_naive(y,H,m),
        "MovingAvg":_moving_average,"WMA":_wma,"SES":_ses,"Holt":_holt,
        "LinearTrend":_linear_trend,"Croston":_croston
    }.get(best,_seasonal_naive)
    fc=method(series,h)
    idx=pd.date_range(series.index[-1]+pd.offsets.MonthBegin(1),periods=h,freq="MS")
    return metrics,pd.DataFrame({"fecha":idx,"yhat":fc})

# --- función que genera pronósticos de todos los modelos ---
def forecast_all_methods(series,h,m=12):
    methods={
        "Naive":_naive,"SeasonalNaive":lambda y,H:_seasonal_naive(y,H,m),
        "MovingAvg":_moving_average,"WMA":_wma,"SES":_ses,"Holt":_holt,
        "LinearTrend":_linear_trend,"Croston":_croston
    }
    idx=pd.date_range(series.index[-1]+pd.offsets.MonthBegin(1),periods=h,freq="MS")
    df=pd.DataFrame({"fecha":idx})
    res=time_series_cv(series,h,min(18,max(12,len(series)//2)))
    for name,fn in methods.items():
        try:df[name]=fn(series,h)
        except:df[name]=np.nan
    return res["best"],res["metrics"],df

# =============================
# Sidebar
# =============================
st.sidebar.header("1) Cargar archivo")
file=st.sidebar.file_uploader("Excel o CSV",type=["xlsx","xls","csv"])
if not file:st.stop()

df=pd.read_excel(file) if file.name.endswith(("xls","xlsx")) else pd.read_csv(file)
cols=df.columns.tolist()

st.sidebar.header("2) Mapear columnas")
fecha=st.sidebar.selectbox("Fecha",cols)
serv=st.sidebar.selectbox("Servicio",cols)
cant=st.sidebar.selectbox("Cantidad",cols)
prec=st.sidebar.selectbox("Precio",cols)

dfm=coerce_schema(df,{"fecha":fecha,"servicio":serv,"cantidad":cant,"precio":prec})

# =============================
# EDA
# =============================
st.header("Exploración")
c1,c2,c3=st.columns(3)
c1.metric("Total servicios",f"{dfm['cantidad'].sum():,.0f}")
c2.metric("Total ingresos",f"₡{dfm['ingreso'].sum():,.0f}")
c3.metric("Meses",len(dfm["fecha"].unique()))

fig=px.line(dfm.groupby("fecha",as_index=False)["cantidad"].sum(),
            x="fecha",y="cantidad",title="Demanda total mensual")
st.plotly_chart(fig,use_container_width=True)

# =============================
# Pronóstico
# =============================
st.header("Pronóstico de demanda")
show_compare=st.checkbox("🔬 Comparar todos los modelos (gráfica y tabla)",value=False)
nivel=st.radio("Nivel",["Por servicio","Por tipo (Lavado/Reparación)","Total"],index=0)

def to_excel(df,name="pronostico"):
    buf=io.BytesIO()
    with pd.ExcelWriter(buf,engine="xlsxwriter") as w:df.to_excel(w,index=False,sheet_name=name)
    return buf.getvalue()

# ---- POR SERVICIO ----
if nivel=="Por servicio":
    servicios=dfm["servicio"].unique()
    sel_all=st.checkbox("Seleccionar todos",False)
    svcs=servicios if sel_all else st.multiselect("Elegí servicio(s)",servicios)
    horizonte=st.selectbox("Horizonte", [3,6,9,12],1)
    if not svcs:st.stop()
    for s in svcs:
        st.subheader(f"🔧 {s}")
        g=dfm[dfm["servicio"]==s].sort_values("fecha")
        y=g.set_index("fecha")["cantidad"]
        metrics,df_fc=fit_and_forecast(y,horizonte)
        hist=g[["fecha","cantidad"]].rename(columns={"cantidad":"y"});hist["serie"]="histórico"
        fc=df_fc.rename(columns={"yhat":"y"});fc["serie"]="pronóstico"
        plot=pd.concat([hist,fc])
        fig=px.line(plot,x="fecha",y="y",color="serie",title=f"{s} – histórico vs pronóstico ({horizonte} m)")
        st.plotly_chart(fig,use_container_width=True)
        st.write("Mejor modelo:",min(metrics,key=metrics.get))
        if show_compare:
            best,errs,df_all=forecast_all_methods(y,horizonte)
            plot_m=df_all.melt("fecha",var_name="modelo",value_name="yhat")
            fig2=px.line(plot_m,x="fecha",y="yhat",color="modelo",title=f"{s} – comparación de modelos")
            fig2.add_scatter(x=hist["fecha"],y=hist["y"],mode="lines+markers",name="histórico",line=dict(dash="dot"))
            st.plotly_chart(fig2,use_container_width=True)
            if errs:
                st.dataframe(pd.DataFrame(sorted(errs.items(),key=lambda x:x[1]),columns=["Modelo","sMAPE"]))
            st.dataframe(df_all,use_container_width=True)

# ---- POR TIPO ----
elif nivel=="Por tipo (Lavado/Reparación)":
    tipos=st.multiselect("Elegí tipo(s)",["Lavado","Reparación"],default=["Lavado","Reparación"])
    horizonte=st.selectbox("Horizonte", [3,6,9,12],1)
    for t in tipos:
        st.subheader(f"🧩 {t}")
        g=dfm[dfm["tipo_servicio"]==t].groupby("fecha",as_index=False)["cantidad"].sum()
        y=g.set_index("fecha")["cantidad"]
        metrics,df_fc=fit_and_forecast(y,horizonte)
        hist=g.rename(columns={"cantidad":"y"});hist["serie"]="histórico"
        fc=df_fc.rename(columns={"yhat":"y"});fc["serie"]="pronóstico"
        fig=px.line(pd.concat([hist,fc]),x="fecha",y="y",color="serie",title=f"{t} – histórico vs pronóstico")
        st.plotly_chart(fig,use_container_width=True)
        st.write("Mejor modelo:",min(metrics,key=metrics.get))
        if show_compare:
            best,errs,df_all=forecast_all_methods(y,horizonte)
            plot_m=df_all.melt("fecha",var_name="modelo",value_name="yhat")
            fig2=px.line(plot_m,x="fecha",y="yhat",color="modelo",title=f"{t} – comparación de modelos")
            fig2.add_scatter(x=hist["fecha"],y=hist["y"],mode="lines+markers",name="histórico",line=dict(dash="dot"))
            st.plotly_chart(fig2,use_container_width=True)
            if errs:
                st.dataframe(pd.DataFrame(sorted(errs.items(),key=lambda x:x[1]),columns=["Modelo","sMAPE"]))
            st.dataframe(df_all,use_container_width=True)

# ---- TOTAL ----
else:
    horizonte=st.selectbox("Horizonte", [3,6,9,12],1)
    g=dfm.groupby("fecha",as_index=False)["cantidad"].sum()
    y=g.set_index("fecha")["cantidad"]
    metrics,df_fc=fit_and_forecast(y,horizonte)
    hist=g.rename(columns={"cantidad":"y"});hist["serie"]="histórico"
    fc=df_fc.rename(columns={"yhat":"y"});fc["serie"]="pronóstico"
    fig=px.line(pd.concat([hist,fc]),x="fecha",y="y",color="serie",title=f"TOTAL – histórico vs pronóstico ({horizonte} m)")
    st.plotly_chart(fig,use_container_width=True)
    st.write("Mejor modelo:",min(metrics,key=metrics.get))
    if show_compare:
        best,errs,df_all=forecast_all_methods(y,horizonte)
        plot_m=df_all.melt("fecha",var_name="modelo",value_name="yhat")
        fig2=px.line(plot_m,x="fecha",y="yhat",color="modelo",title="TOTAL – comparación de modelos")
        fig2.add_scatter(x=hist["fecha"],y=hist["y"],mode="lines+markers",name="histórico",line=dict(dash="dot"))
        st.plotly_chart(fig2,use_container_width=True)
        if errs:
            st.dataframe(pd.DataFrame(sorted(errs.items(),key=lambda x:x[1]),columns=["Modelo","sMAPE"]))
        st.dataframe(df_all,use_container_width=True)

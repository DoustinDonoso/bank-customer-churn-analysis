import functools

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import qualitative

# ----------------------------------------------------------------------------- data

CLEAN_CSV = "Customer-Churn-Records-clean.csv"
SEGMENTS_CSV = "Customer-Risk-Value-Segments.csv"

PALETTE = {
    "bg": "#0f1420",
    "card": "#171e2e",
    "ink": "#e8edf7",
    "muted": "#98a4bd",
    "accent": "#4f8cff",
    "good": "#2ecc8f",
    "warn": "#f2b13c",
    "bad": "#ff5c7c",
    "grid": "rgba(152,164,189,0.14)",
}

RISK_COLORS = {"Riesgo alto": PALETTE["bad"], "Riesgo medio": PALETTE["warn"], "Riesgo bajo": PALETTE["good"]}


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    raw = pd.read_csv(CLEAN_CSV)
    seg = pd.read_csv(SEGMENTS_CSV)
    extra = [
        "Surname", "CreditScore", "Tenure", "EstimatedSalary", "HasCrCard", "Complain",
        "Satisfaction Score", "Card Type", "Point Earned",
    ]
    # `Balance`, `Age`, `Geography`, `Gender`, `NumOfProducts`, `IsActiveMember` y `Exited`
    # ya vienen en el CSV de segmentos: repetirlos crearía columnas _x/_y tras el merge.
    df = seg.merge(raw[["CustomerId"] + extra], on="CustomerId", how="inner")
    df["Satisfaction Score"] = df["Satisfaction Score"].astype(float)
    df["HasBalance"] = df["Balance"] > 0
    return df


df = load_data()


# ----------------------------------------------------------------------------- theme

def base_layout(fig: go.Figure, height: int = 320, legend: bool = False) -> go.Figure:
    fig.update_layout(
        template=None,
        height=height,
        margin=dict(l=8, r=8, t=38, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["ink"], size=12.5, family="Inter, system-ui, sans-serif"),
        title=dict(x=0, xanchor="left", font=dict(size=15, color=PALETTE["ink"])),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#1e2740", bordercolor=PALETTE["grid"], font=dict(color=PALETTE["ink"])),
    )
    fig.update_xaxes(gridcolor=PALETTE["grid"], zeroline=False, linecolor=PALETTE["grid"], tickfont=dict(size=11))
    fig.update_yaxes(gridcolor=PALETTE["grid"], zeroline=False, linecolor=PALETTE["grid"], tickfont=dict(size=11))
    return fig


def style_css() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{ background: radial-gradient(1200px 620px at 12% -8%, #1b2440 0%, {PALETTE['bg']} 55%) !important; }}
        section[data-testid="stSidebar"] {{ background: {PALETTE['card']} !important; }}
        section[data-testid="stSidebar"] * {{ color: {PALETTE['ink']} !important; }}
        h1, h2, h3, h4 {{ color: {PALETTE['ink']} !important; font-weight: 700 !important; letter-spacing: -0.02em; }}
        [data-testid="stMetricValue"] {{ color: {PALETTE['ink']} !important; font-weight: 700 !important; }}
        [data-testid="stMetricLabel"] {{ color: {PALETTE['muted']} !important; text-transform: uppercase; letter-spacing: 0.06em; font-size: 0.74rem !important; }}
        div[data-testid="stMetric"] {{ background: {PALETTE['card']}; border: 1px solid {PALETTE['grid']};
            border-radius: 14px; padding: 16px 18px; box-shadow: 0 6px 22px rgba(0,0,0,0.28); }}
        [data-testid="stPlotlyChart"], [data-testid="stTable"], .stDataFrame {{
            background: {PALETTE['card']}; border: 1px solid {PALETTE['grid']}; border-radius: 14px; padding: 8px; }}
        .qcard {{ background: {PALETTE['card']}; border: 1px solid {PALETTE['grid']}; border-radius: 16px;
            padding: 18px 20px; margin-bottom: 14px; box-shadow: 0 8px 26px rgba(0,0,0,0.30); }}
        .qtitle {{ color: {PALETTE['muted']}; font-size: 0.72rem; text-transform: uppercase;
            letter-spacing: 0.12em; font-weight: 700; margin-bottom: 8px; }}
        .qbig {{ color: {PALETTE['ink']}; font-size: 1.9rem; font-weight: 700; line-height: 1.15; }}
        .qsub {{ color: {PALETTE['muted']}; font-size: 0.84rem; margin-top: 5px; }}
        .badge {{ display: inline-block; padding: 5px 13px; border-radius: 999px; font-size: 0.78rem;
            font-weight: 700; letter-spacing: 0.02em; color: #0d111c; white-space: nowrap; flex: 0 0 auto; }}
        .lede {{ color: {PALETTE['muted']}; font-size: 0.95rem; margin-bottom: 4px; }}
        ul.actions {{ margin: 6px 0 0; padding-left: 0; list-style: none; }}
        ul.actions li {{ padding: 9px 0; border-bottom: 1px solid {PALETTE['grid']}; color: {PALETTE['ink']}; font-size: 0.92rem; }}
        ul.actions li:last-child {{ border-bottom: none; }}
        .flag {{ color: {PALETTE['bad']}; font-weight: 700; }}
        .ok {{ color: {PALETTE['good']}; font-weight: 700; }}
        .stTabs [data-baseweb="tab"] {{ color: {PALETTE['muted']}; font-weight: 600; }}
        .stTabs [aria-selected="true"] {{ color: {PALETTE['ink']} !important; border-bottom: 2px solid {PALETTE['accent']} !important; }}
        div[data-testid="stTextInput"] input, .stSelectbox div[data-baseweb="select"] > div {{
            background: {PALETTE['card']} !important; color: {PALETTE['ink']} !important; border-color: {PALETTE['grid']} !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="Retención de clientes · Riesgo y valor", layout="wide")
style_css()

st.markdown(
    "<div class='qtitle' style='margin-bottom:2px'>Banca · Analítica de retención</div>"
    "<h1 style='margin-top:0'>Riesgo de abandono y valor del cliente</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='lede'>Puntaje predictivo <code>p(churn)</code>, pérdida esperada "
    "<code>P(abandono) × Balance</code> y segmentación valor × riesgo sobre 10.000 relaciones.</div>",
    unsafe_allow_html=True,
)


def fmt_eur(v: float) -> str:
    return f"€{v:,.0f}".replace(",", ".")


def fmt_pct(v: float, digits: int = 1) -> str:
    return f"{v * 100:.{digits}f}%".replace(".", ",")


def badge(text: str, color: str) -> str:
    return f"<span class='badge' style='background:{color}'>{text}</span>"


# ----------------------------------------------------------------------------- charts

def chart_kpi_bar(frame: pd.DataFrame, by: str, order: list[str], title: str, color: str = PALETTE["accent"]) -> go.Figure:
    g = frame.groupby(by, observed=True).agg(churn=("Exited", "mean"), n=("Exited", "size")).reindex(order)
    fig = go.Figure(
        go.Bar(
            x=g.index.astype(str),
            y=g["churn"] * 100,
            marker=dict(color=color, line_width=0),
            text=[f"{v:.1f}%<br><span style='font-size:10px;color:{PALETTE['muted']}'>n={int(n)}</span>"
                  for v, n in zip(g["churn"] * 100, g["n"])],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}<br>Churn %{y:.1f}%<extra></extra>",
        )
    )
    fig.update_yaxes(range=[0, max(50, (g["churn"].max() * 100) * 1.4)], ticksuffix="%")
    fig.update_layout(title=title)
    return base_layout(fig)


def chart_complain_satisfaction(frame: pd.DataFrame) -> go.Figure:
    sat = frame.copy()
    sat["sat"] = sat["Satisfaction Score"].round().astype(int)
    piv = sat.pivot_table(index="sat", columns="Complain", values="Exited", aggfunc="mean")
    fig = go.Figure()
    for flag, color, label in [(0, PALETTE["good"], "Sin queja"), (1, PALETTE["bad"], "Con queja")]:
        fig.add_trace(
            go.Scatter(
                x=piv.index.astype(str), y=piv[flag] * 100, mode="lines+markers+text",
                line=dict(color=color, width=3, shape="spline"), marker=dict(size=10, color=color),
                text=[f"{v:.1f}%" for v in piv[flag] * 100], textposition="top center",
                textfont=dict(color=color, size=11), name=label,
                hovertemplate=f"{label}<br>Satisfacción %{{x}}<br>Churn %{{y:.1f}}%<extra></extra>",
            )
        )
    fig.update_yaxes(range=[0, 118], ticksuffix="%", title=None)
    fig.update_xaxes(title="Puntuación de satisfacción (1 = insatisfecho · 5 = satisfecho)")
    return base_layout(fig, height=340, legend=True)


def chart_matrix(frame: pd.DataFrame, metric: str, colorscale: list, fmt) -> go.Figure:
    pv = frame.pivot_table(index="ValueTier", columns="RiskTier", values=metric, aggfunc="mean", observed=True)
    pv = pv.reindex(index=["Valor alto", "Valor medio", "Valor bajo"], columns=["Riesgo bajo", "Riesgo medio", "Riesgo alto"])
    fig = go.Figure(
        go.Heatmap(
            z=pv.to_numpy(), x=pv.columns.astype(str), y=pv.index.astype(str),
            colorscale=colorscale, text=[[fmt(v) for v in row] for row in pv.to_numpy()],
            texttemplate="%{text}", textfont=dict(size=14, color="#0d111c"),
            colorbar=dict(thickness=10, outlinewidth=0, tickfont=dict(size=10)),
            hovertemplate="%{y} × %{x}<br>valor %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(title=None)
    fig.update_xaxes(title="Riesgo")
    # Plotly apila la primera categoría abajo; se invierte para que "Valor alto" quede arriba.
    fig.update_yaxes(title="Valor", autorange="reversed")
    return base_layout(fig, height=300)


def chart_expected_loss(frame: pd.DataFrame) -> go.Figure:
    seg = (
        frame.groupby("Segment", observed=True)
        .agg(loss=("expected_loss", "sum"), n=("Exited", "size"), churn=("Exited", "mean"))
        .sort_values("loss")
    )
    colors = [PALETTE["bad"] if "alto riesgo" in s and "alto valor" in s else
              PALETTE["warn"] if "alto riesgo" in s else
              PALETTE["accent"] if "riesgo medio" in s else PALETTE["good"] for s in seg.index]
    fig = go.Figure(
        go.Bar(
            x=seg["loss"], y=[s.split(" (")[0] for s in seg.index], orientation="h",
            marker=dict(color=colors, line_width=0),
            text=[f"{fmt_eur(v)}<br><span style='font-size:10px'>churn {fmt_pct(c,0)} · n={int(n)}</span>"
                  for v, c, n in zip(seg["loss"], seg["churn"], seg["n"])],
            textposition="outside", cliponaxis=False,
            hovertemplate="%{y}<br>Pérdida esperada %{x:,.0f} €<extra></extra>",
        )
    )
    fig.update_xaxes(title="Pérdida esperada (€)", tickformat=",.0f", range=[0, seg["loss"].max() * 1.5])
    fig.update_layout(title=None)
    return base_layout(fig, height=330)


def chart_age_curve(frame: pd.DataFrame) -> go.Figure:
    bins = [0, 25, 30, 35, 40, 45, 50, 55, 60, 65, 100]
    tmp = frame.assign(band=pd.cut(frame["Age"], bins))
    g = tmp.groupby("band", observed=True).agg(churn=("Exited", "mean"), n=("Exited", "size"), bal=("Balance", "mean"))
    x = [str(iv) for iv in g.index]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=g["bal"], name="Saldo medio", marker=dict(color="rgba(79,140,255,0.28)", line_width=0),
                         hovertemplate="Saldo medio %{y:,.0f} €<extra></extra>", yaxis="y2"))
    fig.add_trace(go.Scatter(x=x, y=g["churn"] * 100, name="Tasa de abandono", mode="lines+markers",
                             line=dict(color=PALETTE["bad"], width=3.5, shape="spline"), marker=dict(size=9),
                             hovertemplate="Churn %{y:.1f}%<extra></extra>"))
    fig.add_vline(x=3.5, line_dash="dot", line_color=PALETTE["muted"], line_width=1.2,
                  annotation_text="punto de inflexión · 40 años", annotation_font_size=10.5,
                  annotation_font_color=PALETTE["muted"], annotation_position="top left")
    fig.update_yaxes(title="Tasa de abandono (%)", range=[0, 70])
    fig.update_layout(
        yaxis2=dict(title=None, overlaying="y", side="right", showgrid=False, tickformat=",f", range=[0, g["bal"].max() * 2.1]),
        legend=dict(orientation="h", y=-0.28, x=1, xanchor="right"),
    )
    return base_layout(fig, height=360, legend=True)


# ----------------------------------------------------------------------------- page 1

def page_overview(frame: pd.DataFrame) -> None:
    st.markdown("## Resumen ejecutivo")
    st.markdown(
        "<div class='lede'>El abandono no es homogéneo: se concentra en clientes mayores de 40 años, "
        "inactivos con saldo alto, de un solo producto y con queja abierta. La lectura de negocio es que "
        "<b>27% de la cartera concentra casi dos tercios de la pérdida esperada</b>.</div>",
        unsafe_allow_html=True,
    )

    churn = frame["Exited"].mean()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Tasa de abandono total", fmt_pct(churn), f"{int(frame['Exited'].sum()):,} abandonos".replace(",", "."), delta_color="off")
    c2.metric("Clientes analizados", f"{len(frame):,}".replace(",", "."), f"{frame['CustomerId'].nunique():,} únicos".replace(",", "."), delta_color="off")
    c3.metric("Saldo promedio", fmt_eur(frame["Balance"].mean()), f"total {fmt_eur(frame['Balance'].sum())}", delta_color="off")
    c4.metric("Satisfacción promedio", f"{frame['Satisfaction Score'].mean():.2f} / 5", "escala 1–5 · no discrimina", delta_color="off")
    c5.metric("Pérdida esperada", fmt_eur(frame["expected_loss"].sum()), f"{fmt_pct(frame['p_churn'].mul(frame['Balance']).sum() / frame['Balance'].sum())} del saldo", delta_color="off")

    st.caption(
        "Nota de lectura: la satisfacción media es prácticamente idéntica en los tres tiers de riesgo "
        "(3,04 / 3,04 / 2,97), por lo que el CSAT actual no predice el abandono; el comportamiento sí."
    )

    st.divider()
    st.markdown("### Tendencias de abandono")
    left, right = st.columns([1, 1])
    with left:
        geo = frame.groupby("Geography", observed=True).agg(churn=("Exited", "mean"), bal=("Balance", "sum"), n=("Exited", "size"))
        geo = geo.sort_values("churn", ascending=False)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=geo.index.astype(str), y=geo["churn"] * 100, name="Tasa de abandono",
            marker=dict(color=[PALETTE["bad"] if v > 0.25 else PALETTE["warn"] if v > 0.15 else PALETTE["good"]
                                for v in geo["churn"]], line_width=0),
            text=[f"{v*100:.1f}%" for v in geo["churn"]], textposition="outside", cliponaxis=False,
            hovertemplate="%{x}<br>Churn %{y:.1f}%<extra></extra>"))
        fig.update_yaxes(range=[0, 50], ticksuffix="%")
        fig.update_layout(title="Abandono por geografía · Alemania es el foco crítico")
        st.plotly_chart(base_layout(fig, 300), width="stretch")
        st.caption(
            f"Alemania concentra {fmt_eur(geo.loc['Germany','bal'])} de saldo con "
            f"{fmt_pct(geo.loc['Germany','churn'])} de abandono "
            f"({fmt_pct(geo.loc['France','churn'])} en Francia y {fmt_pct(geo.loc['Spain','churn'])} en España)."
        )

    with right:
        gen = frame[frame["Gender"].isin(["Male", "Female"])]
        fig = chart_kpi_bar(gen, "Gender", ["Female", "Male"], "Abandono por género", PALETTE["accent"])
        st.plotly_chart(base_layout(fig, 300), width="stretch")
        st.caption(
            "Las clientas abandonan más que los clientes. El género entra en el modelo como factor "
            "protector para los hombres (coef. estandarizado −0,26)."
        )

    fig = chart_kpi_bar(frame, "Card Type", ["DIAMOND", "PLATINUM", "GOLD", "SILVER"],
                        "Abandono por tipo de tarjeta · sin señal comercial relevante", PALETTE["muted"])
    st.plotly_chart(base_layout(fig, 280), width="stretch")
    st.caption(
        "El tipo de tarjeta apenas mueve la aguja: la segmentación de producto actual no captura riesgo. "
        "Ni tarjeta, ni score crediticio, ni puntos, ni antigüedad (tenure ≈ 5 años en todos los tiers) "
        "protegen frente al abandono."
    )

    st.divider()
    st.markdown("### Quejas, satisfacción y abandono")
    left, right = st.columns([1.15, 1])
    with left:
        fig = chart_complain_satisfaction(frame)
        fig.update_layout(title="Curvas casi planas por satisfacción · la queja es el detonador")
        st.plotly_chart(base_layout(fig, 340, True), width="stretch")
        st.caption(
            "La línea roja (queja registrada) se mantiene en torno al 99% de abandono en todos los niveles "
            "de satisfacción. La verde (sin queja) en torno al 0%. La satisfacción, en cambio, es casi plana "
            "en ambas líneas: el CSAT no aporta señal."
        )
    with right:
        cmp_ = frame.groupby("Complain", observed=True).agg(
            n=("Exited", "size"), churn=("Exited", "mean"), bal=("Balance", "mean"), sat=("Satisfaction Score", "mean"),
            inact=("IsActiveMember", lambda s: 1 - s.mean()), p=("p_churn", "mean"))
        for flag, label, col in [(1, "Con queja", PALETTE["bad"]), (0, "Sin queja", PALETTE["good"])]:
            r = cmp_.loc[flag]
            st.markdown(
                f"<div class='qcard'><div class='qtitle'>{label}</div>"
                f"<div class='qbig'>{fmt_pct(r['churn'])} <span style='font-size:0.9rem;color:{PALETTE['muted']}'>abandono</span></div>"
                f"<div class='qsub'>{int(r['n']):,} clientes · saldo medio {fmt_eur(r['bal'])} · {fmt_pct(r['inact'],0)} inactivos · "
                f"satisfacción {r['sat']:.2f}/5 · p(churn) medio {fmt_pct(r['p'],0)}</div></div>".replace(",", "."),
                unsafe_allow_html=True,
            )
        st.markdown(
            f"<div class='qcard' style='border-left:3px solid {PALETTE['warn']}'>"
            f"<div class='qtitle'>Cómo actuar sobre esto</div>"
            f"<div class='qsub'>En este dataset <code>Complain</code> está casi alineado con el resultado "
            f"(99,5% de abandono con queja), por lo que se excluyó del entrenamiento como fuga de información. "
            f"Su valor real es operativo: <b>la queja no se predice, se atiende</b>. Ventanilla de rescate en ≤48h "
            f"para todo cliente con queja abierta y saldo &gt; €50k, con escalado automático a las 72h.</div></div>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("### Segmentación valor × riesgo")
    left, mid, right = st.columns([1, 1, 1.25])
    with left:
        st.markdown("##### Churn observado")
        fig = chart_matrix(frame, "Exited", [[0, "#2ecc8f"], [0.5, "#f2b13c"], [1, "#ff5c7c"]], lambda v: f"{v*100:.0f}%")
        st.plotly_chart(base_layout(fig, 300), width="stretch")
    with mid:
        st.markdown("##### P(churn) predicho")
        fig = chart_matrix(frame, "p_churn", [[0, "#4f8cff"], [0.5, "#8f6df5"], [1, "#ff5c7c"]], lambda v: f"{v*100:.0f}%")
        st.plotly_chart(base_layout(fig, 300), width="stretch")
    with right:
        st.markdown("##### Pérdida esperada por segmento")
        fig = chart_expected_loss(frame)
        st.plotly_chart(base_layout(fig, 330), width="stretch")

    hi = frame[frame["RiskTier"] == "Riesgo alto"]
    st.info(
        f"**Triaje de carteras.** Los segmentos de alto riesgo con saldo ('Proteger' + 'Retener selectivo') "
        f"suman {len(hi[hi['Balance']>0]):,} clientes ({frame['RiskTier'].eq('Riesgo alto').mean()*100:.1f}% de la cartera), "
        f"el 58,8% de todos los abandonos y el 64,5% de la pérdida esperada. El segmento 'Gestionar coste' "
        f"aglutina {len(frame[frame['Segment'].str.contains('Gestionar')]):,} clientes con 52,9% de abandono "
        f"y saldo exactamente €0: no generan pérdida esperada y no deben recibir presupuesto de retención."
        .replace(",", "."),
    )

    st.divider()
    st.markdown("### Perfil del cliente en riesgo alto")
    left, right = st.columns([1.3, 1])
    with left:
        fig = chart_age_curve(frame)
        fig.update_layout(title="El abandono se decide a partir de los 40")
        st.plotly_chart(base_layout(fig, 360, True), width="stretch")
    with right:
        prof = frame.groupby("RiskTier", observed=True).agg(
            churn=("Exited", "mean"), inact=("IsActiveMember", lambda s: 1 - s.mean()),
            q=("Complain", "mean"), age=("Age", "mean"), bal=("Balance", "mean"), prod=("NumOfProducts", "mean"))
        prof = prof.reindex(["Riesgo bajo", "Riesgo medio", "Riesgo alto"])
        rows = "".join(
            f"<tr><td style='padding:7px 10px;border-bottom:1px solid {PALETTE['grid']};color:{RISK_COLORS[i]};font-weight:700'>{i}</td>"
            + "".join(
                f"<td style='padding:7px 10px;border-bottom:1px solid {PALETTE['grid']};text-align:right'>{f(v)}</td>"
                for v, f in [(r["churn"], lambda v: fmt_pct(v, 1)), (r["inact"], lambda v: fmt_pct(v, 0)),
                             (r["q"], lambda v: fmt_pct(v, 0)), (r["age"], lambda v: f"{v:.0f}"),
                             (r["bal"], lambda v: fmt_eur(v)), (r["prod"], lambda v: f"{v:.2f}")])
            + "</tr>"
            for i, r in prof.iterrows()
        )
        st.markdown(
            f"<div class='qcard'><div class='qtitle'>Retrato por tier de riesgo</div>"
            f"<table style='width:100%;border-collapse:collapse;font-size:0.85rem'>"
            f"<tr style='color:{PALETTE['muted']};font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em'>"
            f"<th style='text-align:left;padding:6px 10px'>Tier</th><th style='padding:6px 10px'>Churn</th>"
            f"<th style='padding:6px 10px'>Inactivos</th><th style='padding:6px 10px'>Con queja</th>"
            f"<th style='padding:6px 10px'>Edad</th><th style='padding:6px 10px'>Saldo</th>"
            f"<th style='padding:6px 10px'>Prod.</th></tr>{rows}</table></div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "66% de los clientes de riesgo alto están inactivos frente al 31% de riesgo bajo; el saldo medio "
            "crece con el riesgo (€97.080 frente a €48.607): la cartera que se va es la que más dinero tiene."
        )

    left, mid, right = st.columns(3)
    with left:
        sub = frame[frame["IsActiveMember"] == 0]
        fig = chart_kpi_bar(sub, "HasBalance", [False, True], "Inactivos: saldo alto dispara el riesgo", PALETTE["bad"])
        fig.update_xaxes(tickvals=[0, 1], ticktext=["Sin saldo", "Con saldo"])
        st.plotly_chart(base_layout(fig, 290), width="stretch")
    with mid:
        g = frame.groupby("NumOfProducts", observed=True).agg(churn=("Exited", "mean"), n=("Exited", "size"))
        g = g[g["n"] > 100]
        fig = go.Figure(go.Bar(
            x=[f"{p} producto{'s' if p>1 else ''}" for p in g.index], y=g["churn"] * 100,
            marker=dict(color=[PALETTE["warn"], PALETTE["good"], PALETTE["bad"], PALETTE["bad"]][: len(g)], line_width=0),
            text=[f"{v:.1f}%<br><span style='font-size:10px;color:{PALETTE['muted']}'>n={int(n)}</span>"
                  for v, n in zip(g["churn"] * 100, g["n"])], textposition="outside", cliponaxis=False,
            hovertemplate="%{x}<br>Churn %{y:.1f}%<extra></extra>"))
        fig.update_yaxes(range=[0, 118], ticksuffix="%")
        fig.update_layout(title="El segundo producto retiene; el tercero rompe")
        st.plotly_chart(base_layout(fig, 290), width="stretch")
    with right:
        g = frame[frame["Geography"] == "Germany"].groupby("HasBalance", observed=True)["Exited"].mean()
        fig = chart_kpi_bar(frame[frame["Geography"] == "Germany"], "IsActiveMember", [0, 1],
                           "Alemania · inactividad y abandono", PALETTE["accent"])
        fig.update_xaxes(tickvals=[0, 1], ticktext=["Inactivo", "Activo"])
        st.plotly_chart(base_layout(fig, 290), width="stretch")

    st.divider()
    st.markdown("### Cartera priorizada por pérdida esperada")
    frame_out = frame.sort_values("expected_loss", ascending=False).head(15)[
        ["CustomerId", "Surname", "Geography", "Gender", "Age", "Balance", "NumOfProducts",
         "IsActiveMember", "Complain", "p_churn", "expected_loss", "Segment"]]
    st.dataframe(
        frame_out, width="stretch", hide_index=True,
        column_config={
            "CustomerId": st.column_config.NumberColumn("Cliente", format="%d"),
            "Balance": st.column_config.NumberColumn("Saldo", format="€ %d"),
            "expected_loss": st.column_config.NumberColumn("Pérdida esperada", format="€ %d"),
            "p_churn": st.column_config.ProgressColumn("P(abandono)", min_value=0, max_value=1, format="percent"),
            "Age": st.column_config.NumberColumn("Edad"),
            "IsActiveMember": st.column_config.CheckboxColumn("Activo"),
            "Complain": st.column_config.CheckboxColumn("Queja"),
        },
    )
    st.caption("Estos 15 clientes suman más pérdida esperada que todo el segmento 'Cultivar' junto: la prioridad comercial se define en euros, no en número de relaciones.")


# ----------------------------------------------------------------------------- page 2

def recommendations(c: pd.Series) -> list[tuple[str, str, str]]:
    """(severidad, acción, motivo) con severidad ∈ {bad, warn, good}."""
    out = []
    risk = c["RiskTier"]
    bal = c["Balance"]
    inact = c["IsActiveMember"] == 0
    age = c["Age"]
    prod = c["NumOfProducts"]

    if c["Complain"] == 1:
        out.append(("bad",
                    "Cerrar la queja abierta antes de cualquier oferta comercial",
                    "Cliente con queja registrada: con queja abierta, el abandono observado es del 99,5%. "
                    "Proponer valor con el expediente abierto multiplica la salida."))
    if inact and bal > 50_000:
        out.append(("bad",
                    "Llamada de gestor dedicada en 15 días + domiciliación de ingresos",
                    f"Inactivo con {fmt_eur(bal)} en cuenta: este perfil abandona al 31,7% (3,3× el de un "
                    "activo sin saldo). El objetivo no es el contacto, es recuperar una transacción recurrente."))
    elif inact:
        out.append(("warn",
                    "Campaña digital de reactivación de bajo coste",
                    "Inactivo sin saldo relevante: no justifica coste humano; migrar a autogestión con estímulo digital."))
    if 38 <= age <= 55:
        out.append(("bad",
                    "Re-evaluación de necesidades vitales (hipoteca, jubilación, hijos, herencia)",
                    f"Edad {age:.0f}: el churn sube del 12,1% en 30–40 al 46,4% en 45–50 y 58,4% en 50–55. "
                    "El abandono aquí es cambio de necesidades no atendido, no falta de lealtad."))
    elif age > 55:
        out.append(("warn",
                    "Consolidar como cliente de patrimonio / sucesión",
                    f"Edad {age:.0f}: la tasa cae tras los 60 (24,8%); conviene asegurar traspaso generacional y renta."))
    if prod == 1:
        out.append(("good",
                    "Cross-sell de un segundo producto (máximo prioridad)",
                    "Con un producto el abandono es del 27,7%; con dos cae al 7,6%. Es el efecto de retención "
                    "más potente y barato de todo el dataset."))
    elif prod >= 3:
        out.append(("bad",
                    "Expediente de causa raíz sobre la cesta de productos y revisión de comisiones",
                    f"{prod:.0f} productos: el abandono observado es del 82,7% (3) y del 100% (4). "
                    "Suspender cualquier venta adicional hasta diagnosticar la fricción."))
    if c["Geography"] == "Germany":
        out.append(("bad",
                    "Prioridad task force Alemania: ajuste de comisiones y oferta de ahorro",
                    f"Abandono del {fmt_pct(0.324)} en clientes alemanes, 41,1% si están inactivos, "
                    "sobre el mayor pool de saldo (€300,4M). Es un problema de fit de producto."))
    if c["Gender"] == "Female" and risk != "Riesgo bajo":
        out.append(("warn",
                    "Incluir en el programa de prevención de género femenino",
                    "El género entra en el modelo como factor protector para los hombres (−0,26): las clientas "
                    "abandonan más y requieren tratamiento específico."))
    if bal > 100_000 and c["ValueTier"] == "Valor alto":
        out.append(("bad",
                    f"Asignar a playbook 'Proteger' (objetivo: retener {fmt_eur(bal * c['p_churn'])})",
                    "Alto valor con alto riesgo: 43,2% de churn real y 40,1% de la pérdida esperada total del banco."))
    if not out:
        out.append(("good",
                    "Mantener en low-touch digital y revisar en el siguiente ciclo de scoring",
                    f"P(abandono) de {fmt_pct(c['p_churn'],0)}: no justifica intervención costosa."))
    return out[:5]


def chart_gauge(p: float) -> go.Figure:
    color = PALETTE["bad"] if p >= 0.6 else PALETTE["warn"] if p >= 0.3 else PALETTE["good"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100,
        number={"suffix": " %", "font": dict(size=40, color=PALETTE["ink"])},
        title={"text": "P(abandono) · modelo", "font": dict(size=12, color=PALETTE["muted"])},
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor=PALETTE["muted"], dtick=25),
            bar=dict(color=color, thickness=0.32),
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
            steps=[dict(range=[0, 30], color="rgba(46,204,143,0.16)"),
                   dict(range=[30, 60], color="rgba(242,177,60,0.16)"),
                   dict(range=[60, 100], color="rgba(255,92,124,0.18)")],
            threshold=dict(line=dict(color=color, width=2), thickness=0.85, value=p * 100),
        ),
    ))
    fig.update_layout(
        height=230, margin=dict(l=18, r=18, t=42, b=6),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["ink"], family="Inter, system-ui, sans-serif"),
    )
    return fig


def chart_peer_compare(c: pd.Series, frame: pd.DataFrame) -> go.Figure:
    peer = frame[(frame["Age"] - c["Age"]).abs() <= 4]
    cats = ["Edad", "Saldo", "Score\ncrediticio", "Salario\nestimado", "Productos", "Antigüedad"]
    mine = [c["Age"], c["Balance"], c["CreditScore"], c["EstimatedSalary"], c["NumOfProducts"], c["Tenure"]]
    base = [peer["Age"].mean(), peer["Balance"].mean(), peer["CreditScore"].mean(),
            peer["EstimatedSalary"].mean(), peer["NumOfProducts"].mean(), peer["Tenure"].mean()]
    ratio = [100 * m / b if b else 0 for m, b in zip(mine, base)]
    fig = go.Figure(go.Scatterpolar(
        r=ratio, theta=cats, fill="toself",
        line=dict(color=PALETTE["accent"], width=2),
        fillcolor="rgba(79,140,255,0.22)",
        marker=dict(size=8, color=PALETTE["accent"]),
        hovertemplate="%{theta}: %{r:.0f} vs cohorte<extra></extra>",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(range=[0, max(220, min(max(ratio) * 1.25, 400))], tickangle=0,
                            gridcolor=PALETTE["grid"], tickfont=dict(size=9, color=PALETTE["muted"]),
                            showticklabels=False),
            angularaxis=dict(gridcolor=PALETTE["grid"], tickfont=dict(size=10.5, color=PALETTE["muted"])),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=290, margin=dict(l=78, r=78, t=26, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["ink"], family="Inter, system-ui, sans-serif"),
    )
    return fig


def chart_balance_context(c: pd.Series, frame: pd.DataFrame) -> go.Figure:
    same_geo = frame[frame["Geography"] == c["Geography"]]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=same_geo["Balance"], nbinsx=42,
                               marker=dict(color="rgba(152,164,189,0.30)", line_width=0),
                               name=f"Distribución {c['Geography']}",
                               hovertemplate="%{x}: %{y} clientes<extra></extra>"))
    fig.add_vline(x=c["Balance"], line_color=PALETTE["bad"], line_width=2.5,
                  annotation_text=f"este cliente · {fmt_eur(c['Balance'])}",
                  annotation_font_size=10.5, annotation_font_color=PALETTE["bad"],
                  annotation_position="top right")
    fig.update_layout(title=None)
    fig.update_xaxes(title="Saldo (€)", tickformat=",.0f")
    fig.update_yaxes(title="Clientes")
    return base_layout(fig, 260, legend=False)


def page_customer(frame: pd.DataFrame) -> None:
    st.markdown("## Perfil y búsqueda de cliente")
    st.markdown(
        "<div class='lede'>Localice una relación concreta, revise su categoría de riesgo y obtenga el "
        "playbook de retención recomendado para ese perfil.</div>",
        unsafe_allow_html=True,
    )

    top = frame.sort_values("expected_loss", ascending=False).iloc[0]
    mode = st.radio("Buscar por", ["Apellido (Surname)", "CustomerID"], horizontal=True, key="search_mode")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        if mode == "Apellido (Surname)":
            surnames = sorted(frame["Surname"].dropna().unique().tolist())
            default_surname = str(top["Surname"])
            choice = st.selectbox(
                "Apellido del cliente", surnames,
                index=surnames.index(default_surname) if default_surname in surnames else 0,
            )
            pool = frame[frame["Surname"] == choice].sort_values("expected_loss", ascending=False)
            pick = st.selectbox(
                "Cliente (los apellidos se repiten)", pool["CustomerId"].tolist(),
                format_func=lambda i: (f"{i} · {pool.loc[pool['CustomerId'] == i, 'Geography'].iloc[0]} · "
                                       f"edad {pool.loc[pool['CustomerId'] == i, 'Age'].iloc[0]:.0f}"),
            )
            c = frame[frame["CustomerId"] == pick].iloc[0]
        else:
            q = st.text_input("CustomerID", value=str(int(top["CustomerId"])))
            ids = frame["CustomerId"].astype(str)
            match = frame[ids == q.strip()]
            if match.empty:
                cand = frame[ids.str.contains(q.strip(), na=False, regex=False)].head(20)
                if cand.empty:
                    st.warning(f"Sin coincidencias para “{q}”. Pruebe con un CustomerID exacto, por ejemplo {int(top['CustomerId'])}.")
                    st.stop()
                pick = st.selectbox("Coincidencias parciales", cand["CustomerId"].tolist())
                c = frame[frame["CustomerId"] == pick].iloc[0]
            else:
                c = match.iloc[0]
    with col_b:
        st.markdown(
            f"<div style='height:34px'></div><div class='qcard' style='padding:12px 14px'>"
            f"<div class='qtitle'>Vista inicial</div><div class='qsub'>Cliente con mayor pérdida esperada: "
            f"<b>{fmt_eur(top['expected_loss'])}</b> ({top['Surname']}, {int(top['CustomerId'])}).</div></div>",
            unsafe_allow_html=True,
        )

    st.divider()

    risk_col = RISK_COLORS[c["RiskTier"]]
    exited = bool(c["Exited"])
    status = ("ya abandonó (Exited = 1) · foco en recuperación" if exited else "en cartera · accionable hoy")
    st.markdown(
        f"<div class='qcard'><div style='display:flex;align-items:baseline;gap:14px;flex-wrap:wrap'>"
        f"<div style='font-size:1.55rem;font-weight:700;flex:0 1 auto;min-width:0'>{c['Surname']}</div>"
        f"<div style='color:{PALETTE['muted']};font-size:0.9rem;flex:1 1 100%'>Cliente {int(c['CustomerId'])} · "
        f"{c['Geography']} · {c['Gender']} · {c['Age']:.0f} años</div>"
        f"{badge(c['RiskTier'], risk_col)} {badge(c['ValueTier'], PALETTE['accent'])} "
        f"{badge(c['Segment'], PALETTE['muted'])}"
        f"</div><div class='qsub' style='margin-top:6px'>{status}</div></div>",
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Saldo", fmt_eur(c["Balance"]), c["ValueTier"].replace("Valor ", ""), delta_color="off")
    m2.metric("Pérdida esperada", fmt_eur(c["expected_loss"]), f"{fmt_pct(c['p_churn'],0)} × {fmt_eur(c['Balance'])}", delta_color="off")
    m3.metric("Productos", f"{c['NumOfProducts']:.0f}",
              "riesgo alto" if c["NumOfProducts"] >= 3 else "óptimo" if c["NumOfProducts"] == 2 else "oportunidad cross-sell",
              delta_color="inverse" if c["NumOfProducts"] == 1 else "off")
    m4.metric("Estado operativo", "Activo" if c["IsActiveMember"] else "Inactivo",
              f"antigüedad {c['Tenure']:.0f} años", delta_color="off")

    left, mid, right = st.columns([0.9, 1.1, 1])
    with left:
        st.plotly_chart(chart_gauge(c["p_churn"]), width="stretch")
    with mid:
        st.markdown("##### Posición financiera frente a su cohorte de edad (±4 años)")
        st.plotly_chart(chart_peer_compare(c, frame), width="stretch")
        st.caption("Barras por encima del 100% indican que el cliente supera a su cohorte en esa dimensión.")
    with right:
        st.markdown("##### Recomendaciones personalizadas")
        items = ""
        for sev, action, reason in recommendations(c):
            col = PALETTE[sev]
            items += (
                f"<li><span class='flag' style='color:{col}'>▸</span> <b>{action}</b>"
                f"<div style='color:{PALETTE['muted']};font-size:0.82rem;margin-top:3px'>{reason}</div></li>"
            )
        st.markdown(f"<div class='qcard' style='padding:14px 16px'><ul class='actions'>{items}</ul></div>",
                    unsafe_allow_html=True)

    st.divider()
    st.markdown("### Ficha del cliente")
    a, b, ccol = st.columns([1, 1, 1])
    with a:
        st.markdown("**Ubicación y segmento**")
        st.table(pd.DataFrame({
            "Campo": ["Geografía", "Género", "Tipo de tarjeta", "Segmento", "Tier de valor", "Tier de riesgo"],
            "Valor": [c["Geography"], c["Gender"], c["Card Type"], c["Segment"], c["ValueTier"], c["RiskTier"]],
        }))
    with b:
        st.markdown("**Datos financieros**")
        st.table(pd.DataFrame({
            "Campo": ["Saldo (Balance)", "Salario estimado", "Score crediticio", "Puntos ganados", "Pérdida esperada"],
            "Valor": [fmt_eur(c["Balance"]), fmt_eur(c["EstimatedSalary"]), f"{c['CreditScore']:.0f}",
                      f"{c['Point Earned']:.0f}", fmt_eur(c["expected_loss"])],
        }))
    with ccol:
        st.markdown("**Relación y satisfacción**")
        st.table(pd.DataFrame({
            "Campo": ["Edad", "Antigüedad (tenure)", "Nº de productos", "Tarjeta de crédito",
                      "Miembro activo", "Queja registrada", "Satisfacción"],
            "Valor": [f"{c['Age']:.0f} años", f"{c['Tenure']:.0f} años", f"{c['NumOfProducts']:.0f}",
                      "Sí" if c["HasCrCard"] else "No", "Sí" if c["IsActiveMember"] else "No",
                      "Sí" if c["Complain"] else "No", f"{c['Satisfaction Score']:.0f} / 5"],
        }))

    st.markdown("##### Contexto de saldo del cliente dentro de su geografía")
    st.plotly_chart(chart_balance_context(c, frame), width="stretch")

    st.divider()
    st.markdown("### Clientes similares con mayor pérdida esperada")
    similar = frame[
        (frame["Geography"] == c["Geography"])
        & (frame["RiskTier"] == c["RiskTier"])
        & (frame["CustomerId"] != c["CustomerId"])
    ].sort_values("expected_loss", ascending=False)
    if similar.empty:
        similar = frame[frame["RiskTier"] == c["RiskTier"]].sort_values("expected_loss", ascending=False)
    cols = ["CustomerId", "Surname", "Age", "Balance", "NumOfProducts", "IsActiveMember", "Complain",
            "Satisfaction Score", "p_churn", "expected_loss", "Exited"]
    st.dataframe(
        similar[cols].head(12), width="stretch", hide_index=True,
        column_config={
            "CustomerId": st.column_config.NumberColumn("Cliente", format="%d"),
            "Balance": st.column_config.NumberColumn("Saldo", format="€ %d"),
            "expected_loss": st.column_config.NumberColumn("Pérdida esperada", format="€ %d"),
            "p_churn": st.column_config.ProgressColumn("P(abandono)", min_value=0, max_value=1, format="percent"),
            "IsActiveMember": st.column_config.CheckboxColumn("Activo"),
            "Complain": st.column_config.CheckboxColumn("Queja"),
            "Exited": st.column_config.CheckboxColumn("Abandonó"),
        },
    )
    st.caption(
        f"Perfil de riesgo {c['RiskTier'].lower()} en {c['Geography']}: {len(similar):,} clientes comparables. "
        "Esta lista es la base del triaje semanal del equipo de retención."
    )


# ----------------------------------------------------------------------------- nav

with st.sidebar:
    st.markdown("<div class='qtitle' style='padding:4px 0 8px'>Navegación</div>", unsafe_allow_html=True)
    page = st.radio("Sección", ["1 · Resumen ejecutivo", "2 · Perfil y búsqueda"], label_visibility="collapsed")
    st.divider()
    st.markdown(
        f"<div class='qtitle'>Cartera</div><div class='qsub' style='color:{PALETTE['muted']}'>"
        f"{len(df):,} clientes · {fmt_pct(df['Exited'].mean())} abandono · "
        f"modelo AUC test 0,856 (regresión logística sin <code>Complain</code>)</div>".replace(",", "."),
        unsafe_allow_html=True,
    )
    st.divider()
    filt_geo = st.multiselect("Filtrar geografía", sorted(df["Geography"].unique()), default=sorted(df["Geography"].unique()))
    filt_risk = st.multiselect("Filtrar tier de riesgo", ["Riesgo alto", "Riesgo medio", "Riesgo bajo"],
                               default=["Riesgo alto", "Riesgo medio", "Riesgo bajo"])
    only_active = st.checkbox("Solo clientes en cartera (Exited = 0)", value=False)

view = df[df["Geography"].isin(filt_geo) & df["RiskTier"].isin(filt_risk)]
if only_active:
    view = view[view["Exited"] == 0]
if view.empty:
    st.error("La combinación de filtros no deja clientes. Amplíe los filtros en la barra lateral.")
    st.stop()

if page.startswith("1"):
    page_overview(view)
else:
    page_customer(view)

st.markdown(
    f"<div style='color:{PALETTE['muted']};font-size:0.76rem;margin-top:26px;padding-top:12px;"
    f"border-top:1px solid {PALETTE['grid']}'>Fuentes: <code>Customer-Churn-Records-clean.csv</code> · "
    f"<code>Customer-Risk-Value-Segments.csv</code> · pipeline de <code>customer_analysis.ipynb</code>. "
    f"Las tarjetas de tipo de tarjeta, score crediticio y points earned se muestran como contexto, "
    f"no como predictores.</div>",
    unsafe_allow_html=True,
)

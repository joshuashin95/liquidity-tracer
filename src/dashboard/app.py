import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import DATA_PROCESSED
from src.prediction.predictor import load_model, predict_next

st.set_page_config(page_title="Liquidity Tracer", layout="wide")
st.title("Liquidity Tracer — 한국 증시 자금 흐름 추적기")


@st.cache_data
def load_signals():
    return pd.read_parquet(DATA_PROCESSED / "cluster_signals.parquet")


@st.cache_resource
def get_model():
    return load_model()


try:
    signals = load_signals()
    model = get_model()
except FileNotFoundError:
    st.error("데이터 없음. 먼저 collector → clustering → signal_engine → predictor 순으로 실행하세요.")
    st.stop()

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("필터")
top_n = st.sidebar.slider("상위 클러스터 수", 3, 20, 10)
date_range = st.sidebar.date_input(
    "기간",
    value=[pd.to_datetime(signals["date"].min()), pd.to_datetime(signals["date"].max())]
)

# ── Predictions ───────────────────────────────────────────────────────────────
st.subheader("차순위 급등 예상 클러스터 (Top N)")
predictions = predict_next(signals, model).head(top_n)
fig_pred = px.bar(
    predictions,
    x="cluster",
    y="surge_prob",
    color="surge_prob",
    color_continuous_scale="Reds",
    labels={"cluster": "클러스터", "surge_prob": "급등 확률"},
)
st.plotly_chart(fig_pred, use_container_width=True)

# ── Signal Heatmap ─────────────────────────────────────────────────────────
st.subheader("거래대금 서지 히트맵")
sig_filtered = signals.copy()
sig_filtered["date"] = pd.to_datetime(sig_filtered["date"])
if len(date_range) == 2:
    sig_filtered = sig_filtered[
        (sig_filtered["date"] >= pd.to_datetime(date_range[0])) &
        (sig_filtered["date"] <= pd.to_datetime(date_range[1]))
    ]

pivot = sig_filtered.pivot_table(index="date", columns="cluster", values="volume_surge", aggfunc="mean")
fig_heat = px.imshow(
    pivot.T,
    color_continuous_scale="YlOrRd",
    labels={"color": "서지 배율", "x": "날짜", "y": "클러스터"},
    aspect="auto",
)
st.plotly_chart(fig_heat, use_container_width=True)

# ── Combined Signal Timeline ──────────────────────────────────────────────
st.subheader("수급 집중 신호 타임라인")
combined = sig_filtered[sig_filtered["combined_signal"]].copy()
fig_timeline = px.scatter(
    combined,
    x="date",
    y="cluster",
    size="volume_surge",
    color="net_institutional",
    color_continuous_scale="Blues",
    labels={"date": "날짜", "cluster": "클러스터", "volume_surge": "서지 배율"},
)
st.plotly_chart(fig_timeline, use_container_width=True)

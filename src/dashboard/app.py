import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import DATA_PROCESSED, DATA_RAW
from src.prediction.predictor import load_model, predict_next

st.set_page_config(page_title="Liquidity Tracer", layout="wide")
st.title("Liquidity Tracer — 한국 증시 자금 흐름 추적기")


@st.cache_data
def load_signals():
    return pd.read_parquet(DATA_PROCESSED / "cluster_signals.parquet")


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def load_cluster_map():
    return pd.read_csv(DATA_PROCESSED / "cluster_map.csv", index_col=0).squeeze()


@st.cache_data
def load_ticker_names() -> dict:
    path = DATA_PROCESSED / "ticker_names.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str)
    return dict(zip(df["ticker"], df["name"]))


@st.cache_data
def get_ticker_latest_signals(tickers: tuple) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        path = DATA_RAW / f"{ticker}_ohlcv.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path).sort_index()
        if len(df) < 20:
            continue
        close = df["종가"]
        volume = df["거래량"]
        tv = close * volume
        tv_surge = (tv / tv.rolling(20).mean()).iloc[-1]
        vol_surge = (volume / volume.rolling(20).mean()).iloc[-1]
        ret_5d = close.pct_change(5).iloc[-1]
        rows.append({
            "ticker": ticker,
            "tv_surge": round(tv_surge, 2),
            "vol_surge": round(vol_surge, 2),
            "ret_5d": ret_5d,
            "close": int(close.iloc[-1]),
        })
    return pd.DataFrame(rows).sort_values("tv_surge", ascending=False)


try:
    signals = load_signals()
    model = get_model()
    cluster_map = load_cluster_map()
    ticker_names = load_ticker_names()
except FileNotFoundError:
    st.error("데이터 없음. 먼저 collector → clustering → signal_engine → predictor 순으로 실행하세요.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("필터")
top_n = st.sidebar.slider("상위 클러스터 수", 3, 20, 10)
top_stocks_n = st.sidebar.slider("추천 종목 수", 3, 20, 5)
date_range = st.sidebar.date_input(
    "기간",
    value=[pd.to_datetime(signals["date"].min()), pd.to_datetime(signals["date"].max())]
)

# ── Predictions ───────────────────────────────────────────────────────────────
st.subheader("차순위 급등 예상 클러스터")
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

# ── 클러스터 구성 종목 ────────────────────────────────────────────────────────
st.subheader("클러스터 구성 종목")
for _, row in predictions.iterrows():
    cid = row["cluster"]
    prob = row["surge_prob"]
    tickers_in_cluster = cluster_map[cluster_map == cid].index.tolist()
    names = [f"{ticker_names.get(t, t)} ({t})" for t in tickers_in_cluster]
    with st.expander(f"클러스터 {cid} — 급등확률 {prob:.1%} | {len(tickers_in_cluster)}개 종목"):
        st.write(", ".join(names))

# ── 추천 종목 ─────────────────────────────────────────────────────────────────
st.subheader("추천 종목")
st.caption("급등 확률 상위 클러스터에서 거래대금 서지가 가장 강한 종목")

top_clusters = predictions.head(2)["cluster"].tolist()
candidate_tickers = cluster_map[cluster_map.isin(top_clusters)].index.tolist()
ticker_signals = get_ticker_latest_signals(tuple(candidate_tickers))

if not ticker_signals.empty:
    top_stocks = ticker_signals.head(top_stocks_n).reset_index(drop=True)
    top_stocks["종목명"] = top_stocks["ticker"].map(ticker_names).fillna(top_stocks["ticker"])
    top_stocks["현재가"] = top_stocks["close"].apply(lambda x: f"{x:,}원")
    top_stocks["5일수익률"] = top_stocks["ret_5d"].apply(lambda x: f"{x*100:.1f}%")
    top_stocks.index = range(1, len(top_stocks) + 1)

    col1, col2 = st.columns([2, 3])
    with col1:
        st.dataframe(
            top_stocks[["종목명", "ticker", "현재가", "tv_surge", "vol_surge", "5일수익률"]].rename(columns={
                "ticker": "종목코드",
                "tv_surge": "거래대금서지",
                "vol_surge": "거래량서지",
            }),
            use_container_width=True,
        )
    with col2:
        top_stocks["label"] = top_stocks["종목명"].astype(str) + " (" + top_stocks["ticker"].astype(str) + ")"
        fig_stocks = px.bar(
            top_stocks,
            x="label",
            y="tv_surge",
            color="tv_surge",
            color_continuous_scale="Oranges",
            labels={"label": "종목", "tv_surge": "거래대금 서지 배율"},
        )
        fig_stocks.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig_stocks, use_container_width=True)
else:
    st.info("추천 종목 데이터를 불러올 수 없습니다.")

# ── Signal Heatmap ────────────────────────────────────────────────────────────
st.subheader("거래대금 서지 히트맵 (주간 평균)")
sig_filtered = signals.copy()
sig_filtered["date"] = pd.to_datetime(sig_filtered["date"])
if len(date_range) == 2:
    sig_filtered = sig_filtered[
        (sig_filtered["date"] >= pd.to_datetime(date_range[0])) &
        (sig_filtered["date"] <= pd.to_datetime(date_range[1]))
    ]

pivot = sig_filtered.pivot_table(index="date", columns="cluster", values="trading_value_surge", aggfunc="mean")
pivot.index = pd.to_datetime(pivot.index)
pivot_weekly = pivot.resample("W").mean()
pivot_weekly.index = pivot_weekly.index.strftime("%Y-%m-%d")
pivot_weekly.columns = [f"클러스터 {c}" for c in pivot_weekly.columns]

fig_heat = px.imshow(
    pivot_weekly.T,
    color_continuous_scale="YlOrRd",
    zmin=0.5,
    zmax=3.0,
    labels={"color": "서지 배율", "x": "주차", "y": "클러스터"},
    aspect="auto",
    text_auto=".1f",
)
fig_heat.update_layout(
    height=300,
    xaxis={"tickangle": -45, "tickfont": {"size": 10}},
    yaxis={"tickfont": {"size": 12}},
    coloraxis_colorbar={"title": "서지배율"},
)
st.plotly_chart(fig_heat, use_container_width=True)

# ── Combined Signal Timeline ──────────────────────────────────────────────────
st.subheader("수급 집중 신호 타임라인")
combined = sig_filtered[sig_filtered["combined_signal"]].copy()
if not combined.empty:
    fig_timeline = px.scatter(
        combined,
        x="date",
        y="cluster",
        size="trading_value_surge",
        color="net_institutional",
        color_continuous_scale="Blues",
        labels={"date": "날짜", "cluster": "클러스터", "trading_value_surge": "거래대금 서지"},
    )
    st.plotly_chart(fig_timeline, use_container_width=True)
else:
    st.info("해당 기간에 수급 집중 신호 없음")

"""④ 대시보드 — 이상탐지 결과 페이지. (소유: mfg-reporter)"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── sys.path 부트스트랩: 레포 루트(config)·대시보드 폴더(_lib) 보장 (로컬·클라우드 공통) ──
import sys as _sys
from pathlib import Path as _Path
for _anc in _Path(__file__).resolve().parents:
    if (_anc / "_lib.py").exists() and str(_anc) not in _sys.path:
        _sys.path.insert(0, str(_anc))
    if (_anc / "config" / "settings.py").exists() and str(_anc) not in _sys.path:
        _sys.path.insert(0, str(_anc))

from config.settings import MODELS_DIR
from _lib import (
    STATUS_COLORS,
    dash_header,
    inject_css,
    kpi_tile,
    render_footer,
    render_sidebar,
    style_fig,
)

st.set_page_config(page_title="이상탐지", page_icon="🚨", layout="wide")
inject_css()
render_sidebar()
dash_header(
    "③ 딥러닝 이상탐지 결과",
    "정상만 학습한 시퀀스 AE의 재구성오차 → 마할라노비스 거리로 이상 판정",
)

scores_path = MODELS_DIR / "scores.parquet"
if not scores_path.exists():
    st.warning("탐지 결과가 없습니다. `python -m src.models.detect` 를 먼저 실행하세요.")
    st.stop()

df = pd.read_parquet(scores_path)
n_anom = int(df["is_anomaly"].sum())
c1, c2, c3 = st.columns(3)
with c1:
    kpi_tile("총 샘플", f"{len(df):,}", accent="#2C7BE5")
with c2:
    kpi_tile("이상 탐지", f"{n_anom:,}", unit="건", accent=STATUS_COLORS["이상"])
with c3:
    kpi_tile("이상 비율", f"{df['is_anomaly'].mean() * 100:.1f}", unit="%",
             accent="#E67E22")

# ── 이상 점수 타임라인 (임계선 + 이상 마커) ──────────────────────────
st.subheader("이상 점수 타임라인")
ts = df.set_index("timestamp")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=ts.index, y=ts["anomaly_score"], name="이상 점수",
    line=dict(color="#2C7BE5", width=1.6),
))
fig.add_trace(go.Scatter(
    x=ts.index, y=ts["threshold"], name="임계값",
    line=dict(color="#E74C3C", width=1.4, dash="dash"),
))
anom = ts[ts["is_anomaly"] == 1]
if not anom.empty:
    fig.add_trace(go.Scatter(
        x=anom.index, y=anom["anomaly_score"], name="이상", mode="markers",
        marker=dict(color="#E74C3C", size=8, symbol="x"),
    ))
fig.update_layout(xaxis_title="시각", yaxis_title="anomaly score")
st.plotly_chart(style_fig(fig, height=360), use_container_width=True)

st.subheader("이상 구간 상세")
st.dataframe(
    df[df["is_anomaly"] == 1][["timestamp", "fault_id", "anomaly_score"]],
    use_container_width=True,
)

render_footer()

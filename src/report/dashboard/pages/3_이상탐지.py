"""④ 대시보드 — ③ 딥러닝 이상탐지 결과 페이지. (소유: mfg-reporter)"""

import numpy as np
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
    fault_name,
    inject_css,
    kpi_tile,
    render_footer,
    render_insight,
    render_sidebar,
    style_fig,
)

st.set_page_config(page_title="이상탐지", page_icon="🚨", layout="wide")
inject_css()
render_sidebar()
dash_header(
    "③ 딥러닝 이상탐지 결과",
    "정상만 학습한 시퀀스 AE의 재구성오차 → 마할라노비스 거리로 이상 판정 (기본 Transformer-AE)",
)

scores_path = MODELS_DIR / "scores.parquet"
if not scores_path.exists():
    st.warning("탐지 결과가 없습니다. `python -m src.models.detect` 를 먼저 실행하세요.")
    st.stop()

df = pd.read_parquet(scores_path).reset_index(drop=True)
y_true = (df["fault_id"] != 0).values
y_pred = (df["is_anomaly"] == 1).values
tp = int((y_pred & y_true).sum())
fp = int((y_pred & ~y_true).sum())
fn = int((~y_pred & y_true).sum())
recall = tp / (tp + fn) if (tp + fn) else 0.0
precision = tp / (tp + fp) if (tp + fp) else 0.0

# ── KPI ──────────────────────────────────────────────────────────────
k = st.columns(4)
with k[0]:
    kpi_tile("총 평가 샘플", f"{len(df):,}", accent="#2C7BE5")
with k[1]:
    kpi_tile("이상 탐지", f"{int(y_pred.sum()):,}", unit="건", accent=STATUS_COLORS["이상"])
with k[2]:
    kpi_tile("정탐율(Recall)", f"{recall*100:.0f}", unit="%",
             accent=STATUS_COLORS["정상"], sub=f"결함 {tp+fn}건 중 {tp}건 포착")
with k[3]:
    kpi_tile("오탐(FP)", f"{fp:,}", unit="건", accent=STATUS_COLORS["주의"],
             delta_good=(fp == 0), sub=f"정밀도 {precision*100:.0f}%")

render_insight(
    f"실 TEP {len(df):,} 샘플 중 결함 **{tp+fn}건**을 정답으로 두고, 모델이 **{tp}건 포착(정탐율 {recall*100:.0f}%)** · "
    f"오탐 **{fp}건**. 아래 타임라인의 빨강 음영(정답 결함 구간)과 마커(모델 탐지)를 비교해 보세요."
)

# ── 이상 점수 타임라인 (정답 결함 구간 음영 + 임계선 + 탐지 마커) ─────
st.subheader("이상 점수 타임라인 — 정답 결함 구간 vs 모델 탐지")
fig = go.Figure()
fig.add_trace(go.Scatter(
    y=df["anomaly_score"], mode="lines", name="이상 점수",
    line=dict(color="#2C7BE5", width=1.3),
))
fig.add_trace(go.Scatter(
    y=df["threshold"], mode="lines", name="임계값",
    line=dict(color="#E74C3C", width=1.2, dash="dash"),
))
# 정답 결함 구간(연속 블록) 음영
mask = y_true
i = 0
while i < len(mask):
    if mask[i]:
        j = i
        while j < len(mask) and mask[j] and df["fault_id"].iloc[j] == df["fault_id"].iloc[i]:
            j += 1
        fig.add_vrect(x0=i, x1=j - 1, fillcolor="rgba(231,76,60,0.10)", line_width=0,
                      annotation_text=f"IDV{int(df['fault_id'].iloc[i])}",
                      annotation_position="top left", annotation_font_size=9)
        i = j
    else:
        i += 1
anom_idx = np.where(y_pred)[0]
fig.add_trace(go.Scatter(
    x=anom_idx, y=df["anomaly_score"].values[anom_idx], mode="markers", name="모델 탐지",
    marker=dict(color="#E74C3C", size=6, symbol="x"),
))
fig.update_layout(xaxis_title="샘플 인덱스(시간순)", yaxis_title="anomaly score")
st.plotly_chart(style_fig(fig, height=360), use_container_width=True)

# ── 점수 분포(분리도) + IDV별 탐지율 ─────────────────────────────────
c1, c2 = st.columns(2)
with c1:
    st.subheader("점수 분포 — 정상 vs 결함")
    fig_d = go.Figure()
    fig_d.add_trace(go.Histogram(x=df.loc[~y_true, "anomaly_score"], name="정상",
                                 marker_color=STATUS_COLORS["정상"], opacity=0.6, nbinsx=50))
    fig_d.add_trace(go.Histogram(x=df.loc[y_true, "anomaly_score"], name="결함",
                                 marker_color=STATUS_COLORS["이상"], opacity=0.6, nbinsx=50))
    thr = float(df["threshold"].median())
    fig_d.add_vline(x=thr, line_dash="dash", line_color="#E74C3C",
                    annotation_text="임계값", annotation_position="top")
    fig_d.update_layout(barmode="overlay", xaxis_title="anomaly score", yaxis_title="빈도")
    st.plotly_chart(style_fig(fig_d, height=320), use_container_width=True)
    st.caption("두 분포가 잘 갈릴수록 탐지가 쉬움. 임계값 오른쪽이 이상으로 판정.")
with c2:
    st.subheader("결함 IDV별 탐지율")
    pf = (df[df["fault_id"] != 0].groupby("fault_id")["is_anomaly"].mean().reset_index())
    pf["label"] = pf["fault_id"].map(lambda x: f"IDV{int(x)}")
    pf = pf.sort_values("is_anomaly")
    colors = ["#E74C3C" if v < 0.2 else "#E67E22" if v < 0.7 else "#27AE60"
              for v in pf["is_anomaly"]]
    fig_pf = go.Figure(go.Bar(
        x=pf["is_anomaly"], y=pf["label"], orientation="h", marker_color=colors,
        text=[f"{v*100:.0f}%" for v in pf["is_anomaly"]], textposition="outside",
        hovertemplate="%{y}<br>탐지율 %{x:.0%}<extra></extra>",
    ))
    fig_pf.update_layout(xaxis_title="탐지율", xaxis_range=[0, 1.15], yaxis_title="")
    st.plotly_chart(style_fig(fig_pf, height=320), use_container_width=True)
    st.caption("초록=잘 탐지, 주황=부분, 빨강=미탐(IDV 3·18은 실 TEP 난탐지 결함).")

# ── 이상 구간 상세 ───────────────────────────────────────────────────
st.subheader("탐지된 이상 구간 상세")
detail = df[df["is_anomaly"] == 1][["timestamp", "fault_id", "anomaly_score"]].copy()
detail["결함"] = detail["fault_id"].map(fault_name)
st.dataframe(detail[["timestamp", "결함", "anomaly_score"]], use_container_width=True, height=280)

render_footer()

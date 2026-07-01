"""④ 대시보드 — ④ 모델 비교 페이지. (소유: mfg-reporter)

mfg-model 산출 비교표/커브/IDV별 탐지율을 read-only로 임베드.
기본 모델은 VAE(F1·ROC-AUC·PR-AUC 최고, 오탐 최저).
"""

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
    CHART_COLORS,
    dash_header,
    inject_css,
    kpi_tile,
    render_footer,
    render_sidebar,
    style_fig,
)

# 기본 모델 및 표시 라벨
DEFAULT_MODEL = "transformer_ae"
MODEL_LABELS = {"lstm_ae": "LSTM-AE", "vae": "VAE", "transformer_ae": "Transformer-AE ★기본"}

st.set_page_config(page_title="모델비교", page_icon="🧪", layout="wide")
inject_css()
render_sidebar()
dash_header(
    "④ 모델 비교 (딥러닝 이상탐지)",
    "LSTM-AE · VAE · Transformer-AE 3종을 동일 실 TEP 데이터(1,940행)에서 비교 — mfg-model 산출",
)

comp_path = MODELS_DIR / "comparison.parquet"
if not comp_path.exists():
    st.warning("모델 비교 산출이 없습니다. `python -m src.models.detect` 를 먼저 실행하세요.")
    st.stop()


@st.cache_data(show_spinner=False)
def load_comparison() -> pd.DataFrame:
    """모델별 종합 성능표(캐싱)."""
    return pd.read_parquet(MODELS_DIR / "comparison.parquet")


@st.cache_data(show_spinner=False)
def load_per_fault() -> pd.DataFrame:
    """결함 IDV별 탐지율(캐싱). 없으면 빈 DataFrame."""
    path = MODELS_DIR / "comparison_per_fault.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


comp = load_comparison().copy()
comp["모델"] = comp["model"].map(MODEL_LABELS).fillna(comp["model"])

# ── 기본 모델 핵심 지표 ───────────────────────────────────
best = comp[comp["model"] == DEFAULT_MODEL]
if not best.empty:
    r = best.iloc[0]
    st.subheader("기본 모델: Transformer-AE 핵심 지표")
    cols = st.columns(5)
    _metrics = [
        ("Precision", r["precision"], "#2C7BE5"),
        ("Recall", r["recall"], "#17A2B8"),
        ("F1", r["f1"], "#E67E22"),
        ("ROC-AUC", r["roc_auc"], "#27AE60"),
        ("PR-AUC", r["pr_auc"], "#6F42C1"),
    ]
    for col, (lbl, val, acc) in zip(cols, _metrics):
        with col:
            kpi_tile(lbl, f"{val:.3f}", accent=acc)
    st.caption(
        f"오탐(FP) {int(r['fp'])} · 미탐(FN) {int(r['fn'])} — "
        "Transformer-AE가 F1·ROC-AUC 최고. 정밀도 우선 운용이라면 VAE(오탐 0·정밀도 1.00)가 유리 "
        "(실 TEP 1,940행 기준)."
    )

# ── 종합 비교표 ───────────────────────────────────────────
st.subheader("모델별 종합 성능")
metric_cols = ["모델", "precision", "recall", "f1", "roc_auc", "pr_auc", "tp", "fp", "fn", "tn"]
metric_cols = [c for c in metric_cols if c in comp.columns]
st.dataframe(
    comp[metric_cols].style.format(
        {c: "{:.3f}" for c in ["precision", "recall", "f1", "roc_auc", "pr_auc"] if c in comp.columns}
    ),
    use_container_width=True,
)

# ── 성능 막대 비교 ────────────────────────────────────────
st.subheader("성능 지표 막대 비교")
bar_metrics = [c for c in ["f1", "roc_auc", "pr_auc"] if c in comp.columns]
_label_map = {"f1": "F1", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"}
fig_bar = go.Figure()
for i, m in enumerate(bar_metrics):
    fig_bar.add_trace(go.Bar(
        x=comp["모델"], y=comp[m], name=_label_map.get(m, m),
        marker_color=CHART_COLORS[i],
        text=[f"{v:.3f}" for v in comp[m]], textposition="outside",
    ))
fig_bar.update_layout(barmode="group", yaxis_title="점수", yaxis_range=[0, 1])
st.plotly_chart(style_fig(fig_bar, height=340), use_container_width=True)

# ── ROC / PR 커브 ─────────────────────────────────────────
st.subheader("ROC · PR 커브")
curves_png = MODELS_DIR / "roc_pr_curves.png"
if curves_png.exists():
    st.image(str(curves_png), caption="모델별 ROC / PR 커브 — mfg-model 산출", use_container_width=True)
else:
    st.info("`roc_pr_curves.png` 산출물이 아직 없습니다.")

# ── 결함 IDV별 탐지율 ─────────────────────────────────────
st.subheader("결함 IDV별 탐지율")
per_fault = load_per_fault()
if per_fault.empty:
    st.info("`comparison_per_fault.parquet` 산출물이 아직 없습니다.")
else:
    pf = per_fault[per_fault["fault_id"] != 0].copy()
    pivot = pf.pivot_table(index="fault_id", columns="model", values="recall")
    pivot = pivot.rename(columns=MODEL_LABELS)
    st.dataframe(pivot.style.format("{:.2f}"), use_container_width=True)
    st.caption(
        "IDV 3·18은 재구성오차가 정상과 사실상 구별되지 않아 AE 계열 원리상 미탐(0.00) — "
        "실제 TEP에서도 잘 알려진 난탐지 결함. IDV 12는 부분 탐지. 한계를 그대로 표기합니다."
    )

# ── MODEL_CARD 핵심 요약 ──────────────────────────────────
st.subheader("모델 카드 핵심 요약")
st.markdown(
    """
    - **데이터**: 실제 TEP 벤치마크(Downs & Vogel, Braatz 배포판) 1,940행 — 정상 1,460 / 결함 480(IDV 10종).
    - **준지도 이상탐지**: 정상 운전만으로 시퀀스 AE 학습 → 재구성오차의 **마할라노비스 거리**로 판정.
    - **임계값**: 정상 점수 분위 `q=0.995` × 여유배수 `1.10`, 행 점수 rolling median 평활(과탐 억제).
    - **기본 모델 Transformer-AE**: F1 0.85·ROC-AUC 0.92로 최고. VAE는 오탐 0(정밀도 1.00)로 정밀도 우선에 유리.
    - **한계**: IDV 3·18 미탐(재구성오차가 정상과 구별 불가) — 실 TEP에서도 잘 알려진 난탐지 결함.

    상세: `src/models/MODEL_CARD.md`
    """
)

render_footer()

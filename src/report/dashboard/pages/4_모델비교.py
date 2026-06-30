"""④ 대시보드 — ④ 모델 비교 페이지. (소유: mfg-reporter)

mfg-model 산출 비교표/커브/IDV별 탐지율을 read-only로 임베드.
기본 모델은 VAE(F1·ROC-AUC·PR-AUC 최고, 오탐 최저).
"""

import pandas as pd
import streamlit as st

from config.settings import MODELS_DIR

# 기본 모델 및 표시 라벨
DEFAULT_MODEL = "vae"
MODEL_LABELS = {"lstm_ae": "LSTM-AE", "vae": "VAE ★기본", "transformer_ae": "Transformer-AE"}

st.title("④ 모델 비교 (딥러닝 이상탐지)")
st.caption("LSTM-AE · VAE · Transformer-AE 3종을 동일 데이터(885행)에서 비교 — mfg-model 산출")

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
    st.subheader("기본 모델: VAE 핵심 지표")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Precision", f"{r['precision']:.3f}")
    c2.metric("Recall", f"{r['recall']:.3f}")
    c3.metric("F1", f"{r['f1']:.3f}")
    c4.metric("ROC-AUC", f"{r['roc_auc']:.3f}")
    c5.metric("PR-AUC", f"{r['pr_auc']:.3f}")
    st.caption(
        f"오탐(FP) {int(r['fp'])} · 미탐(FN) {int(r['fn'])} — "
        "정밀도 우선 운용 시 VAE가 오탐을 가장 적게 냅니다(조건: 현재 885행 기준)."
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
st.bar_chart(comp.set_index("모델")[bar_metrics])

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
        "IDV 1·4·8은 재구성오차가 정상과 사실상 구별되지 않아 AE 계열 원리상 미탐(0.00). "
        "IDV 12는 부분 탐지. 한계를 그대로 표기합니다."
    )

# ── MODEL_CARD 핵심 요약 ──────────────────────────────────
st.subheader("모델 카드 핵심 요약")
st.markdown(
    """
    - **준지도 이상탐지**: 정상 운전만으로 시퀀스 AE 학습 → 재구성오차의 **마할라노비스 거리**로 판정.
    - **임계값**: 정상 점수 분위 `q=0.995` × 여유배수 `1.10`, 행 점수 rolling median 평활(과탐 억제).
    - **기본 모델 VAE**: F1·ROC-AUC·PR-AUC 모두 최고, 오탐 6/671(0.9%)로 최저.
    - **한계**: IDV 1·4·8 미탐(재구성오차 패턴이 정상과 구별 불가). 데이터 규모(885행)에 따른 조건부 결과.

    상세: `src/models/MODEL_CARD.md`
    """
)

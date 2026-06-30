"""④ 대시보드 — ② EDA·통계 페이지. (소유: mfg-reporter)

mfg-eda 산출 PNG/통계를 read-only로 임베드 + 정상 대비 결함 유의변수 정리.
"""

import json

import pandas as pd
import streamlit as st

from config.settings import EDA_DIR

st.title("② EDA · 통계 분석")
st.caption("정상/결함 분포 비교와 통계 검정으로 변수별 분별력을 점검 — mfg-eda 산출 임베드")

if not EDA_DIR.exists():
    st.warning("EDA 산출 폴더가 없습니다. `python -m src.pipeline.eda` 를 먼저 실행하세요.")
    st.stop()


@st.cache_data(show_spinner=False)
def load_summary() -> pd.DataFrame:
    """변수별 요약통계(캐싱). 없으면 빈 DataFrame."""
    path = EDA_DIR / "summary_stats.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_fault_tests() -> pd.DataFrame:
    """정상 vs 결함 t검정/KS검정 결과(캐싱) → 변수별 행 DataFrame."""
    path = EDA_DIR / "fault_tests.json"
    if not path.exists():
        return pd.DataFrame()
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = [{"variable": var, **stats} for var, stats in raw.items()]
    return pd.DataFrame(rows)


# ── 상관 히트맵 ───────────────────────────────────────────
st.subheader("변수 상관 히트맵")
corr_png = EDA_DIR / "corr_heatmap.png"
if corr_png.exists():
    st.image(str(corr_png), caption="52개 공정변수 상관구조", use_container_width=True)
else:
    st.info("`corr_heatmap.png` 산출물이 아직 없습니다.")

# ── 정상 vs 결함 분포 / 시계열 ────────────────────────────
col1, col2 = st.columns(2)
with col1:
    st.subheader("정상 vs 결함 분포")
    dist_png = EDA_DIR / "dist_normal_vs_fault.png"
    if dist_png.exists():
        st.image(str(dist_png), caption="주요 변수 분포 비교", use_container_width=True)
    else:
        st.info("`dist_normal_vs_fault.png` 산출물이 아직 없습니다.")
with col2:
    st.subheader("주요 변수 시계열")
    ts_png = EDA_DIR / "timeseries_key_vars.png"
    if ts_png.exists():
        st.image(str(ts_png), caption="결함 구간 표시 시계열", use_container_width=True)
    else:
        st.info("`timeseries_key_vars.png` 산출물이 아직 없습니다.")

# ── 통계 검정: 상위 유의 변수 ─────────────────────────────
st.subheader("정상 대비 결함 유의 변수 (KS 검정 상위)")
tests = load_fault_tests()
if tests.empty:
    st.info("`fault_tests.json` 산출물이 아직 없습니다.")
else:
    sort_key = "ks_stat" if "ks_stat" in tests.columns else tests.columns[1]
    top = tests.sort_values(sort_key, ascending=False).head(15).reset_index(drop=True)
    st.caption(
        "KS 통계량이 클수록 정상·결함 분포 차이가 뚜렷(분별력 높음). "
        "p값이 작을수록 차이가 통계적으로 유의."
    )
    st.dataframe(top, use_container_width=True)

# ── 변수별 요약통계 ───────────────────────────────────────
st.subheader("변수별 요약통계")
summary = load_summary()
if summary.empty:
    st.info("`summary_stats.parquet` 산출물이 아직 없습니다.")
else:
    st.dataframe(summary, use_container_width=True)

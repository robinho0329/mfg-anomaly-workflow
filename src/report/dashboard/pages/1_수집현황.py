"""④ 대시보드 — ① 수집현황 페이지. (소유: mfg-reporter)

stream.db 적재 산출물을 read-only로 임베드한다(무거운 재계산 없음).
"""

import sqlite3

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.settings import COLLECT_TABLE, DB_PATH, EDA_DIR
from _lib import (
    STATUS_COLORS,
    dash_header,
    inject_css,
    kpi_tile,
    render_footer,
    render_sidebar,
    style_fig,
)

st.set_page_config(page_title="수집현황", page_icon="📥", layout="wide")
inject_css()
render_sidebar()
dash_header(
    "① 수집 현황",
    "TEP 다변량 스트림(52변수)을 SQLite에 누적 — 정상 위주 백필 구성",
)

if not DB_PATH.exists():
    st.warning("수집 DB가 없습니다. `python -m src.collect.scheduler` 로 먼저 적재하세요.")
    st.stop()


@st.cache_data(show_spinner=False)
def load_stream() -> pd.DataFrame:
    """stream.db tep_stream 테이블 적재(캐싱). 테이블 없으면 빈 DataFrame."""
    with sqlite3.connect(DB_PATH) as conn:
        try:
            df = pd.read_sql(f"SELECT * FROM {COLLECT_TABLE}", conn)
        except (pd.errors.DatabaseError, sqlite3.OperationalError):
            return pd.DataFrame()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


df = load_stream()
if df.empty:
    st.warning(f"`{COLLECT_TABLE}` 테이블이 비어 있습니다.")
    st.stop()

# ── 핵심 지표 ──────────────────────────────────────────────
n_total = len(df)
n_normal = int((df["fault_id"] == 0).sum())
n_fault = n_total - n_normal
ratio = n_normal / n_fault if n_fault else float("nan")

c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi_tile("총 적재 행", f"{n_total:,}", accent="#2C7BE5")
with c2:
    kpi_tile("정상 (fault_id=0)", f"{n_normal:,}", accent=STATUS_COLORS["정상"])
with c3:
    kpi_tile("결함 (IDV)", f"{n_fault:,}", accent=STATUS_COLORS["이상"])
with c4:
    kpi_tile("정상 : 결함", f"{ratio:.1f} : 1" if n_fault else "—", accent="#E67E22")

ts_min, ts_max = df["timestamp"].min(), df["timestamp"].max()
span_h = (ts_max - ts_min).total_seconds() / 3600 if pd.notna(ts_min) else 0
st.caption(
    f"시각 범위: {ts_min:%Y-%m-%d %H:%M} ~ {ts_max:%Y-%m-%d %H:%M} "
    f"(약 {span_h:.1f}시간 · 3분 간격 가정)"
)

# ── 적재 타임라인(시간당 적재량) ──────────────────────────
st.subheader("적재 타임라인")
hourly = (
    df.set_index("timestamp")
    .assign(_n=1)
    .resample("1h")["_n"]
    .sum()
    .rename("적재 행 수")
)
fig_tl = go.Figure(go.Bar(x=hourly.index, y=hourly.values, marker_color="#2C7BE5"))
fig_tl.update_layout(xaxis_title="시각", yaxis_title="시간당 적재 행")
st.plotly_chart(style_fig(fig_tl, height=280), use_container_width=True)

# ── 정상 vs 결함 비율 ─────────────────────────────────────
st.subheader("정상 vs 결함 구성")
col_left, col_right = st.columns([1, 1])
with col_left:
    fig_mix = go.Figure(go.Bar(
        x=["정상", "결함"], y=[n_normal, n_fault],
        marker_color=[STATUS_COLORS["정상"], STATUS_COLORS["이상"]],
        text=[f"{n_normal:,}", f"{n_fault:,}"], textposition="outside",
    ))
    fig_mix.update_layout(yaxis_title="행 수")
    st.plotly_chart(style_fig(fig_mix, height=320), use_container_width=True)
with col_right:
    fault_png = EDA_DIR / "fault_counts.png"
    if fault_png.exists():
        st.image(str(fault_png), caption="결함 모드(IDV)별 분포 — mfg-eda 산출", use_container_width=True)
    else:
        st.info("`fault_counts.png` 산출물이 아직 없습니다.")

# ── 최근 수집 테이블 ──────────────────────────────────────
st.subheader("최근 수집 (최신 20행)")
preview_cols = ["timestamp", "fault_id", "xmeas_01", "xmeas_02", "xmv_01", "xmv_02"]
preview_cols = [c for c in preview_cols if c in df.columns]
st.dataframe(
    df.sort_values("timestamp", ascending=False)[preview_cols].head(20),
    use_container_width=True,
)

render_footer()

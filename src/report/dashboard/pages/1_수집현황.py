"""④ 대시보드 — ① 수집현황 페이지. (소유: mfg-reporter)

stream.db 적재 산출물을 read-only로 임베드한다(무거운 재계산 없음).
"""

import sqlite3

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

from config.settings import COLLECT_TABLE, DB_PATH
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

st.set_page_config(page_title="수집현황", page_icon="📥", layout="wide")
inject_css()
render_sidebar()
dash_header(
    "① 수집 현황",
    "실제 TEP 벤치마크(52변수)를 SQLite에 적재 — 정상 운전 + 결함 IDV 10종",
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

render_insight(
    f"실제 TEP 벤치마크를 **정상 위주(정상:결함 {ratio:.1f}:1)**로 구성 — 준지도 이상탐지는 "
    f"정상 데이터로만 학습하므로 정상 표본이 많을수록 유리합니다. 결함은 IDV "
    f"{df.loc[df.fault_id!=0,'fault_id'].nunique()}종을 골고루 포함해 탐지력을 다각도로 평가합니다."
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

# ── 정상 vs 결함 구성 + 결함 IDV별 분포 ───────────────────
st.subheader("데이터 구성")
col_left, col_right = st.columns([1, 1.6])
with col_left:
    fig_mix = go.Figure(go.Pie(
        labels=["정상", "결함"], values=[n_normal, n_fault], hole=0.55,
        marker_colors=[STATUS_COLORS["정상"], STATUS_COLORS["이상"]],
        textinfo="label+percent",
    ))
    fig_mix.update_layout(showlegend=False, height=300, margin=dict(l=10, r=10, t=10, b=10),
                          annotations=[dict(text=f"{n_total:,}<br>행", showarrow=False, font_size=15)])
    st.plotly_chart(fig_mix, use_container_width=True)
with col_right:
    fc = (df[df["fault_id"] != 0].groupby("fault_id").size().reset_index(name="n"))
    fc["label"] = fc["fault_id"].map(lambda x: fault_name(x))
    fc = fc.sort_values("n")
    fig_fc = go.Figure(go.Bar(
        x=fc["n"], y=fc["label"], orientation="h", marker_color="#E67E22",
        text=fc["n"], textposition="outside",
        hovertemplate="%{y}<br>%{x}행<extra></extra>",
    ))
    fig_fc.update_layout(xaxis_title="행 수", yaxis_title="")
    st.plotly_chart(style_fig(fig_fc, height=300), use_container_width=True)
st.caption("결함 IDV별 균등 표본(각 48행). IDV 3·18은 실 TEP에서 난탐지로 알려진 결함.")

# ── 다변량 센서 미리보기 (결함 구간 음영) ──────────────────
st.subheader("다변량 센서 미리보기")
_key = [c for c in ["xmeas_07", "xmeas_09", "xmv_03"] if c in df.columns][:3]
_sel = st.multiselect("표시 변수 (최대 3개 권장)", [c for c in df.columns if c.startswith(("xmeas", "xmv"))],
                      default=_key)
if _sel:
    sdf = df.reset_index(drop=True)
    fig_s = go.Figure()
    for i, v in enumerate(_sel):
        fig_s.add_trace(go.Scatter(y=sdf[v], mode="lines", name=v,
                                   line=dict(width=1.1)))
    mask = (sdf["fault_id"] != 0).values
    j = 0
    while j < len(mask):
        if mask[j]:
            k = j
            while k < len(mask) and mask[k] and sdf["fault_id"].iloc[k] == sdf["fault_id"].iloc[j]:
                k += 1
            fig_s.add_vrect(x0=j, x1=k - 1, fillcolor="rgba(231,76,60,0.08)", line_width=0)
            j = k
        else:
            j += 1
    fig_s.update_layout(xaxis_title="샘플 인덱스(시간순)", yaxis_title="센서 값")
    st.plotly_chart(style_fig(fig_s, height=300), use_container_width=True)
    st.caption("빨강 음영 = 결함 구간. 여러 센서가 결함에 어떻게 동시 반응하는지(다변량성) 확인.")

# ── 최근 수집 테이블 ──────────────────────────────────────
st.subheader("최근 수집 (최신 20행)")
preview_cols = ["timestamp", "fault_id", "xmeas_01", "xmeas_02", "xmv_01", "xmv_02"]
preview_cols = [c for c in preview_cols if c in df.columns]
st.dataframe(
    df.sort_values("timestamp", ascending=False)[preview_cols].head(20),
    use_container_width=True,
)

render_footer()

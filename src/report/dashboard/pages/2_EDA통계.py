"""④ 대시보드 — ② EDA·통계 페이지. (소유: mfg-reporter)

실 TEP 스트림(stream.db)을 직접 로드해 인터랙티브 EDA를 제공하고,
mfg-eda 정적 산출(PNG/통계)은 보조로 임베드한다.
"""

import json

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

from config.settings import EDA_DIR, PROCESS_COLS
from _lib import (
    STATUS_COLORS,
    dash_header,
    fault_name,
    inject_css,
    load_stream,
    render_footer,
    render_insight,
    render_sidebar,
    style_fig,
)

st.set_page_config(page_title="EDA·통계", page_icon="📊", layout="wide")
inject_css()
render_sidebar()
dash_header(
    "② EDA · 통계 분석",
    "실 TEP 52변수의 정상/결함 분포·상관·분별력을 인터랙티브로 점검",
)

df = load_stream()
if df.empty:
    st.warning("수집 스트림(stream.db)이 없습니다. `python -m scripts.build_real_tep` → 수집 후 표시됩니다.")
    st.stop()

VARS = [c for c in PROCESS_COLS if c in df.columns]
df["구분"] = np.where(df["fault_id"] == 0, "정상", "결함")


@st.cache_data(show_spinner=False)
def load_fault_tests() -> pd.DataFrame:
    """정상 vs 결함 KS/t검정 결과 → 변수별 DataFrame."""
    path = EDA_DIR / "fault_tests.json"
    if not path.exists():
        return pd.DataFrame()
    raw = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame([{"variable": v, **s} for v, s in raw.items()])


tests = load_fault_tests()
# 분별력 순위(KS 통계량) — 셀렉트박스 기본값·상관 대상에 활용
if not tests.empty and "ks_stat" in tests.columns:
    ranked = tests.sort_values("ks_stat", ascending=False)["variable"].tolist()
    ranked = [v for v in ranked if v in VARS]
else:
    ranked = sorted(VARS, key=lambda c: df.loc[df.fault_id != 0, c].std(), reverse=True)
top_var = ranked[0] if ranked else VARS[0]

n_norm = int((df.fault_id == 0).sum())
n_fault = int((df.fault_id != 0).sum())
render_insight(
    f"정상 **{n_norm:,}행** · 결함 **{n_fault:,}행**(IDV {df.loc[df.fault_id!=0,'fault_id'].nunique()}종), "
    f"52개 공정변수. 아래에서 변수별 **정상↔결함 분포 차이**와 **상관 구조**를 직접 탐색하세요 — "
    f"분포 차이가 큰 변수일수록 이상탐지에 유리합니다."
)

# ── 1) 분별력 상위 변수 (KS 검정) ────────────────────────────────────
st.subheader("정상 대비 결함 분별력 — KS 통계량 상위")
if not tests.empty and "ks_stat" in tests.columns:
    topk = tests.sort_values("ks_stat", ascending=False).head(15).iloc[::-1]
    fig_ks = go.Figure(go.Bar(
        x=topk["ks_stat"], y=topk["variable"], orientation="h",
        marker_color="#2C7BE5",
        text=[f"{v:.2f}" for v in topk["ks_stat"]], textposition="outside",
        hovertemplate="%{y}<br>KS=%{x:.3f}<extra></extra>",
    ))
    fig_ks.update_layout(xaxis_title="KS 통계량 (↑ 분별력 높음)", yaxis_title="")
    st.plotly_chart(style_fig(fig_ks, height=430), use_container_width=True)
    st.caption("KS 통계량이 클수록 정상·결함 분포 차이가 뚜렷 = 이상탐지 신호가 강한 변수.")
else:
    st.info("`fault_tests.json` 산출이 없어 변수 표준편차 기준으로 대체 정렬합니다.")

# ── 2) 정상 vs 결함 분포 비교 (변수 선택) ────────────────────────────
st.subheader("정상 vs 결함 분포 비교")
sel = st.selectbox(
    "변수 선택", ranked, index=0,
    format_func=lambda c: f"{c}  (분별력 상위)" if c == top_var else c,
)
c1, c2 = st.columns([3, 2])
with c1:
    fig_d = go.Figure()
    for grp, color in [("정상", STATUS_COLORS["정상"]), ("결함", STATUS_COLORS["이상"])]:
        fig_d.add_trace(go.Histogram(
            x=df.loc[df["구분"] == grp, sel], name=grp,
            marker_color=color, opacity=0.6, nbinsx=45,
        ))
    fig_d.update_layout(barmode="overlay", xaxis_title=sel, yaxis_title="빈도")
    st.plotly_chart(style_fig(fig_d, height=330), use_container_width=True)
with c2:
    fig_v = go.Figure()
    for grp, color in [("정상", STATUS_COLORS["정상"]), ("결함", STATUS_COLORS["이상"])]:
        fig_v.add_trace(go.Violin(
            y=df.loc[df["구분"] == grp, sel], name=grp, box_visible=True,
            meanline_visible=True, line_color=color, fillcolor=color, opacity=0.5,
        ))
    fig_v.update_layout(yaxis_title=sel, showlegend=False)
    st.plotly_chart(style_fig(fig_v, height=330), use_container_width=True)

# ── 3) 변수 시계열 + 결함 구간 음영 ──────────────────────────────────
st.subheader("변수 시계열 — 결함 구간 음영")
sel_ts = st.selectbox("시계열 변수", ranked, index=0, key="ts_var")
sdf = df.reset_index(drop=True)
fig_ts = go.Figure()
fig_ts.add_trace(go.Scatter(
    y=sdf[sel_ts], mode="lines", name=sel_ts, line=dict(color="#2C7BE5", width=1.2),
))
# 결함 구간(연속 블록) 음영 + 라벨
mask = (sdf["fault_id"] != 0).values
i = 0
while i < len(mask):
    if mask[i]:
        j = i
        while j < len(mask) and mask[j] and sdf["fault_id"].iloc[j] == sdf["fault_id"].iloc[i]:
            j += 1
        fig_ts.add_vrect(
            x0=i, x1=j - 1, fillcolor="rgba(231,76,60,0.10)", line_width=0,
            annotation_text=fault_name(sdf["fault_id"].iloc[i]).split(" · ")[0],
            annotation_position="top left", annotation_font_size=9,
        )
        i = j
    else:
        i += 1
fig_ts.update_layout(xaxis_title="샘플 인덱스(시간순)", yaxis_title=sel_ts)
st.plotly_chart(style_fig(fig_ts, height=320), use_container_width=True)
st.caption("빨강 음영 = 결함 주입 구간. 정상 구간 대비 값이 어떻게 이탈하는지 확인.")

# ── 4) 상관 히트맵 (분별력 상위 20변수) ──────────────────────────────
st.subheader("변수 상관 히트맵 (분별력 상위 20)")
corr_vars = ranked[:20] if len(ranked) >= 20 else ranked
corr = df[corr_vars].corr()
fig_h = go.Figure(go.Heatmap(
    z=corr.values, x=corr.columns, y=corr.columns,
    colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
    colorbar=dict(title="상관"),
    hovertemplate="%{y} ↔ %{x}<br>r=%{z:.2f}<extra></extra>",
))
fig_h.update_layout(height=560)
st.plotly_chart(style_fig(fig_h, height=560), use_container_width=True)
st.caption("빨강=양의 상관, 파랑=음의 상관. 강하게 묶인 변수군은 함께 이상 신호를 낸다.")

# ── 5) 정적 산출물 (mfg-eda PNG) — 보조 ──────────────────────────────
with st.expander("📎 정적 EDA 산출물 (matplotlib PNG)"):
    for title, fn in [
        ("52변수 상관구조", "corr_heatmap.png"),
        ("주요 변수 분포 비교", "dist_normal_vs_fault.png"),
        ("결함 구간 표시 시계열", "timeseries_key_vars.png"),
        ("결함 모드(IDV) 분포", "fault_counts.png"),
    ]:
        p = EDA_DIR / fn
        if p.exists():
            st.image(str(p), caption=title, use_container_width=True)

# ── 6) 변수별 요약통계 ───────────────────────────────────────────────
st.subheader("변수별 요약통계")
summary_path = EDA_DIR / "summary_stats.parquet"
if summary_path.exists():
    st.dataframe(pd.read_parquet(summary_path), use_container_width=True, height=300)
else:
    st.dataframe(df[VARS].describe().T, use_container_width=True, height=300)

render_footer()

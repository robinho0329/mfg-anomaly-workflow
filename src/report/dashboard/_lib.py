"""대시보드 공통 BI 헬퍼 — 인더스트리얼(제조 관제) 테마.

Tableau Public의 산업/공정 모니터링 대시보드 관례(BAN 타일·헤더 밴드·정돈된 차트·
강한 시그널 액센트)를 제조 이상탐지 도메인에 맞게 이식.

배치: entrypoint 폴더(src/report/dashboard)에 두어 app.py·pages 모두 `import _lib`로 사용.
(Streamlit이 entrypoint 폴더를 sys.path에 올리므로 top-level import 가능.)
"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------
# 상수 — 링크 / 색상 (인더스트리얼 팔레트)
# ----------------------------------------------------------------------
GITHUB_URL = "https://github.com/robinho0329/mfg-anomaly-workflow"

# 범주형 팔레트 (모델/변수 구분) — 강철·시그널 계열
CHART_COLORS = [
    "#2C7BE5", "#E67E22", "#17A2B8", "#6F42C1",
    "#E74C3C", "#27AE60", "#D35400", "#34495E",
]
INK = "#e6edf3"        # 다크 헤더 위 본문(밝은 글자)
INK_DARK = "#1c2733"   # 밝은 배경 위 본문
MUTED = "#8b97a3"
GRID = "rgba(120,140,160,0.18)"

# 상태 색 (정상/이상)
STATUS_COLORS = {"정상": "#27AE60", "이상": "#E74C3C", "주의": "#E67E22"}


def inject_css() -> None:
    """기본 Streamlit 크롬을 숨기고 인더스트리얼 BI 톤을 입히는 전역 CSS.

    각 페이지 상단 set_page_config 직후 1회 호출.
    """
    st.markdown(
        """
        <style>
        /* 기본 크롬 제거 — '관제 앱' 느낌 */
        #MainMenu, footer, header [data-testid="stToolbar"] {visibility: hidden;}
        [data-testid="stDecoration"] {display:none;}
        .block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1400px;}
        /* 헤더 밴드 — 그래파이트→강철 그라데이션 (제조 관제 톤) */
        .dash-header {
            background: linear-gradient(110deg, #11181f 0%, #223140 55%, #2f4a5e 100%);
            color: #fff; padding: 18px 24px; border-radius: 12px; margin-bottom: 18px;
            box-shadow: 0 4px 18px rgba(17,24,31,0.22);
            border-left: 5px solid #E67E22;
        }
        .dash-header h1 {color:#fff; font-size: 1.5rem; margin:0; font-weight:800;}
        .dash-header p {color: #c4d2de; margin: 6px 0 0; font-size: 0.9rem;}
        /* KPI BAN 타일 */
        .kpi-card {
            background:#fff; border-radius:12px; padding:16px 18px;
            border:1px solid #e6ebf0; border-top:4px solid var(--accent,#E67E22);
            box-shadow:0 2px 10px rgba(28,39,51,0.06); height:100%;
        }
        .kpi-label {font-size:0.82rem; color:#7b8a9a; font-weight:600; margin:0;}
        .kpi-value {font-size:1.9rem; font-weight:800; color:#1c2733; line-height:1.15; margin:2px 0 0;}
        .kpi-unit  {font-size:0.95rem; font-weight:600; color:#7b8a9a;}
        .kpi-delta {font-size:0.82rem; font-weight:700; margin-top:4px;}
        .kpi-sub   {font-size:0.72rem; color:#9aa7b4; margin-top:6px;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def dash_header(title: str, subtitle: str = "") -> None:
    """Tableau식 헤더 밴드(그라데이션 배너)."""
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f'<div class="dash-header"><h1>{title}</h1>{sub}</div>',
        unsafe_allow_html=True,
    )


def kpi_tile(
    label: str,
    value: str,
    *,
    unit: str = "",
    delta: str | None = None,
    delta_good: bool | None = None,
    accent: str = "#E67E22",
    sub: str = "",
) -> None:
    """BAN(Big Aggregate Number) 타일 — 큰 숫자 + 라벨 + 증감 + 보조설명."""
    if delta is None:
        delta_html = ""
    else:
        color = {True: "#27AE60", False: "#E74C3C", None: "#7b8a9a"}[delta_good]
        delta_html = f'<div class="kpi-delta" style="color:{color}">{delta}</div>'
    unit_html = f'<span class="kpi-unit"> {unit}</span>' if unit else ""
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    st.markdown(
        f"""
        <div class="kpi-card" style="--accent:{accent}">
            <p class="kpi-label">{label}</p>
            <div class="kpi-value">{value}{unit_html}</div>
            {delta_html}{sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_fig(fig: go.Figure, *, height: int | None = None) -> go.Figure:
    """plotly figure에 BI 공통 테마 적용(투명 배경·정돈된 그리드·일관 폰트)."""
    fig.update_layout(
        template="plotly_white",
        font=dict(family="sans-serif", size=12, color=INK_DARK),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=30, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        colorway=CHART_COLORS,
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    if height:
        fig.update_layout(height=height)
    return fig


def render_sidebar() -> None:
    """모든 페이지 공통 사이드바 — 워크플로우 단계·링크."""
    with st.sidebar:
        st.markdown("### 🏭 제조 이상탐지")
        st.caption("TEP 다변량 · 딥러닝 AE 워크플로우")
        st.divider()
        st.markdown("**⚙️ 자동화 4단계**")
        st.markdown(
            "① 수집 → ② EDA/통계 → ③ 딥러닝 탐지 → ④ 리포트\n\n"
            "각 단계는 산출물(parquet·PNG)만 로드 — 무거운 재계산 없음."
        )
        st.divider()
        st.markdown("**🔗 링크**")
        st.markdown(f"[📂 GitHub 레포]({GITHUB_URL})")
        st.divider()
        st.caption("포트폴리오: 제조 AI/예지보전 · 준지도 이상탐지(정상만 학습)")


def render_footer() -> None:
    """모든 페이지 하단 공통 푸터 — 포트폴리오 프레이밍."""
    st.divider()
    st.caption(
        f"🏭 실제 TEP 벤치마크 이상탐지 · 수집→EDA→딥러닝(Transformer/VAE/LSTM-AE)→리포트 "
        f"· [코드]({GITHUB_URL}) · 제조 AI/예지보전 직무 포트폴리오"
    )


# ----------------------------------------------------------------------
# 도메인 상수 / 인사이트 / 데이터 로더
# ----------------------------------------------------------------------
# TEP 결함 모드(IDV) 짧은 설명 — 라벨/툴팁용 (Downs & Vogel 표준 disturbance)
FAULT_LABELS: dict[int, str] = {
    0: "정상 운전",
    1: "IDV1 · A/C 공급비 변화",
    2: "IDV2 · B 조성 변화",
    3: "IDV3 · D 공급온도(난탐지)",
    4: "IDV4 · 반응기 냉각수 입구온도",
    6: "IDV6 · A 공급 손실",
    7: "IDV7 · C 헤더 압력손실",
    8: "IDV8 · A/B/C 조성 변화",
    12: "IDV12 · 응축기 냉각수온도(랜덤)",
    14: "IDV14 · 반응기 냉각밸브 고착",
    18: "IDV18 · 미지 결함(난탐지)",
}


def fault_name(fid: int) -> str:
    """fault_id → 짧은 라벨."""
    return FAULT_LABELS.get(int(fid), f"IDV{int(fid)}")


def render_insight(body: str) -> None:
    """데이터 해석을 쉬운 말로 전달하는 인사이트 콜아웃."""
    st.info(f"💡 {body}")


@st.cache_data(show_spinner=False, ttl=600)
def load_stream():
    """stream.db(tep_stream) 원시 52변수 스트림 로드. 없으면 빈 DataFrame (배포 방어)."""
    import sqlite3

    import pandas as pd

    from config.settings import COLLECT_TABLE, DB_PATH

    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql(f"SELECT * FROM {COLLECT_TABLE}", conn)
    except Exception:  # noqa: BLE001 — 테이블 부재 등 배포 환경 방어
        return pd.DataFrame()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df

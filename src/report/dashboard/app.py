"""④ 대시보드 — Streamlit 멀티페이지 진입점. (소유: mfg-reporter)

실행: streamlit run src/report/dashboard/app.py
페이지: pages/ 디렉토리에 단계별 모듈 분리(수집현황·EDA·이상탐지).
"""

import streamlit as st

# ── sys.path 부트스트랩: 레포 루트(config)·대시보드 폴더(_lib) 보장 (로컬·클라우드 공통) ──
import sys as _sys
from pathlib import Path as _Path
for _anc in _Path(__file__).resolve().parents:
    if (_anc / "_lib.py").exists() and str(_anc) not in _sys.path:
        _sys.path.insert(0, str(_anc))
    if (_anc / "config" / "settings.py").exists() and str(_anc) not in _sys.path:
        _sys.path.insert(0, str(_anc))

from _lib import (
    dash_header,
    inject_css,
    render_footer,
    render_sidebar,
    render_tep_pid,
)

st.set_page_config(page_title="제조 공정 이상탐지 워크플로우", page_icon="🏭", layout="wide")
inject_css()
render_sidebar()
dash_header(
    "🏭 제조 공정 이상탐지 자동화 워크플로우",
    "실제 TEP 벤치마크 · 수집 → 전처리/EDA → 딥러닝 이상탐지 → 리포트 (준지도 AE 계열)",
)

# ── TEP 공정 개요 + 구성도 ────────────────────────────────────────────
st.subheader("🧪 Tennessee Eastman Process (TEP)란?")
st.markdown(
    "**TEP**는 이스트만 케미컬(Downs & Vogel, 1993)이 공개한 **가상 화학공정 시뮬레이션**으로, "
    "공정 이상탐지 연구의 **표준 벤치마크**입니다. 기체 반응물 4종(A·C·D·E)으로 액체 제품 2종(G·H)을 "
    "생산하며, 반응기·응축기·분리기·압축기·스트리퍼로 구성됩니다. "
    "이 대시보드는 그 실제 벤치마크 데이터(52변수 시계열)로 이상탐지 파이프라인을 검증합니다."
)

st.markdown("**공정 구성도 (P&ID)**")
render_tep_pid()
st.caption(
    "반응기를 중심으로 응축·분리·재순환·스트리핑을 거쳐 제품(G·H) 생산. "
    "**파란 원**=계기 버블(측정변수 XMEAS: FI·PI·TI·LI·AI), **주황 밸브**=제어밸브(조작변수 XMV 1~11), "
    "**원 안 숫자**=공정 스트림(1~11), **점선**=재순환."
)

_c1, _c2 = st.columns(2)
with _c1:
    st.markdown(
        "**데이터 스키마 (52변수 + 라벨)**\n"
        "- **XMEAS 1~41** — 측정변수: 유량·압력·온도·액위·**조성 분석값**\n"
        "- **XMV 1~11** — 조작변수: 밸브 개도(공급·재순환·퍼지·냉각수)\n"
        "- **fault_id** — `0`=정상, `1~20`=결함 모드(**IDV**)\n"
        "- **샘플링** — 3분 간격 다변량 시계열"
    )
with _c2:
    st.markdown(
        "**결함 모드(IDV) 10종 — 이 대시보드 수록**\n"
        "- 잘 탐지: IDV 1·2·4·6·7·8·14\n"
        "- 부분 탐지: IDV 12\n"
        "- **난탐지(미탐): IDV 3·18** — 실 TEP에서도 알려진 난제"
    )

st.divider()

st.markdown(
    """
    이 대시보드는 **수집 → 전처리/EDA → 딥러닝 이상탐지 → 리포트** 자동화 워크플로우의
    산출물을 단계별로 보여줍니다(무거운 재계산 없이 산출물 로드만).

    데이터는 **실제 Tennessee Eastman Process 벤치마크**(Downs & Vogel, Braatz 배포판)를
    프로젝트 스키마(52변수)로 적재한 것입니다 — 정상 운전 + 결함 IDV 10종.

    | 단계 | 산출물 | 담당 에이전트 | 페이지 |
    |------|--------|--------------|--------|
    | ① 수집 | `data/db/stream.db` (1,940행 · TEP 52변수) | `mfg-collector` | ① 수집현황 |
    | ② 전처리·EDA·통계 | `data/eda/*` (PNG·통계 검정) | `mfg-eda` | ② EDA통계 |
    | ③ 딥러닝 이상탐지 | `data/models/scores.parquet` | `mfg-model` | ③ 이상탐지 |
    | ④ 모델 비교·리포트 | `data/models/comparison.*`, `reports/` | `mfg-reporter` | ④ 모델비교 |

    **핵심 요약**: 정상 1,460 / 결함 480 (IDV 1·2·3·4·6·7·8·12·14·18) · 기본 모델 **Transformer-AE**
    (정상만 학습한 시퀀스 오토인코더 + 재구성오차의 **마할라노비스 거리** 판정,
    F1 0.85 · ROC-AUC 0.92). VAE는 **오탐 0건(정밀도 1.00)**로 정밀도 우선 운용에 유리.
    IDV 3·18은 실제 TEP에서도 재구성오차가 정상과 구별되지 않아 **미탐(0.00)** — 한계도 그대로 표기합니다.

    왼쪽 사이드바에서 페이지를 선택하세요.
    """
)

render_footer()

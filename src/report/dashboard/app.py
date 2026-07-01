"""④ 대시보드 — Streamlit 멀티페이지 진입점. (소유: mfg-reporter)

실행: streamlit run src/report/dashboard/app.py
페이지: pages/ 디렉토리에 단계별 모듈 분리(수집현황·EDA·이상탐지).
"""

import streamlit as st

from _lib import dash_header, inject_css, render_footer, render_sidebar

st.set_page_config(page_title="제조 공정 이상탐지 워크플로우", page_icon="🏭", layout="wide")
inject_css()
render_sidebar()
dash_header(
    "🏭 제조 공정 이상탐지 자동화 워크플로우",
    "TEP 다변량 시계열 · 수집 → 전처리/EDA → 딥러닝 이상탐지 → 리포트 (준지도 AE 계열)",
)

st.markdown(
    """
    이 대시보드는 **수집 → 전처리/EDA → 딥러닝 이상탐지 → 리포트** 자동화 워크플로우의
    산출물을 단계별로 보여줍니다(무거운 재계산 없이 산출물 로드만).

    | 단계 | 산출물 | 담당 에이전트 | 페이지 |
    |------|--------|--------------|--------|
    | ① 수집 | `data/db/stream.db` (885행 · TEP 52변수) | `mfg-collector` | ① 수집현황 |
    | ② 전처리·EDA·통계 | `data/eda/*` (PNG·통계 검정) | `mfg-eda` | ② EDA통계 |
    | ③ 딥러닝 이상탐지 | `data/models/scores.parquet` | `mfg-model` | ③ 이상탐지 |
    | ④ 모델 비교·리포트 | `data/models/comparison.*`, `reports/` | `mfg-reporter` | ④ 모델비교 |

    **핵심 요약**: 정상 690 / 결함 195(IDV 10종) · 기본 모델 **VAE**
    (정상만 학습한 시퀀스 오토인코더 + 재구성오차의 **마할라노비스 거리** 판정,
    F1 0.78 · ROC-AUC 0.83 · 오탐 0.9%).
    IDV 1·4·8은 AE 원리상 미탐 — 한계도 그대로 표기합니다.

    왼쪽 사이드바에서 페이지를 선택하세요.
    """
)

render_footer()

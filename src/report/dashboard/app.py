"""④ 대시보드 — Streamlit 멀티페이지 진입점. (소유: mfg-reporter)

실행: streamlit run src/report/dashboard/app.py
페이지: pages/ 디렉토리에 단계별 모듈 분리(수집현황·EDA·이상탐지).
"""

import streamlit as st

st.set_page_config(page_title="제조 공정 이상탐지 워크플로우", page_icon="🏭", layout="wide")

st.title("🏭 제조 공정 이상탐지 자동화 워크플로우")
st.caption("TEP 다변량 시계열 · 수집 → 전처리/EDA → 딥러닝 이상탐지 → 리포트")

st.markdown(
    """
    이 대시보드는 **자동화 워크플로우**의 산출물을 단계별로 보여줍니다.

    | 단계 | 산출물 | 담당 에이전트 |
    |------|--------|--------------|
    | ① 수집 | `data/db/stream.db` | `mfg-collector` |
    | ② 전처리·EDA·통계 | `data/processed`, `data/eda` | `mfg-eda` |
    | ③ 딥러닝 이상탐지 | `data/models/scores.parquet` | `mfg-model` |
    | ④ 대시보드·PPT | `reports/` | `mfg-reporter` |

    왼쪽 사이드바에서 페이지를 선택하세요.
    """
)

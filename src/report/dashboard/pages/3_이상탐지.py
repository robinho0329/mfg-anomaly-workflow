"""④ 대시보드 — 이상탐지 결과 페이지. (소유: mfg-reporter)"""

import pandas as pd
import streamlit as st

from config.settings import MODELS_DIR

st.title("③ 딥러닝 이상탐지 결과")

scores_path = MODELS_DIR / "scores.parquet"
if not scores_path.exists():
    st.warning("탐지 결과가 없습니다. `python -m src.models.detect` 를 먼저 실행하세요.")
    st.stop()

df = pd.read_parquet(scores_path)
col1, col2, col3 = st.columns(3)
col1.metric("총 샘플", f"{len(df):,}")
col2.metric("이상 탐지", f"{int(df['is_anomaly'].sum()):,}")
col3.metric("이상 비율", f"{df['is_anomaly'].mean() * 100:.1f}%")

st.subheader("이상 점수 타임라인")
ts = df.set_index("timestamp")[["anomaly_score", "threshold"]]
st.line_chart(ts)

st.subheader("이상 구간 상세")
st.dataframe(df[df["is_anomaly"] == 1][["timestamp", "fault_id", "anomaly_score"]], use_container_width=True)

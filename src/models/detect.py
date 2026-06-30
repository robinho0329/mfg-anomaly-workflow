"""③ 딥러닝 이상탐지 — 학습·탐지 오케스트레이션. (소유: mfg-model)

흐름: processed/clean → 정상구간으로 스케일러+LSTM-AE 학습 → 전체 점수 산출
      → 정상 분위수 임계값으로 이상 플래그 → data/models/scores.parquet 저장.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config.settings import (
    ANOMALY_QUANTILE,
    MODELS_DIR,
    PROCESS_COLS,
    PROCESSED_DIR,
    SEQ_LEN,
)
from src.models.lstm_ae import LSTMAutoencoder, make_sequences


def _load_clean() -> pd.DataFrame:
    path = PROCESSED_DIR / "clean.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def run() -> pd.DataFrame:
    """학습 + 탐지 실행 → 점수/플래그 Parquet 저장."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = _load_clean()
    if len(df) < SEQ_LEN + 10:
        print(f"[model] 데이터 부족({len(df)}행) — collect/pipeline 먼저 실행")
        return pd.DataFrame()

    normal = df[df["fault_id"] == 0]
    scaler = StandardScaler().fit(normal[PROCESS_COLS])
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")

    Xn = scaler.transform(normal[PROCESS_COLS])
    ae = LSTMAutoencoder(n_features=len(PROCESS_COLS)).fit(make_sequences(Xn))

    Xall = scaler.transform(df[PROCESS_COLS])
    seqs = make_sequences(Xall)
    scores = ae.reconstruction_error(seqs)

    # 정상 점수 분위수로 임계값 설정
    train_scores = ae.reconstruction_error(make_sequences(Xn))
    thr = float(np.quantile(train_scores, ANOMALY_QUANTILE))

    # 시퀀스 점수를 마지막 스텝 기준으로 행에 정렬
    out = df.iloc[SEQ_LEN - 1:].copy()
    out["anomaly_score"] = scores
    out["threshold"] = thr
    out["is_anomaly"] = (scores > thr).astype(int)
    out.to_parquet(MODELS_DIR / "scores.parquet", engine="pyarrow", index=False)
    print(f"[model] 탐지 완료: {out['is_anomaly'].sum()}/{len(out)} 이상, 임계값={thr:.4f}")
    return out


if __name__ == "__main__":
    run()

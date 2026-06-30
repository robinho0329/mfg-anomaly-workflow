"""③ 딥러닝 이상탐지 — 학습·탐지 오케스트레이션. (소유: mfg-model)

흐름: processed/clean → 정상구간으로 스케일러+AE 학습 → 전체 점수 산출
      → 평활 + 정상분위 임계값(여유 배수)으로 이상 플래그
      → data/models/scores.parquet 저장.

출력 스키마(고정, mfg-reporter 의존): timestamp, fault_id, anomaly_score, threshold, is_anomaly
모델 교체 가능: MODEL_REGISTRY에서 이름으로 선택. 기본은 비교에서 가장 우수한 모델.
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
    SCORE_SMOOTH_WINDOW,
    SEQ_LEN,
    THRESHOLD_MARGIN,
)
from src.models.lstm_ae import LSTMAutoencoder, make_sequences
from src.models.scoring import score_to_flags, sequence_labels
from src.models.transformer_ae import TransformerAutoencoder
from src.models.vae import LSTMVae

# 사용 가능한 모델 레지스트리(동일 인터페이스: build/fit/reconstruction_error)
MODEL_REGISTRY = {
    "lstm_ae": LSTMAutoencoder,
    "vae": LSTMVae,
    "transformer_ae": TransformerAutoencoder,
}

# 비교 평가(compare.py)에서 F1·recall 최고였던 모델을 기본값으로 사용
# transformer_ae: F1=0.947, recall=0.960 (lstm_ae는 F1=0.929·FP=0 대안)
DEFAULT_MODEL = "transformer_ae"


def _load_clean() -> pd.DataFrame:
    path = PROCESSED_DIR / "clean.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def fit_and_score(
    df: pd.DataFrame,
    model_name: str = DEFAULT_MODEL,
    quantile: float = ANOMALY_QUANTILE,
    margin: float = THRESHOLD_MARGIN,
    smooth_window: int = SCORE_SMOOTH_WINDOW,
    save_scaler: bool = False,
) -> pd.DataFrame:
    """단일 모델 학습 + 전체 스트림 탐지 → 출력 프레임 반환(저장은 호출측).

    학습 표본은 윈도우 전체가 정상인 '순수 정상 시퀀스'로 한정(준지도).
    임계값은 전체 스트림 중 마지막 스텝이 정상인 시퀀스 점수의 고분위×여유로 보정.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"알 수 없는 모델: {model_name} (가능: {list(MODEL_REGISTRY)})")
    if len(df) < SEQ_LEN + 10:
        return pd.DataFrame()

    # 1) 스케일링: 정상 운전 구간 기준
    normal = df[df["fault_id"] == 0]
    scaler = StandardScaler().fit(normal[PROCESS_COLS])
    if save_scaler:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, MODELS_DIR / "scaler.joblib")

    # 2) 전체 스트림 시퀀스 + 라벨
    Xall = scaler.transform(df[PROCESS_COLS])
    seqs = make_sequences(Xall)
    labels = sequence_labels(df["fault_id"].to_numpy())
    last_fault, pure_normal = labels["last_fault"], labels["pure_normal"]

    # 3) 순수 정상 시퀀스로 학습(graceful: 부족하면 마지막 스텝 정상 시퀀스로 폴백)
    train_seqs = seqs[pure_normal]
    if len(train_seqs) < 10:
        train_seqs = seqs[last_fault == 0]
    model = MODEL_REGISTRY[model_name](n_features=len(PROCESS_COLS)).fit(train_seqs)

    # 4) 전체 점수 → 평활 + 임계값(순수 정상 보정) + 플래그
    raw = model.reconstruction_error(seqs)
    flags = score_to_flags(raw, pure_normal, quantile, margin, smooth_window)

    out = df.iloc[SEQ_LEN - 1:].copy()
    out["anomaly_score"] = flags["score"]
    out["threshold"] = flags["threshold"]
    out["is_anomaly"] = flags["is_anomaly"]
    return out


def run(model_name: str = DEFAULT_MODEL) -> pd.DataFrame:
    """학습 + 탐지 실행 → 점수/플래그 Parquet 저장."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = _load_clean()
    if len(df) < SEQ_LEN + 10:
        print(f"[model] 데이터 부족({len(df)}행) — collect/pipeline 먼저 실행")
        return pd.DataFrame()

    out = fit_and_score(df, model_name=model_name, save_scaler=True)
    if out.empty:
        print("[model] 탐지 산출 없음(데이터 부족)")
        return out

    cols = ["timestamp", "fault_id", "anomaly_score", "threshold", "is_anomaly"]
    out[cols].to_parquet(MODELS_DIR / "scores.parquet", engine="pyarrow", index=False)
    thr = float(out["threshold"].iloc[0])
    print(
        f"[model:{model_name}] 탐지 완료: "
        f"{int(out['is_anomaly'].sum())}/{len(out)} 이상, 임계값={thr:.4f}"
    )
    return out


if __name__ == "__main__":
    run()

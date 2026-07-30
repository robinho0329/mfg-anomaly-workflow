"""③ 딥러닝 이상탐지 — 학습·탐지 오케스트레이션. (소유: mfg-model)

흐름: processed/clean → 시간순 앞 TRAIN_RATIO 구간의 정상만으로 스케일러+AE+임계값 적합
      → 전체 스트림 점수 산출(운영 스코어링) → 평활 + 임계값으로 이상 플래그
      → data/models/scores.parquet 저장.

시간 분할(누수 방지): 적합은 경계 이전 구간만 사용한다. 각 행은 split 컬럼으로
train / purge / eval 구간이 표시되며, 평가 지표(compare.py)는 eval 구간만 쓴다.

출력 스키마(mfg-reporter 의존): timestamp, fault_id, anomaly_score, threshold, is_anomaly
+ split(train|purge|eval). 모델 교체 가능: MODEL_REGISTRY에서 이름으로 선택.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config.settings import (
    ANOMALY_QUANTILE,
    DEFAULT_MODEL,
    MODELS_DIR,
    PROCESS_COLS,
    PROCESSED_DIR,
    PURGE_STEPS,
    SCORE_SMOOTH_WINDOW,
    SEQ_LEN,
    THRESHOLD_MARGIN,
    TRAIN_RATIO,
)
from src.models.lstm_ae import LSTMAutoencoder, make_sequences
from src.models.scoring import MahalanobisScorer, score_to_flags, sequence_labels
from src.models.transformer_ae import TransformerAutoencoder
from src.models.vae import LSTMVae

# 사용 가능한 모델 레지스트리(동일 인터페이스: build/fit/reconstruction_error)
MODEL_REGISTRY = {
    "lstm_ae": LSTMAutoencoder,
    "vae": LSTMVae,
    "transformer_ae": TransformerAutoencoder,
}

# 기본 모델은 config.settings.DEFAULT_MODEL 단일 정의를 따른다 (보고 모듈과 동기화)


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
    smooth_center: bool | None = None,
    save_scaler: bool = False,
    train_frac: float | None = TRAIN_RATIO,
) -> pd.DataFrame:
    """단일 모델 학습 + 전체 스트림 탐지 → 출력 프레임 반환(저장은 호출측).

    시간 분할: 시간순 앞 train_frac 구간에서만 스케일러·모델·임계값을 적합한다.
    학습 표본은 그 구간에서 윈도우 전체가 정상인 '순수 정상 시퀀스'(준지도).
    점수는 마지막 스텝 변수별 오차 벡터의 마할라노비스 거리(학습구간 정상 분포 기준),
    임계값은 학습구간 순수 정상 점수의 고분위×여유로 보정한다.

    출력의 split 컬럼: train(경계 이전) / purge(경계 직후 중첩 구간) / eval(홀드아웃).
    학습구간 표본이 10 시퀀스 미만이면 분할을 포기하고 전 구간 적합으로 폴백하며
    (소데이터 graceful) 이때 split은 전부 train으로 표시된다.
    출력 attrs["split"]에 분할 메타(방식·경계 시각·행수)를 담는다.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"알 수 없는 모델: {model_name} (가능: {list(MODEL_REGISTRY)})")
    if len(df) < SEQ_LEN + 10:
        return pd.DataFrame()

    # 0) 시간순 정렬 보장 — 분할 경계가 시간 경계가 되도록
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    boundary = int(n * train_frac) if train_frac else n

    # 1) 스케일링: 학습구간 정상 운전 기준 (경계 밖 통계 유입 금지)
    normal_train = df.iloc[:boundary]
    normal_train = normal_train[normal_train["fault_id"] == 0]
    if len(normal_train) < SEQ_LEN:
        normal_train = df[df["fault_id"] == 0]  # 소데이터 폴백
        boundary = n
    scaler = StandardScaler().fit(normal_train[PROCESS_COLS])
    if save_scaler:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, MODELS_DIR / "scaler.joblib")

    # 2) 전체 스트림 시퀀스 + 라벨
    Xall = scaler.transform(df[PROCESS_COLS])
    seqs = make_sequences(Xall)
    labels = sequence_labels(df["fault_id"].to_numpy())
    last_fault, pure_normal = labels["last_fault"], labels["pure_normal"]

    # 시퀀스 i의 윈도우는 행 [i, i+SEQ_LEN-1] — 학습은 경계 안에서 끝나는 시퀀스만
    seq_end = np.arange(len(seqs)) + SEQ_LEN - 1
    in_train = seq_end < boundary

    # 3) 학습구간 순수 정상 시퀀스로 학습(부족 시 단계적 폴백)
    fit_mask = pure_normal & in_train
    if fit_mask.sum() < 10:
        fit_mask = (last_fault == 0) & in_train
    if fit_mask.sum() < 10:  # 분할 유지 불가 → 전 구간 폴백(무분할로 표시)
        boundary = n
        in_train = seq_end < boundary
        fit_mask = pure_normal if pure_normal.sum() >= 10 else (last_fault == 0)
    model = MODEL_REGISTRY[model_name](n_features=len(PROCESS_COLS)).fit(seqs[fit_mask])

    # 4) 마할라노비스 점수: 학습구간 정상 오차벡터 분포 기준 거리
    err_all = model.reconstruction_error_vector(seqs)
    scorer = MahalanobisScorer().fit(err_all[fit_mask])
    raw = scorer.score(err_all)

    # 5) 평활 + 임계값(학습구간 순수 정상 기준) + 플래그
    flags = score_to_flags(raw, pure_normal & in_train, quantile, margin,
                           smooth_window, smooth_center)

    out = df.iloc[SEQ_LEN - 1:].copy()
    out["anomaly_score"] = flags["score"]
    out["threshold"] = flags["threshold"]
    out["is_anomaly"] = flags["is_anomaly"]

    # 6) 구간 표시: 윈도우가 경계 이전에 끝나면 train,
    #    시작이 경계+PURGE 이후면 eval, 그 사이(중첩·완충)는 purge
    seq_start = seq_end - (SEQ_LEN - 1)
    split = np.where(
        seq_end < boundary, "train",
        np.where(seq_start >= boundary + PURGE_STEPS, "eval", "purge"),
    )
    out["split"] = split
    out.attrs["split"] = {
        "method": "time_ordered",
        "train_ratio": float(train_frac) if train_frac and boundary < n else None,
        "boundary_time": str(df.iloc[boundary]["timestamp"]) if boundary < n else None,
        "purge_steps": int(PURGE_STEPS),
        "train_seqs": int((split == "train").sum()),
        "eval_seqs": int((split == "eval").sum()),
    }
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

    cols = ["timestamp", "fault_id", "anomaly_score", "threshold", "is_anomaly", "split"]
    out[cols].to_parquet(MODELS_DIR / "scores.parquet", engine="pyarrow", index=False)
    thr = float(out["threshold"].iloc[0])
    n_eval = int((out["split"] == "eval").sum())
    print(
        f"[model:{model_name}] 탐지 완료: "
        f"{int(out['is_anomaly'].sum())}/{len(out)} 이상, 임계값={thr:.4f}, "
        f"홀드아웃 평가 구간 {n_eval}행"
    )
    return out


if __name__ == "__main__":
    run()

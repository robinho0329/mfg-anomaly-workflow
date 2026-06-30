"""③ 딥러닝 이상탐지 — 시퀀스 라벨링·점수 평활·임계값 산정 공유 유틸. (소유: mfg-model)

detect.py(단일 모델 탐지)와 compare.py(모델 비교)가 동일한 점수→플래그 로직을
공유하도록 분리했다. 과탐(false positive) 억제 로직이 이 모듈에 집중된다.

핵심 아이디어(베이스라인 과탐 원인 대응):
  - 임계값을 fault 행을 제거해 이어붙인 '순수 정상 시퀀스'가 아니라,
    전체 스트림에서 **마지막 스텝이 정상인 시퀀스**(경계 오염 포함)의 고분위로 보정.
  - 분위 임계값에 안전 여유 배수(THRESHOLD_MARGIN)를 곱한다.
  - 행 점수를 rolling median으로 평활해 단발 스파이크 오탐을 억제한다.
"""

import numpy as np
import pandas as pd

from config.settings import (
    ANOMALY_QUANTILE,
    SCORE_SMOOTH_WINDOW,
    SEQ_LEN,
    THRESHOLD_MARGIN,
)


def sequence_labels(fault_id: np.ndarray, seq_len: int = SEQ_LEN) -> dict:
    """시퀀스(마지막 스텝 정렬)별 라벨 정보를 산출한다.

    make_sequences가 행 [seq_len-1 .. N-1]에 시퀀스를 부여하므로,
    각 시퀀스의 '대표 행'은 윈도우의 마지막 스텝이다.

    Returns:
        dict: last_fault(마지막 스텝 fault_id), pure_normal(윈도우 전체 정상 여부)
    """
    fault_id = np.asarray(fault_id)
    n = len(fault_id)
    if n < seq_len:
        return {"last_fault": np.empty(0, int), "pure_normal": np.empty(0, bool)}
    last_fault = fault_id[seq_len - 1:]
    # 윈도우 내 모든 스텝이 정상(==0)인지: 슬라이딩 최대값으로 판정
    pure_normal = np.array(
        [bool((fault_id[i:i + seq_len] == 0).all()) for i in range(n - seq_len + 1)]
    )
    return {"last_fault": last_fault, "pure_normal": pure_normal}


def smooth_scores(scores: np.ndarray, window: int = SCORE_SMOOTH_WINDOW) -> np.ndarray:
    """행 점수 시계열을 rolling median으로 평활(단발 스파이크 오탐 억제)."""
    if window <= 1 or len(scores) < window:
        return np.asarray(scores, dtype=float)
    s = pd.Series(scores, dtype=float)
    return s.rolling(window, center=True, min_periods=1).median().to_numpy()


def calibrate_threshold(
    scores: np.ndarray,
    calib_mask: np.ndarray,
    quantile: float = ANOMALY_QUANTILE,
    margin: float = THRESHOLD_MARGIN,
) -> float:
    """**순수 정상 시퀀스**(윈도우 전체 정상) 점수의 고분위 × 여유배수.

    경계 오염(정상→결함 전환 윈도우)을 제외한 깨끗한 정상 분포로 임계값을 정해,
    정상 구간 오탐을 억제하면서도 임계값이 과도하게 높아지지 않게 한다.
    여유배수(margin)로 노이즈 변동에 대한 안전마진을 확보한다.
    순수 정상 표본이 부족하면 전체 분위로 graceful 폴백.
    """
    scores = np.asarray(scores, dtype=float)
    calib_mask = np.asarray(calib_mask, dtype=bool)
    base = scores[calib_mask]
    if len(base) < 5:
        base = scores  # 정상 표본 부족 시 graceful 폴백
    return float(np.quantile(base, quantile) * margin)


def score_to_flags(
    raw_scores: np.ndarray,
    calib_mask: np.ndarray,
    quantile: float = ANOMALY_QUANTILE,
    margin: float = THRESHOLD_MARGIN,
    smooth_window: int = SCORE_SMOOTH_WINDOW,
) -> dict:
    """원시 시퀀스 점수 → (평활 점수, 임계값, 이상 플래그) 일괄 산출.

    Args:
        raw_scores: 시퀀스별 원시 재구성오차.
        calib_mask: 임계값 보정에 쓸 순수 정상 시퀀스 마스크.

    Returns:
        dict: score(평활), threshold(float), is_anomaly(0/1 배열)
    """
    smoothed = smooth_scores(raw_scores, smooth_window)
    thr = calibrate_threshold(smoothed, calib_mask, quantile, margin)
    is_anom = (smoothed > thr).astype(int)
    return {"score": smoothed, "threshold": thr, "is_anomaly": is_anom}

"""③ 딥러닝 이상탐지 — 시퀀스 라벨링·점수 평활·임계값 산정 공유 유틸. (소유: mfg-model)

detect.py(단일 모델 탐지)와 compare.py(모델 비교)가 동일한 점수→플래그 로직을
공유하도록 분리했다. 과탐(false positive) 억제 로직이 이 모듈에 집중된다.

핵심 아이디어:
  - 점수: 마지막 스텝의 **변수별 재구성오차 벡터**에 대한 마할라노비스 거리.
    정상 오차 분포(평균·공분산) 기준이라, 일부 변수만 이상한 결함과
    재구성오차가 정상보다 비정상적으로 작은 결함(부호 무관)도 분리한다.
  - 임계값: 순수 정상 시퀀스(윈도우 전체 정상) 점수 고분위 × 안전 여유 배수.
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


class MahalanobisScorer:
    """정상 재구성오차 벡터 분포 기준 마할라노비스 거리 스코어러.

    준지도: 순수 정상 시퀀스의 변수별 오차 벡터로 평균·공분산을 추정하고,
    임의 시퀀스 오차 벡터의 (제곱) 마할라노비스 거리를 이상 점수로 산출한다.
    공분산은 소표본/고차원 안정성을 위해 Ledoit-Wolf 축소 추정을 사용한다.
    """

    def __init__(self):
        self.mean_ = None
        self.estimator_ = None

    def fit(self, err_normal: np.ndarray) -> "MahalanobisScorer":
        """정상 오차 벡터 (N, F)로 평균·축소공분산 적합."""
        from sklearn.covariance import LedoitWolf

        err_normal = np.asarray(err_normal, dtype=float)
        self.mean_ = err_normal.mean(axis=0)
        self.estimator_ = LedoitWolf().fit(err_normal)
        return self

    def score(self, err: np.ndarray) -> np.ndarray:
        """오차 벡터 (W, F) → 제곱 마할라노비스 거리 (W,)."""
        err = np.asarray(err, dtype=float)
        if len(err) == 0:
            return np.empty(0)
        return self.estimator_.mahalanobis(err - self.mean_)


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

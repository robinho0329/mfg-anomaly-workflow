"""③ 딥러닝 이상탐지 모델 단위 테스트. (소유: mfg-model)

TensorFlow 학습은 무거우므로, TF가 필요한 테스트는 미설치 시 자동 스킵한다.
시퀀스 생성·점수/임계값 로직 등 순수 함수는 TF 없이도 검증한다.
"""

import importlib.util

import numpy as np
import pytest

from config.settings import PROCESS_COLS, SEQ_LEN
from src.models.lstm_ae import make_sequences
from src.models.scoring import (
    MahalanobisScorer,
    calibrate_threshold,
    score_to_flags,
    sequence_labels,
    smooth_scores,
)

_HAS_TF = importlib.util.find_spec("tensorflow") is not None
_RNG = np.random.default_rng(42)


# ── 시퀀스 생성 ────────────────────────────────────────────
def test_make_sequences_shape():
    """(N, F) → (N-seq_len+1, seq_len, F) 윈도우 형태."""
    X = _RNG.normal(size=(50, len(PROCESS_COLS)))
    seqs = make_sequences(X, seq_len=SEQ_LEN)
    assert seqs.shape == (50 - SEQ_LEN + 1, SEQ_LEN, len(PROCESS_COLS))


def test_make_sequences_too_short():
    """행 수 < seq_len 이면 빈 배열(graceful)."""
    X = _RNG.normal(size=(SEQ_LEN - 1, len(PROCESS_COLS)))
    assert make_sequences(X, seq_len=SEQ_LEN).shape[0] == 0


# ── 시퀀스 라벨링 ─────────────────────────────────────────
def test_sequence_labels():
    """마지막 스텝 fault / 순수 정상 판정."""
    fault = np.array([0] * 25 + [3] * 10)  # 35행
    lab = sequence_labels(fault, seq_len=SEQ_LEN)
    n_seq = 35 - SEQ_LEN + 1
    assert len(lab["last_fault"]) == n_seq
    assert len(lab["pure_normal"]) == n_seq
    # 마지막 시퀀스는 결함 행으로 끝남
    assert lab["last_fault"][-1] == 3
    # 첫 시퀀스(행 0~19)는 전부 정상
    assert bool(lab["pure_normal"][0]) is True
    # 결함이 윈도우에 포함된 시퀀스는 순수 정상이 아님
    assert bool(lab["pure_normal"][-1]) is False


# ── 점수 평활/임계값 ──────────────────────────────────────
def test_smooth_scores_reduces_spike():
    """단발 스파이크가 rolling median으로 억제된다."""
    s = np.ones(20)
    s[10] = 100.0
    sm = smooth_scores(s, window=5)
    assert sm[10] < 5.0  # 스파이크 평활됨
    assert len(sm) == len(s)


def test_calibrate_threshold_uses_normal_mask():
    """임계값은 순수 정상 분포 기반이며 결함 점수에 끌려가지 않는다."""
    scores = np.concatenate([np.ones(50), np.full(10, 100.0)])
    mask = np.array([True] * 50 + [False] * 10)
    thr = calibrate_threshold(scores, mask, quantile=0.99, margin=1.1)
    assert 1.0 <= thr < 2.0  # 정상(1.0) 근처, 결함(100)에 영향 안 받음


def test_calibrate_threshold_fallback():
    """정상 표본 부족 시 전체 분포로 graceful 폴백."""
    scores = np.arange(1, 11, dtype=float)
    mask = np.zeros(10, dtype=bool)  # 정상 표본 0개
    thr = calibrate_threshold(scores, mask, quantile=0.9, margin=1.0)
    assert thr > 0  # 예외 없이 산출


def test_score_to_flags_separation():
    """정상/결함이 분리될 때 결함만 플래그된다."""
    raw = np.concatenate([np.full(40, 1.0), np.full(10, 50.0)])
    mask = np.array([True] * 40 + [False] * 10)
    out = score_to_flags(raw, mask, quantile=0.99, margin=1.1, smooth_window=3)
    assert out["is_anomaly"][:40].sum() == 0      # 정상 구간 오탐 없음
    assert out["is_anomaly"][40:].sum() == 10     # 결함 전부 탐지


# ── 마할라노비스 스코어러 ─────────────────────────────────
def test_mahalanobis_scorer_separates_outlier():
    """정상 분포에서 벗어난 오차 벡터에 큰 점수를 부여한다."""
    err_normal = _RNG.normal(loc=1.0, scale=0.1, size=(200, 8))
    scorer = MahalanobisScorer().fit(err_normal)
    d_normal = scorer.score(err_normal)
    d_outlier = scorer.score(np.full((1, 8), 5.0))  # 평균에서 크게 벗어남
    assert d_outlier[0] > np.quantile(d_normal, 0.99)


def test_mahalanobis_scorer_low_error_outlier():
    """재구성오차가 비정상적으로 '작은' 벡터도 이상으로 분리(부호 무관)."""
    err_normal = _RNG.normal(loc=1.0, scale=0.1, size=(200, 8))
    scorer = MahalanobisScorer().fit(err_normal)
    d_low = scorer.score(np.zeros((1, 8)))  # 0(정상보다 작음)
    assert d_low[0] > np.quantile(scorer.score(err_normal), 0.95)


def test_mahalanobis_scorer_empty():
    """빈 입력 시 빈 배열(graceful)."""
    scorer = MahalanobisScorer().fit(_RNG.normal(size=(50, 8)))
    assert scorer.score(np.empty((0, 8))).shape[0] == 0


# ── 모델(TF 필요) ─────────────────────────────────────────
@pytest.mark.skipif(not _HAS_TF, reason="TensorFlow 미설치")
def test_reconstruction_error_shape():
    """재구성오차는 시퀀스당 스칼라 1개(마지막 스텝 기준)."""
    from src.models.lstm_ae import LSTMAutoencoder

    X = _RNG.normal(size=(40, len(PROCESS_COLS)))
    seqs = make_sequences(X)
    ae = LSTMAutoencoder(n_features=len(PROCESS_COLS)).fit(seqs)
    err = ae.reconstruction_error(seqs)
    assert err.shape == (len(seqs),)
    assert np.all(err >= 0)


@pytest.mark.skipif(not _HAS_TF, reason="TensorFlow 미설치")
def test_reconstruction_error_vector_shape():
    """변수별 오차 벡터는 (시퀀스수, 피처수) 형태."""
    from src.models.lstm_ae import LSTMAutoencoder

    X = _RNG.normal(size=(40, len(PROCESS_COLS)))
    seqs = make_sequences(X)
    ae = LSTMAutoencoder(n_features=len(PROCESS_COLS)).fit(seqs)
    ev = ae.reconstruction_error_vector(seqs)
    assert ev.shape == (len(seqs), len(PROCESS_COLS))
    assert np.all(ev >= 0)


@pytest.mark.skipif(not _HAS_TF, reason="TensorFlow 미설치")
def test_models_reproducible():
    """동일 시드 → 동일 재구성오차(결정성)."""
    from src.models.vae import LSTMVae

    X = _RNG.normal(size=(45, len(PROCESS_COLS)))
    seqs = make_sequences(X)
    e1 = LSTMVae(n_features=len(PROCESS_COLS)).fit(seqs).reconstruction_error(seqs)
    e2 = LSTMVae(n_features=len(PROCESS_COLS)).fit(seqs).reconstruction_error(seqs)
    np.testing.assert_allclose(e1, e2, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not _HAS_TF, reason="TensorFlow 미설치")
def test_empty_sequences_graceful():
    """빈 시퀀스 입력 시 빈 점수 배열 반환."""
    from src.models.transformer_ae import TransformerAutoencoder

    model = TransformerAutoencoder(n_features=len(PROCESS_COLS))
    model.build()
    assert model.reconstruction_error(np.empty((0, SEQ_LEN, len(PROCESS_COLS)))).shape[0] == 0

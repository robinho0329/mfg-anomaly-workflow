"""③ 딥러닝 이상탐지 — LSTM 오토인코더. (소유: mfg-model)

정상 운전 시퀀스만으로 학습한 재구성 모델. 재구성오차가 큰 구간을 이상으로 판정.
TensorFlow가 없는 환경에서도 import가 깨지지 않도록 지연 임포트한다.

확장 슬롯: VAE(vae.py) / Transformer-AE(transformer_ae.py)는 같은 인터페이스
(build / fit / reconstruction_error)로 형제 모듈로 추가되어 있다.

과탐 개선: 드롭아웃 정규화 + EarlyStopping(검증손실 기준)으로 과적합을 억제하고
정상 재구성 분포를 안정화한다.
"""

import numpy as np

from config.settings import (
    AE_BATCH,
    AE_DROPOUT,
    AE_EPOCHS,
    AE_LATENT_DIM,
    AE_PATIENCE,
    AE_VAL_SPLIT,
    RANDOM_STATE,
    SEQ_LEN,
)


def make_sequences(X: np.ndarray, seq_len: int = SEQ_LEN) -> np.ndarray:
    """(N, F) 행렬 → (N-seq_len+1, seq_len, F) 슬라이딩 윈도우."""
    if len(X) < seq_len:
        return np.empty((0, seq_len, X.shape[1]))
    return np.stack([X[i:i + seq_len] for i in range(len(X) - seq_len + 1)])


class LSTMAutoencoder:
    """시퀀스 재구성 기반 이상탐지기."""

    name = "lstm_ae"

    def __init__(self, n_features: int, seq_len: int = SEQ_LEN, latent_dim: int = AE_LATENT_DIM):
        self.n_features = n_features
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.model = None

    def build(self):
        """Keras LSTM-AE 구성(지연 임포트)."""
        from tensorflow.keras import layers, models

        from src.models.tf_seed import enable_determinism

        enable_determinism(RANDOM_STATE)
        inp = layers.Input(shape=(self.seq_len, self.n_features))
        enc = layers.LSTM(self.latent_dim, activation="tanh", dropout=AE_DROPOUT)(inp)
        dec = layers.RepeatVector(self.seq_len)(enc)
        dec = layers.LSTM(
            self.latent_dim, activation="tanh", return_sequences=True, dropout=AE_DROPOUT
        )(dec)
        out = layers.TimeDistributed(layers.Dense(self.n_features))(dec)
        self.model = models.Model(inp, out)
        self.model.compile(optimizer="adam", loss="mse")
        return self.model

    def fit(self, seqs: np.ndarray):
        """정상 시퀀스로 학습(EarlyStopping으로 과적합 억제)."""
        import tensorflow as tf

        if self.model is None:
            self.build()
        # 검증 표본이 확보될 때만 조기종료 적용(소표본 graceful 처리)
        use_val = len(seqs) >= 10
        callbacks = []
        val_split = 0.0
        if use_val:
            val_split = AE_VAL_SPLIT
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=AE_PATIENCE, restore_best_weights=True
                )
            )
        self.model.fit(
            seqs, seqs, epochs=AE_EPOCHS, batch_size=AE_BATCH,
            validation_split=val_split, shuffle=True, verbose=0, callbacks=callbacks,
        )
        return self

    def reconstruction_error(self, seqs: np.ndarray) -> np.ndarray:
        """시퀀스의 **마지막 스텝** 평균제곱 재구성오차 → 이상 점수.

        과탐 개선 핵심: 윈도우 전체 평균이 아니라 그 행 자신(마지막 스텝)의
        복원오차를 사용한다. 결함이 윈도우 앞부분에만 있고 마지막 스텝이 정상이면
        복원이 잘 되어, 결함 직후 회복 구간의 경계 오탐이 사라진다.
        """
        if len(seqs) == 0:
            return np.empty(0)
        pred = self.model.predict(seqs, verbose=0)
        return np.mean((seqs[:, -1, :] - pred[:, -1, :]) ** 2, axis=1)

    def reconstruction_error_vector(self, seqs: np.ndarray) -> np.ndarray:
        """시퀀스 마지막 스텝의 **변수별** 제곱오차 (W, F).

        마할라노비스 거리 점수에 쓰인다. 일부 변수만 이상한 결함도 포착하고,
        재구성오차가 정상보다 비정상적으로 작은 결함(부호 무관)도 분리한다.
        """
        if len(seqs) == 0:
            return np.empty((0, self.n_features))
        pred = self.model.predict(seqs, verbose=0)
        return (seqs[:, -1, :] - pred[:, -1, :]) ** 2

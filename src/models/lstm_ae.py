"""③ 딥러닝 이상탐지 — LSTM 오토인코더. (소유: mfg-model)

정상 운전 시퀀스만으로 학습한 재구성 모델. 재구성오차가 큰 구간을 이상으로 판정.
TensorFlow가 없는 환경에서도 import가 깨지지 않도록 지연 임포트한다.

확장 슬롯: VAE / Transformer-AE / USAD 등은 같은 인터페이스
(build / fit / reconstruction_error)로 형제 모듈을 추가한다.
"""

import numpy as np

from config.settings import AE_BATCH, AE_EPOCHS, AE_LATENT_DIM, RANDOM_STATE, SEQ_LEN


def make_sequences(X: np.ndarray, seq_len: int = SEQ_LEN) -> np.ndarray:
    """(N, F) 행렬 → (N-seq_len+1, seq_len, F) 슬라이딩 윈도우."""
    if len(X) < seq_len:
        return np.empty((0, seq_len, X.shape[1]))
    return np.stack([X[i:i + seq_len] for i in range(len(X) - seq_len + 1)])


class LSTMAutoencoder:
    """시퀀스 재구성 기반 이상탐지기."""

    def __init__(self, n_features: int, seq_len: int = SEQ_LEN, latent_dim: int = AE_LATENT_DIM):
        self.n_features = n_features
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.model = None

    def build(self):
        """Keras LSTM-AE 구성(지연 임포트)."""
        import tensorflow as tf
        from tensorflow.keras import layers, models

        tf.random.set_seed(RANDOM_STATE)
        inp = layers.Input(shape=(self.seq_len, self.n_features))
        enc = layers.LSTM(self.latent_dim, activation="tanh")(inp)
        dec = layers.RepeatVector(self.seq_len)(enc)
        dec = layers.LSTM(self.latent_dim, activation="tanh", return_sequences=True)(dec)
        out = layers.TimeDistributed(layers.Dense(self.n_features))(dec)
        self.model = models.Model(inp, out)
        self.model.compile(optimizer="adam", loss="mse")
        return self.model

    def fit(self, seqs: np.ndarray):
        """정상 시퀀스로 학습."""
        if self.model is None:
            self.build()
        self.model.fit(
            seqs, seqs, epochs=AE_EPOCHS, batch_size=AE_BATCH,
            shuffle=True, verbose=0,
        )
        return self

    def reconstruction_error(self, seqs: np.ndarray) -> np.ndarray:
        """시퀀스별 평균제곱 재구성오차 → 이상 점수."""
        pred = self.model.predict(seqs, verbose=0)
        return np.mean((seqs - pred) ** 2, axis=(1, 2))

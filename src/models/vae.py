"""③ 딥러닝 이상탐지 — 시퀀스 VAE(변분 오토인코더). (소유: mfg-model)

LSTM 인코더로 잠재 분포(μ, logσ²)를 추정하고 재매개변수화 샘플링 후 LSTM 디코더로
시퀀스를 복원하는 β-VAE. lstm_ae.py와 **동일 인터페이스**(build/fit/reconstruction_error)
를 제공해 detect.py·compare.py가 모델을 교체 가능하다.

VAE의 확률적 잠재 공간은 정상 운전의 다양성을 매끄럽게 표현해, 정상 구간
재구성오차 분포를 안정화(과탐 억제)하는 효과를 기대한다.
TensorFlow 미설치 환경에서도 import가 깨지지 않도록 지연 임포트한다.
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
    VAE_BETA,
)


def _build_sampling_layer():
    """재매개변수화 트릭 샘플링 레이어(지연 임포트)."""
    import tensorflow as tf
    from tensorflow.keras import layers

    class Sampling(layers.Layer):
        """z = μ + σ·ε, ε~N(0,1). KL 발산을 손실에 추가한다."""

        def __init__(self, beta: float = VAE_BETA, **kwargs):
            super().__init__(**kwargs)
            self.beta = beta

        def call(self, inputs):
            z_mean, z_log_var = inputs
            eps = tf.random.normal(shape=tf.shape(z_mean))
            kl = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
            )
            self.add_loss(self.beta * kl)
            return z_mean + tf.exp(0.5 * z_log_var) * eps

    return Sampling


class LSTMVae:
    """시퀀스 변분 오토인코더 기반 이상탐지기."""

    name = "vae"

    def __init__(self, n_features: int, seq_len: int = SEQ_LEN, latent_dim: int = AE_LATENT_DIM):
        self.n_features = n_features
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.model = None

    def build(self):
        """Keras LSTM-VAE 구성(지연 임포트)."""
        from tensorflow.keras import layers, models

        from src.models.tf_seed import enable_determinism

        enable_determinism(RANDOM_STATE)
        Sampling = _build_sampling_layer()

        inp = layers.Input(shape=(self.seq_len, self.n_features))
        h = layers.LSTM(self.latent_dim, activation="tanh", dropout=AE_DROPOUT)(inp)
        z_mean = layers.Dense(self.latent_dim)(h)
        z_log_var = layers.Dense(self.latent_dim)(h)
        z = Sampling(beta=VAE_BETA)([z_mean, z_log_var])

        dec = layers.RepeatVector(self.seq_len)(z)
        dec = layers.LSTM(
            self.latent_dim, activation="tanh", return_sequences=True, dropout=AE_DROPOUT
        )(dec)
        out = layers.TimeDistributed(layers.Dense(self.n_features))(dec)

        self.model = models.Model(inp, out)
        self.model.compile(optimizer="adam", loss="mse")
        return self.model

    def fit(self, seqs: np.ndarray):
        """정상 시퀀스로 학습(EarlyStopping)."""
        import tensorflow as tf

        if self.model is None:
            self.build()
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

        윈도우 평균이 아니라 그 행 자신의 복원오차를 사용해 결함 직후
        회복 구간의 경계 오탐을 억제한다(lstm_ae와 동일 정책).
        """
        if len(seqs) == 0:
            return np.empty(0)
        pred = self.model.predict(seqs, verbose=0)
        return np.mean((seqs[:, -1, :] - pred[:, -1, :]) ** 2, axis=1)

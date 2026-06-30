"""③ 딥러닝 이상탐지 — Transformer 오토인코더. (소유: mfg-model)

멀티헤드 셀프어텐션 인코더로 시퀀스 전역 의존성을 포착해 재구성하는 AE.
lstm_ae.py와 **동일 인터페이스**(build/fit/reconstruction_error)를 제공한다.

LSTM 대비 장기 의존성·다변량 상호작용을 병렬로 학습하며, 본 워크플로우의
짧은 윈도우(SEQ_LEN=20)·다변량(52) 정상 패턴을 안정적으로 복원하는 것을 목표로 한다.
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
    TRANSFORMER_FF_DIM,
    TRANSFORMER_HEADS,
)


class TransformerAutoencoder:
    """멀티헤드 셀프어텐션 기반 시퀀스 재구성 이상탐지기."""

    name = "transformer_ae"

    def __init__(self, n_features: int, seq_len: int = SEQ_LEN, latent_dim: int = AE_LATENT_DIM):
        self.n_features = n_features
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.model = None

    def _encoder_block(self, x, layers):
        """프리노름 Transformer 인코더 블록(어텐션 + 피드포워드, 잔차연결)."""
        attn = layers.MultiHeadAttention(
            num_heads=TRANSFORMER_HEADS, key_dim=self.latent_dim, dropout=AE_DROPOUT
        )(x, x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn)
        ff = layers.Dense(TRANSFORMER_FF_DIM, activation="relu")(x)
        ff = layers.Dropout(AE_DROPOUT)(ff)
        ff = layers.Dense(x.shape[-1])(ff)
        return layers.LayerNormalization(epsilon=1e-6)(x + ff)

    def build(self):
        """Keras Transformer-AE 구성(지연 임포트)."""
        import tensorflow as tf
        from tensorflow.keras import layers, models

        from src.models.tf_seed import enable_determinism

        enable_determinism(RANDOM_STATE)
        inp = layers.Input(shape=(self.seq_len, self.n_features))
        # 입력 임베딩 + 학습형 위치 인코딩
        x = layers.Dense(self.latent_dim)(inp)
        pos = tf.range(start=0, limit=self.seq_len, delta=1)
        pos_emb = layers.Embedding(input_dim=self.seq_len, output_dim=self.latent_dim)(pos)
        x = x + pos_emb

        x = self._encoder_block(x, layers)
        x = self._encoder_block(x, layers)

        out = layers.TimeDistributed(layers.Dense(self.n_features))(x)
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

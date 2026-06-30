"""③ 딥러닝 이상탐지 — TensorFlow 결정성/재현성 설정. (소유: mfg-model)

모델 학습 결과가 실행마다 동일하도록 모든 난수원과 연산 결정성을 고정한다.
TF 미설치 환경에서 import가 깨지지 않도록 호출 시점에 지연 임포트한다.
각 모델의 build() 첫머리에서 호출한다.
"""

import os
import random

import numpy as np

from config.settings import RANDOM_STATE

_DETERMINISM_DONE = False


def enable_determinism(seed: int = RANDOM_STATE) -> None:
    """파이썬/넘파이/TF 난수 시드 + TF 연산 결정성을 일괄 고정(멱등)."""
    global _DETERMINISM_DONE
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)

    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    if not _DETERMINISM_DONE:
        try:
            tf.config.experimental.enable_op_determinism()
        except (AttributeError, RuntimeError):
            # 구버전 TF 또는 그래프 이미 구성됨 → 시드만으로 graceful 진행
            pass
        _DETERMINISM_DONE = True

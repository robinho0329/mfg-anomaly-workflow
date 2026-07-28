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

# 현재 활성 시드 — 기본은 RANDOM_STATE. 시드 반복 평가(repeat_eval)가 매 회차마다
# 이 값을 바꿔, 모델 초기화까지 회차별로 달라지게 한다.
_ACTIVE_SEED = RANDOM_STATE


def set_active_seed(seed: int) -> None:
    """이후 build()가 사용할 시드를 지정한다(시드 반복 평가용)."""
    global _ACTIVE_SEED
    _ACTIVE_SEED = int(seed)


def enable_determinism(seed: int | None = None) -> None:
    """파이썬/넘파이/TF 난수 시드 + TF 연산 결정성을 일괄 고정(멱등).

    seed 미지정 시 활성 시드(_ACTIVE_SEED)를 쓴다 — 호출측이 시드를 하드코딩하면
    반복 평가에서 회차마다 같은 초기화를 반복하게 되므로 기본값에 맡긴다.
    """
    global _DETERMINISM_DONE
    seed = _ACTIVE_SEED if seed is None else int(seed)
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

"""① 수집 — TEP(Tennessee Eastman Process) 유사 다변량 공정 시뮬레이터.

실제 TEP 데이터(Rieth et al. 2017, Harvard Dataverse)를 `data/raw/`에 드롭하면
그대로 사용할 수 있고, 데이터가 없을 때는 본 시뮬레이터가 동일 스키마(52변수)의
정상/결함 스트림을 합성해 '살아있는 수집' 데모를 가능하게 한다.

설계 의도(소유: mfg-collector):
- 정상 운전은 변수별 평균 주위의 상관된 가우시안 + 완만한 드리프트로 모델링.
- 결함 IDV는 특정 변수군에 스텝/드리프트/분산증가 형태의 교란을 주입.
- 재현성을 위해 seed 고정(random_state=42).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import (
    PROCESS_COLS,
    N_XMEAS,
    N_XMV,
    RANDOM_STATE,
    SAMPLE_INTERVAL_MIN,
)


class TEPSimulator:
    """TEP 유사 공정 스트림 생성기.

    Parameters
    ----------
    seed : int
        재현성 시드.
    """

    def __init__(self, seed: int = RANDOM_STATE):
        self.rng = np.random.default_rng(seed)
        n = len(PROCESS_COLS)
        # 변수별 기준 평균/표준편차 (XMEAS는 넓게, XMV는 0~100 밸브개도 가정)
        self._mu = np.concatenate([
            self.rng.uniform(0.1, 100.0, N_XMEAS),
            self.rng.uniform(20.0, 80.0, N_XMV),
        ])
        self._sigma = np.concatenate([
            self.rng.uniform(0.5, 5.0, N_XMEAS),
            self.rng.uniform(1.0, 3.0, N_XMV),
        ])
        # 변수 간 상관을 위한 공통 잠재 요인 적재행렬
        self._loading = self.rng.normal(0, 1, size=(n, 3))
        self._t = 0  # 누적 스텝(드리프트용)

    def _normal_step(self, k: int) -> np.ndarray:
        """정상 운전 k개 스텝 생성."""
        n = len(PROCESS_COLS)
        latent = self.rng.normal(0, 1, size=(k, 3))
        common = latent @ self._loading.T            # 상관 구조
        idio = self.rng.normal(0, 1, size=(k, n))    # 개별 잡음
        steps = np.arange(self._t, self._t + k)[:, None]
        drift = 0.0005 * self._sigma * np.sin(steps / 200.0)  # 완만한 운전 드리프트
        x = self._mu + self._sigma * (0.6 * common + 0.4 * idio) + drift
        self._t += k
        return x

    def _inject_fault(self, x: np.ndarray, fault_id: int) -> np.ndarray:
        """정상 스트림 x에 IDV(fault_id) 교란을 주입."""
        if fault_id <= 0:
            return x
        n = x.shape[1]
        # 결함마다 영향 변수군을 결정(시드 기반 재현)
        frng = np.random.default_rng(RANDOM_STATE + fault_id)
        affected = frng.choice(n, size=max(3, n // 8), replace=False)
        kind = fault_id % 3
        ramp = np.linspace(0, 1, x.shape[0])[:, None]
        if kind == 0:       # 스텝 변화
            x[:, affected] += 6.0 * self._sigma[affected]
        elif kind == 1:     # 드리프트
            x[:, affected] += ramp * 8.0 * self._sigma[affected]
        else:               # 분산 증가
            x[:, affected] += frng.normal(0, 4.0, x[:, affected].shape) * self._sigma[affected]
        return x

    def generate(self, n_rows: int, fault_id: int = 0, start_ts: pd.Timestamp | None = None) -> pd.DataFrame:
        """n_rows개 샘플 생성 → DataFrame(timestamp, fault_id, 52변수).

        Parameters
        ----------
        n_rows : int
            생성 행 수.
        fault_id : int
            0=정상, 1~20=해당 IDV 결함 주입.
        start_ts : pd.Timestamp | None
            시작 타임스탬프(없으면 현재시각).
        """
        x = self._normal_step(n_rows)
        x = self._inject_fault(x, fault_id)
        start_ts = start_ts or pd.Timestamp.now().floor("min")
        ts = pd.date_range(start_ts, periods=n_rows, freq=f"{SAMPLE_INTERVAL_MIN}min")
        df = pd.DataFrame(x, columns=PROCESS_COLS)
        df.insert(0, "fault_id", fault_id)
        df.insert(0, "timestamp", ts)
        return df


if __name__ == "__main__":
    # 빠른 동작 확인
    sim = TEPSimulator()
    demo = sim.generate(5, fault_id=0)
    print(demo[["timestamp", "fault_id"] + PROCESS_COLS[:4]].to_string(index=False))

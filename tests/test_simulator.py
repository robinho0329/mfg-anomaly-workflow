"""① 수집 시뮬레이터 단위 테스트. (소유: mfg-collector)"""

import numpy as np

from config.settings import PROCESS_COLS
from src.collect.simulator import TEPSimulator


def test_generate_shape():
    """생성 행 수·컬럼 스키마 검증."""
    sim = TEPSimulator(seed=42)
    df = sim.generate(50, fault_id=0)
    assert len(df) == 50
    assert all(c in df.columns for c in PROCESS_COLS)
    assert {"timestamp", "fault_id"}.issubset(df.columns)


def test_reproducible():
    """동일 시드 → 동일 결과(재현성)."""
    a = TEPSimulator(seed=42).generate(20, fault_id=0)
    b = TEPSimulator(seed=42).generate(20, fault_id=0)
    np.testing.assert_array_almost_equal(a[PROCESS_COLS].values, b[PROCESS_COLS].values)


def test_fault_shifts_distribution():
    """결함 주입 시 정상 대비 분포가 이동/확대됨."""
    sim = TEPSimulator(seed=42)
    normal = sim.generate(200, fault_id=0)
    faulty = TEPSimulator(seed=42).generate(200, fault_id=1)
    # 결함은 일부 변수의 전체 변동성을 키운다
    assert faulty[PROCESS_COLS].std().mean() > normal[PROCESS_COLS].std().mean()

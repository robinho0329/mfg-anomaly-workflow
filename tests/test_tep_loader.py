"""① 수집 — 실데이터 어댑터(tep_loader) 단위 테스트. (소유: mfg-collector)"""

import pandas as pd

from config.settings import PROCESS_COLS, XMEAS_COLS, XMV_COLS
from src.collect.tep_loader import find_tep_files, load_raw_tep


def _make_real_csv(path):
    """Rieth et al. 2017 형식 유사 CSV 생성(faultNumber + xmeas_1.. + xmv_1..)."""
    cols = {"faultNumber": [3, 3, 3]}
    cols["simulationRun"] = [1, 1, 1]  # 무시되어야 할 부가 컬럼
    for i in range(1, 42):
        cols[f"xmeas_{i}"] = [0.1 * i, 0.2 * i, 0.3 * i]
    for i in range(1, 12):
        cols[f"xmv_{i}"] = [float(i), float(i) + 1, float(i) + 2]
    pd.DataFrame(cols).to_csv(path, index=False)


def test_load_real_csv_maps_schema(tmp_path):
    """실데이터 CSV가 프로젝트 스키마(fault_id + 52변수)로 매핑된다."""
    _make_real_csv(tmp_path / "TEP_Faulty_Training.csv")
    df = load_raw_tep(tmp_path)
    assert df is not None
    assert len(df) == 3
    assert all(c in df.columns for c in PROCESS_COLS)
    assert df["fault_id"].unique().tolist() == [3]
    # zero-pad 매핑 확인(xmeas_1 → xmeas_01)
    assert df["xmeas_01"].iloc[0] == 0.1
    assert df["xmv_11"].iloc[0] == 11.0


def test_snapshot_not_treated_as_real(tmp_path):
    """시뮬레이터 스냅샷(snapshot_*)은 실데이터로 인식되지 않는다."""
    cols = {"fault_id": [0]}
    for c in PROCESS_COLS:
        cols[c] = [1.0]
    pd.DataFrame(cols).to_parquet(tmp_path / "snapshot_20260630T000000.parquet", engine="pyarrow")
    assert find_tep_files(tmp_path) == []
    assert load_raw_tep(tmp_path) is None


def test_no_data_returns_none(tmp_path):
    """실데이터가 없으면 None(graceful)."""
    assert load_raw_tep(tmp_path) is None


def test_missing_values_imputed(tmp_path):
    """결측/문자값은 숫자 변환·중앙값 보간되어 NaN이 남지 않는다."""
    _make_real_csv(tmp_path / "tep_demo.csv")
    df = pd.read_csv(tmp_path / "tep_demo.csv")
    df.loc[0, "xmeas_5"] = None
    df.to_csv(tmp_path / "tep_demo.csv", index=False)
    out = load_raw_tep(tmp_path)
    assert out is not None
    assert out[PROCESS_COLS].isna().sum().sum() == 0

"""① 수집 — 실제 TEP 데이터 드롭인 어댑터. (소유: mfg-collector)

`data/raw/` 에 실제 TEP 데이터(예: Rieth et al. 2017, Harvard Dataverse)가
존재하면 시뮬레이터 대신 그것을 프로젝트 스키마로 매핑해 적재하기 위한 모듈.

기대 입력 포맷
--------------
- CSV 또는 Parquet. 파일명은 `config.settings.TEP_REAL_GLOBS` 글롭에 매칭되어야 함
  (예: ``TEP_FaultFree_Training.csv``, ``TEP_Faulty_Training.csv``).
  시뮬레이터 자체 스냅샷(``snapshot_*``)은 실데이터로 취급하지 않는다.
- 컬럼(대소문자/언더스코어 유연 매칭):
    * 결함 라벨:  ``faultNumber`` | ``fault_id`` | ``fault`` | ``idv``  → fault_id
    * 측정 변수:  ``xmeas_1..xmeas_41`` (또는 ``xmeas1`` 등)            → xmeas_01..41
    * 조작 변수:  ``xmv_1..xmv_11``    (또는 ``xmv1`` 등)              → xmv_01..11
    * (선택) ``simulationRun``, ``sample`` 등 부가 컬럼은 무시.
- timestamp 컬럼이 있으면 사용하고, 없으면 호출측(scheduler)이 연속 부여한다.

설계 원칙
---------
- 실데이터가 없거나 매핑 가능한 컬럼이 부족하면 **None 반환(graceful degradation)**.
- 결측/문자 → 숫자 강제 변환 후 변수별 중앙값으로 보간(없으면 0).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from config.settings import (
    N_XMEAS,
    N_XMV,
    PROCESS_COLS,
    RAW_DIR,
    SNAPSHOT_PREFIX,
    TEP_REAL_GLOBS,
    XMEAS_COLS,
    XMV_COLS,
)

# 결함 라벨로 인정하는 컬럼명 후보(소문자 비교).
_FAULT_ALIASES = ("faultnumber", "fault_id", "faultid", "fault", "idv")


def find_tep_files(raw_dir: Path = RAW_DIR) -> list[Path]:
    """raw_dir 에서 실제 TEP 파일 후보를 찾는다(시뮬레이터 스냅샷 제외)."""
    if not raw_dir.exists():
        return []
    found: list[Path] = []
    for pattern in TEP_REAL_GLOBS:
        for path in raw_dir.glob(pattern):
            if path.name.startswith(SNAPSHOT_PREFIX):
                continue  # 우리 시뮬레이터 산출물은 실데이터 아님
            found.append(path)
    # 중복 제거 + 결정적 순서
    return sorted(set(found))


def _norm(col: str) -> str:
    """컬럼명 정규화 — 소문자 + 영숫자만(언더스코어 차이 흡수)."""
    return re.sub(r"[^a-z0-9]", "", col.lower())


def _build_rename_map(columns: list[str]) -> dict[str, str] | None:
    """실데이터 컬럼 → 프로젝트 스키마(xmeas_01.., xmv_01.., fault_id) 매핑 생성.

    측정/조작 변수가 충분히 매칭되지 않으면 None(어댑터 적용 불가).
    """
    norm_to_orig = {_norm(c): c for c in columns}
    rename: dict[str, str] = {}

    # fault 라벨
    for alias in _FAULT_ALIASES:
        if alias in norm_to_orig:
            rename[norm_to_orig[alias]] = "fault_id"
            break

    # xmeas_1..41, xmv_1..11 (zero-pad 차이 흡수)
    matched = 0
    for i in range(1, N_XMEAS + 1):
        key = _norm(f"xmeas_{i}")
        if key in norm_to_orig:
            rename[norm_to_orig[key]] = f"xmeas_{i:02d}"
            matched += 1
    for i in range(1, N_XMV + 1):
        key = _norm(f"xmv_{i}")
        if key in norm_to_orig:
            rename[norm_to_orig[key]] = f"xmv_{i:02d}"
            matched += 1

    # 변수 매칭이 절반 미만이면 포맷 불일치로 간주.
    if matched < (N_XMEAS + N_XMV) // 2:
        return None
    return rename


def _read_any(path: Path) -> pd.DataFrame:
    """확장자에 따라 CSV/Parquet 읽기."""
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, engine="pyarrow")
    return pd.read_csv(path)


def _coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """프로젝트 스키마(fault_id + 52변수)로 정렬·형변환·결측보간."""
    # fault_id 보강
    if "fault_id" not in df.columns:
        df["fault_id"] = 0
    df["fault_id"] = pd.to_numeric(df["fault_id"], errors="coerce").fillna(0).astype(int)

    # 누락 변수 컬럼은 0으로 생성(스키마 고정)
    for col in PROCESS_COLS:
        if col not in df.columns:
            df[col] = 0.0

    # 변수 숫자 강제 변환 + 변수별 중앙값 보간(없으면 0)
    feats = df[PROCESS_COLS].apply(pd.to_numeric, errors="coerce")
    feats = feats.fillna(feats.median(numeric_only=True)).fillna(0.0)
    df[PROCESS_COLS] = feats

    keep = (["timestamp"] if "timestamp" in df.columns else []) + ["fault_id"] + PROCESS_COLS
    return df[keep].reset_index(drop=True)


def load_raw_tep(raw_dir: Path = RAW_DIR) -> pd.DataFrame | None:
    """raw_dir 의 실제 TEP 데이터를 프로젝트 스키마 DataFrame으로 반환.

    Returns
    -------
    pd.DataFrame | None
        매핑된 데이터(컬럼: [timestamp?] fault_id, xmeas_01..41, xmv_01..11).
        실데이터가 없거나 포맷 불일치면 None.
    """
    files = find_tep_files(raw_dir)
    if not files:
        return None

    frames: list[pd.DataFrame] = []
    for path in files:
        try:
            raw = _read_any(path)
        except (OSError, ValueError, pd.errors.ParserError):
            continue  # 손상 파일은 건너뜀
        rename = _build_rename_map(list(raw.columns))
        if rename is None:
            continue  # 포맷 불일치 파일은 건너뜀
        frames.append(_coerce_schema(raw.rename(columns=rename)))

    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    return combined if len(combined) else None


if __name__ == "__main__":
    df = load_raw_tep()
    if df is None:
        print("[tep_loader] data/raw/ 에 실제 TEP 데이터 없음 → 시뮬레이터 사용")
    else:
        print(f"[tep_loader] 실데이터 {len(df)}행 로드, fault 분포:\n{df['fault_id'].value_counts().sort_index()}")

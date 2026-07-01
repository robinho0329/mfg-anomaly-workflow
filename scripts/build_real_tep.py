"""실제 TEP 벤치마크 데이터 다운로드 → 프로젝트 스키마 CSV 생성. (소유: mfg-collector)

출처: Tennessee Eastman Process 표준 시뮬레이션 데이터(Downs & Vogel 1993).
      Braatz 그룹 배포판(공개) — GitHub `camaramm/tennessee-eastman-profBraatz`.
      각 `dXX.dat`는 52변수(xmeas 41 + xmv 11) 공백구분 텍스트.
      d00=정상(52x500, 전치 필요), dXX=결함 IDV XX(480x52).

산출: `data/raw/TEP_real.csv` (faultNumber + xmeas_1..41 + xmv_1..11).
      → tep_loader가 이 파일을 실데이터로 인식·적재하고, 없으면 시뮬레이터로 폴백.

실행: python -m scripts.build_real_tep
"""
from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
COLS = [f"xmeas_{i}" for i in range(1, 42)] + [f"xmv_{i}" for i in range(1, 12)]  # 52

# 사용할 결함 IDV(스텝·랜덤·드리프트·난탐지 혼합) + 결함당 행 수
FAULTS = [1, 2, 3, 4, 6, 7, 8, 12, 14, 18]
ROWS_PER_FAULT = 48


def _fetch_dat(name: str) -> pd.DataFrame:
    """dXX.dat 원격 로드 → (samples x 52) DataFrame."""
    with urllib.request.urlopen(f"{BASE}/{name}", timeout=30) as resp:
        raw = resp.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(raw), sep=r"\s+", header=None)
    if df.shape[0] == 52:  # d00.dat 은 전치(52x500)로 배포됨
        df = df.T.reset_index(drop=True)
    assert df.shape[1] == 52, f"{name}: 예상 52열, 실제 {df.shape}"
    df.columns = COLS
    return df


def build() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    blocks: list[pd.DataFrame] = []

    # 정상: d00(500) + d00_te(960)
    for name in ("d00.dat", "d00_te.dat"):
        b = _fetch_dat(name)
        b["faultNumber"] = 0
        blocks.append(b)

    # 결함: dXX.dat 앞부분 슬라이스
    for x in FAULTS:
        b = _fetch_dat(f"d{x:02d}.dat").iloc[:ROWS_PER_FAULT].copy()
        b["faultNumber"] = x
        blocks.append(b)

    real = pd.concat(blocks, ignore_index=True)[["faultNumber"] + COLS]
    out = RAW_DIR / "TEP_real.csv"
    real.to_csv(out, index=False)

    n_norm = int((real["faultNumber"] == 0).sum())
    n_fault = int((real["faultNumber"] != 0).sum())
    print(f"생성: {out} ({real.shape[0]}행, {out.stat().st_size/1e6:.2f}MB)")
    print(f"정상 {n_norm} / 결함 {n_fault} ({n_norm / n_fault:.1f}:1), IDV={FAULTS}")
    return out


if __name__ == "__main__":
    build()

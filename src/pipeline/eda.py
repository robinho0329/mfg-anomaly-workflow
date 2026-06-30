"""② EDA·통계분석 — 정제 데이터의 탐색적 분석 + 통계 검정. (소유: mfg-eda)

산출(data/eda/): 분포/상관 히트맵 PNG, 정상 vs 결함 t검정·KS검정 결과 JSON.
대시보드(④)와 PPT가 이 산출물을 그대로 임베드한다.
"""

import json

import pandas as pd
from scipy import stats

from config.settings import EDA_DIR, PROCESS_COLS
from src.pipeline.preprocess import load_stream, clean


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """변수별 기술통계(평균/표준편차/왜도/첨도)."""
    desc = df[PROCESS_COLS].describe().T
    desc["skew"] = df[PROCESS_COLS].skew()
    desc["kurtosis"] = df[PROCESS_COLS].kurtosis()
    return desc


def fault_tests(df: pd.DataFrame) -> dict:
    """정상(fault_id==0) vs 결함 그룹 간 변수별 t검정·KS검정.

    결함 데이터가 없으면 빈 dict 반환(graceful degradation).
    """
    normal = df[df["fault_id"] == 0]
    faulty = df[df["fault_id"] != 0]
    if normal.empty or faulty.empty:
        return {}
    results = {}
    for c in PROCESS_COLS:
        t_stat, t_p = stats.ttest_ind(normal[c], faulty[c], equal_var=False)
        ks_stat, ks_p = stats.ks_2samp(normal[c], faulty[c])
        results[c] = {
            "t_stat": float(t_stat), "t_p": float(t_p),
            "ks_stat": float(ks_stat), "ks_p": float(ks_p),
        }
    return results


def run() -> dict:
    """EDA 파이프라인 실행 → 통계 JSON 저장."""
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    df = clean(load_stream())
    if df.empty:
        print("[eda] 데이터 없음 — 수집(collect) 먼저 실행 필요")
        return {}
    summary = summary_stats(df)
    summary.to_parquet(EDA_DIR / "summary_stats.parquet", engine="pyarrow")
    tests = fault_tests(df)
    (EDA_DIR / "fault_tests.json").write_text(
        json.dumps(tests, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[eda] 기술통계 {len(summary)}변수 + 검정 {len(tests)}변수 → {EDA_DIR}")
    return {"summary": summary, "tests": tests}


if __name__ == "__main__":
    run()

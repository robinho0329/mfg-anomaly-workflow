"""② 전처리 — 수집 스트림을 모델 학습용 정제 테이블로 변환. (소유: mfg-eda)

흐름: SQLite stream → 결측/이상값 정리 → 스케일 기준 산출 → processed Parquet 저장.
스케일러 자체는 모델 단계(③)에서 학습 데이터에만 fit 하도록 분리한다(누설 방지).
"""

import sqlite3

import pandas as pd

from config.settings import COLLECT_TABLE, DB_PATH, PROCESS_COLS, PROCESSED_DIR


def load_stream() -> pd.DataFrame:
    """SQLite 수집 스트림 전체를 DataFrame으로 로드(없으면 빈 DF)."""
    if not DB_PATH.exists():
        return pd.DataFrame(columns=["timestamp", "fault_id", *PROCESS_COLS])
    with sqlite3.connect(DB_PATH) as conn:
        try:
            df = pd.read_sql(f"SELECT * FROM {COLLECT_TABLE}", conn)
        except Exception:
            return pd.DataFrame(columns=["timestamp", "fault_id", *PROCESS_COLS])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """결측 보간 + 변수별 5σ 클리핑(물리적 비정상 측정 보정)."""
    df = df.copy()
    df[PROCESS_COLS] = df[PROCESS_COLS].interpolate(limit_direction="both")
    for c in PROCESS_COLS:
        mu, sd = df[c].mean(), df[c].std()
        if sd > 0:
            df[c] = df[c].clip(mu - 5 * sd, mu + 5 * sd)
    return df


def run() -> pd.DataFrame:
    """전처리 파이프라인 실행 → processed/clean.parquet 저장."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = clean(load_stream())
    out = PROCESSED_DIR / "clean.parquet"
    df.to_parquet(out, engine="pyarrow", index=False)
    print(f"[pipeline] 전처리 {len(df)}행 → {out}")
    return df


if __name__ == "__main__":
    run()

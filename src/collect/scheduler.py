"""① 수집 — 스트림 적재 스케줄러.

시뮬레이터에서 배치를 생성해 SQLite(stream.db)에 누적하고,
주기 호출(cron/Task Scheduler)로 '살아있는 수집'을 구현한다. (소유: mfg-collector)

사용:
    python -m src.collect.scheduler --rows 20 --fault 0
"""

from __future__ import annotations

import argparse
import sqlite3

import pandas as pd

from config.settings import COLLECT_BATCH_ROWS, COLLECT_TABLE, DB_PATH, RAW_DIR
from src.collect.simulator import TEPSimulator


def _last_timestamp(conn: sqlite3.Connection) -> pd.Timestamp | None:
    """기존 적재의 마지막 timestamp를 반환(없으면 None)."""
    try:
        row = conn.execute(f"SELECT MAX(timestamp) FROM {COLLECT_TABLE}").fetchone()
        return pd.Timestamp(row[0]) if row and row[0] else None
    except sqlite3.OperationalError:
        return None  # 테이블 미존재(최초 실행)


def collect_batch(rows: int = COLLECT_BATCH_ROWS, fault_id: int = 0) -> int:
    """배치 1회 수집 → SQLite 적재 + raw Parquet 스냅샷. 적재 행 수 반환."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        last_ts = _last_timestamp(conn)
        start = (last_ts + pd.Timedelta(minutes=3)) if last_ts is not None else None
        sim = TEPSimulator()
        df = sim.generate(rows, fault_id=fault_id, start_ts=start)
        # SQLite는 timestamp를 문자열로 저장
        out = df.copy()
        out["timestamp"] = out["timestamp"].astype(str)
        out.to_sql(COLLECT_TABLE, conn, if_exists="append", index=False)

    # raw 스냅샷(감사/재현용)
    snap = RAW_DIR / f"snapshot_{df['timestamp'].iloc[0]:%Y%m%dT%H%M%S}.parquet"
    df.to_parquet(snap, engine="pyarrow", index=False)
    print(f"[collect] {len(df)}행 적재 → {COLLECT_TABLE} (fault_id={fault_id}), 스냅샷={snap.name}")
    return len(df)


def main() -> None:
    parser = argparse.ArgumentParser(description="TEP 스트림 수집 스케줄러")
    parser.add_argument("--rows", type=int, default=COLLECT_BATCH_ROWS, help="배치 행 수")
    parser.add_argument("--fault", type=int, default=0, help="주입 결함 IDV(0=정상)")
    args = parser.parse_args()
    collect_batch(rows=args.rows, fault_id=args.fault)


if __name__ == "__main__":
    main()

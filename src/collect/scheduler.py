"""① 수집 — 스트림 적재 스케줄러.

시뮬레이터(또는 raw 실데이터)에서 배치를 생성해 SQLite(stream.db)에 누적하고,
주기 호출(Task Scheduler/cron)로 '살아있는 수집'을 구현한다. (소유: mfg-collector)

분기 규칙:
    data/raw/ 에 실제 TEP 데이터가 있으면 그것을 우선 적재(tep_loader),
    없으면 TEP 시뮬레이터로 합성 스트림 생성.

사용:
    python -m src.collect.scheduler --rows 20 --fault 0      # 1회 배치
    python -m src.collect.scheduler --backfill                # 결정적 백필(DB 확대)

OS 스케줄러 상시화 방법은 ``src/collect/README.md`` 참고.
"""

from __future__ import annotations

import argparse
import sqlite3

import pandas as pd

from config.settings import (
    BACKFILL_PLAN,
    COLLECT_BATCH_ROWS,
    COLLECT_META_TABLE,
    COLLECT_TABLE,
    DB_PATH,
    RANDOM_STATE,
    RAW_DIR,
    SAMPLE_INTERVAL_MIN,
)
from src.collect.simulator import TEPSimulator
from src.collect.tep_loader import load_raw_tep


def _last_timestamp(conn: sqlite3.Connection) -> pd.Timestamp | None:
    """기존 적재의 마지막 timestamp를 반환(없으면 None)."""
    try:
        row = conn.execute(f"SELECT MAX(timestamp) FROM {COLLECT_TABLE}").fetchone()
        return pd.Timestamp(row[0]) if row and row[0] else None
    except sqlite3.OperationalError:
        return None  # 테이블 미존재(최초 실행)


def _meta_get(conn: sqlite3.Connection, key: str, default: int = 0) -> int:
    """보조 메타 테이블에서 정수값 조회(없으면 default)."""
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {COLLECT_META_TABLE} (key TEXT PRIMARY KEY, value INTEGER)"
    )
    row = conn.execute(
        f"SELECT value FROM {COLLECT_META_TABLE} WHERE key = ?", (key,)
    ).fetchone()
    return int(row[0]) if row else default


def _meta_set(conn: sqlite3.Connection, key: str, value: int) -> None:
    """보조 메타 테이블에 정수값 upsert."""
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {COLLECT_META_TABLE} (key TEXT PRIMARY KEY, value INTEGER)"
    )
    conn.execute(
        f"INSERT INTO {COLLECT_META_TABLE} (key, value) VALUES (?, ?) "
        f"ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def _next_real_batch(conn: sqlite3.Connection, rows: int) -> pd.DataFrame | None:
    """실데이터에서 다음 rows개 슬라이스를 오프셋 기반으로 반환(소진 시 None)."""
    real = load_raw_tep(RAW_DIR)
    if real is None or len(real) == 0:
        return None
    offset = _meta_get(conn, "real_offset", 0)
    if offset >= len(real):
        return None  # 실데이터 모두 소진
    chunk = real.iloc[offset:offset + rows].copy()
    _meta_set(conn, "real_offset", offset + len(chunk))
    return chunk.reset_index(drop=True)


def collect_batch(rows: int = COLLECT_BATCH_ROWS, fault_id: int = 0, seed: int = RANDOM_STATE) -> int:
    """배치 1회 수집 → SQLite 적재 + raw Parquet 스냅샷. 적재 행 수 반환.

    실데이터(data/raw/)가 있으면 우선 적재하고, 없으면 시뮬레이터로 생성한다.
    timestamp는 기존 적재 끝에서 연속 부여(누락/중복 방지).
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        last_ts = _last_timestamp(conn)
        start = (last_ts + pd.Timedelta(minutes=SAMPLE_INTERVAL_MIN)) if last_ts is not None else None

        # 1) 실데이터 우선
        df = _next_real_batch(conn, rows)
        source = "real"
        if df is None:
            # 2) 시뮬레이터 폴백 (배치별 시드로 다양성 확보, 결정적)
            df = TEPSimulator(seed=seed).generate(rows, fault_id=fault_id, start_ts=start)
            source = "sim"

        # timestamp 연속 부여(실데이터/시뮬 공통) — 누락·중복 방지
        start_ts = start if start is not None else pd.Timestamp.now().floor("min")
        df = df.copy()
        df["timestamp"] = pd.date_range(start_ts, periods=len(df), freq=f"{SAMPLE_INTERVAL_MIN}min")
        # 컬럼 순서 보장: timestamp, fault_id, 52변수
        ordered = ["timestamp", "fault_id"] + [c for c in df.columns if c not in ("timestamp", "fault_id")]
        df = df[ordered]

        out = df.copy()
        out["timestamp"] = out["timestamp"].astype(str)  # SQLite 문자열 저장
        out.to_sql(COLLECT_TABLE, conn, if_exists="append", index=False)

    # raw 스냅샷(감사/재현용)
    snap = RAW_DIR / f"snapshot_{df['timestamp'].iloc[0]:%Y%m%dT%H%M%S}.parquet"
    df.to_parquet(snap, engine="pyarrow", index=False)
    fids = sorted(df["fault_id"].unique().tolist())
    print(f"[collect:{source}] {len(df)}행 적재 → {COLLECT_TABLE} (fault_id={fids}), 스냅샷={snap.name}")
    return len(df)


def backfill() -> int:
    """BACKFILL_PLAN을 따라 결정적으로 DB를 확대한다. 누적 적재 행 수 반환.

    정상:결함 ≈ 3.5:1, 신규 IDV 다양 주입. 배치별 시드는 인덱스로 파생해 재현 가능.
    """
    total = 0
    for idx, (rows, fault_id) in enumerate(BACKFILL_PLAN):
        total += collect_batch(rows=rows, fault_id=fault_id, seed=RANDOM_STATE + idx)
    print(f"[backfill] 계획 {len(BACKFILL_PLAN)}배치 완료 — 총 {total}행 추가")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="TEP 스트림 수집 스케줄러")
    parser.add_argument("--rows", type=int, default=COLLECT_BATCH_ROWS, help="배치 행 수")
    parser.add_argument("--fault", type=int, default=0, help="주입 결함 IDV(0=정상)")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE, help="시뮬레이터 시드(다양성/재현성)")
    parser.add_argument("--backfill", action="store_true", help="BACKFILL_PLAN으로 DB 결정적 확대")
    args = parser.parse_args()
    if args.backfill:
        backfill()
    else:
        collect_batch(rows=args.rows, fault_id=args.fault, seed=args.seed)


if __name__ == "__main__":
    main()

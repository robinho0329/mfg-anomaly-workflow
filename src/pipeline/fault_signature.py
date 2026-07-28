"""② 결함 시그니처 — 결함모드별로 '어느 태그가 움직이는지'를 데이터에서 귀속. (소유: mfg-eda)

현장 검수팀 반려 기준 ①("태그명 없이 주요 변수라고 쓰면 반려")에 대한 답이다.
결함모드별 감시 태그를 문헌에서 인용해 적으면 절대규칙 ⑥(확인 안 된 정보를
사실처럼 쓰지 않는다)에 걸리므로, **측정 가능한 것만** 산출한다:

  각 결함 구간의 태그값이 정상 구간 대비 몇 표준편차 이동했는가.

설비 귀속(xmeas_21이 어느 설비인가)은 여기서 주장하지 않는다 — TEP 문헌 대조가
필요한 별도 항목이며 '미측정'으로 남긴다.

산출: data/eda/fault_signature.json
실행: python -m src.pipeline.fault_signature
"""

import json
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import EDA_DIR, PROCESS_COLS, PROCESSED_DIR  # noqa: E402

TOP_N = 5          # 결함모드당 보고할 상위 태그 수
MIN_SHIFT = 1.0    # 이 이하 이동은 '뚜렷한 신호 없음'으로 본다(정상 변동 범위)


def _load_clean() -> pd.DataFrame:
    path = PROCESSED_DIR / "clean.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def signatures(df: pd.DataFrame, top_n: int = TOP_N) -> dict:
    """결함모드별 태그 이동량(정상 대비 표준화) 상위 N개."""
    if df.empty or "fault_id" not in df.columns:
        return {}
    normal = df[df["fault_id"] == 0]
    if len(normal) < 2:
        return {}
    mu, sd = normal[PROCESS_COLS].mean(), normal[PROCESS_COLS].std()
    sd = sd.replace(0, pd.NA)  # 상수 태그는 이동량 정의 불가

    out: dict = {}
    for fid in sorted(int(f) for f in df["fault_id"].unique() if f != 0):
        seg = df[df["fault_id"] == fid]
        shift = ((seg[PROCESS_COLS].mean() - mu) / sd).dropna()
        if shift.empty:
            continue
        top = shift.reindex(shift.abs().sort_values(ascending=False).index)[:top_n]
        out[str(fid)] = {
            "n_rows": int(len(seg)),
            "max_abs_shift": float(top.abs().max()),
            "distinct": bool(top.abs().max() >= MIN_SHIFT),
            "tags": [
                {"tag": t, "shift_sigma": round(float(v), 2), "direction": "+" if v > 0 else "−"}
                for t, v in top.items()
            ],
        }
    return out


def run() -> dict:
    """결함 시그니처 산출 → data/eda/fault_signature.json."""
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    df = _load_clean()
    sig = signatures(df)
    payload = {
        "top_n": TOP_N,
        "min_shift_sigma": MIN_SHIFT,
        "note": (
            "정상 구간 대비 표준화 평균 이동량(σ). 설비 귀속은 미측정 — "
            "TEP 문헌 대조가 필요하며 본 산출물은 태그 수준까지만 주장한다."
        ),
        "signatures": sig,
    }
    out = EDA_DIR / "fault_signature.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[fault_signature] 결함모드 {len(sig)}종 → {out}")
    for fid, s in sig.items():
        tags = " · ".join(f"{t['tag']}{t['direction']}{abs(t['shift_sigma']):.1f}σ" for t in s["tags"][:3])
        mark = "" if s["distinct"] else "  ⚠️ 뚜렷한 이동 없음"
        print(f"  IDV {fid:>2}: {tags}{mark}")
    return payload


if __name__ == "__main__":
    run()

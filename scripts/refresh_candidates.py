"""과제 후보 즉시 갱신 — 스캔 → 오피스로 복사 → (선택) 커밋·푸시.

깃허브 Actions 가 매일 아침 같은 일을 하지만, 지금 바로 반영하고 싶을 때 쓴다.

실행:
    python scripts/refresh_candidates.py           # 스캔 + 복사
    python scripts/refresh_candidates.py --push    # 복사 후 오피스 저장소에 커밋·푸시
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.report.discover import run as discover_run  # noqa: E402

OFFICE = _ROOT.parent / "ai-company-office"
SRC = _ROOT / "data" / "discovery" / "candidates.json"


def main() -> int:
    ap = argparse.ArgumentParser(description="과제 후보 갱신 후 오피스로 복사")
    ap.add_argument("--push", action="store_true", help="오피스 저장소에 커밋·푸시까지")
    ap.add_argument("--no-remote", action="store_true", help="gh 조회 생략(빠름)")
    args = ap.parse_args()

    result = discover_run(use_remote=not args.no_remote)
    if not result:
        print("[refresh] 후보를 만들지 못했습니다")
        return 1

    if not OFFICE.exists():
        print(f"[refresh] 오피스 저장소가 없습니다: {OFFICE}")
        return 1

    dst = OFFICE / "candidates.json"
    before = dst.read_text(encoding="utf-8") if dst.exists() else None
    shutil.copy(SRC, dst)
    changed = before != dst.read_text(encoding="utf-8")
    print(f"[refresh] 복사 → {dst}  ({'변경됨' if changed else '변화 없음'})")

    if not args.push:
        return 0
    if not changed:
        print("[refresh] 변화가 없어 커밋하지 않습니다")
        return 0

    n_proj = len(result.get("projects", []))
    msg = f"chore: 과제 후보 갱신 — 저장소 {n_proj}곳"
    for cmd in (["git", "add", "candidates.json"],
                ["git", "commit", "-m", msg],
                ["git", "push"]):
        out = subprocess.run(cmd, cwd=OFFICE, capture_output=True, text=True)
        if out.returncode != 0:
            print(f"[refresh] 실패: {' '.join(cmd)}\n{out.stderr.strip()}")
            return 1
    print("[refresh] 오피스 저장소에 푸시 완료")
    return 0


if __name__ == "__main__":
    sys.exit(main())

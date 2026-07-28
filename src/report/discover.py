"""⓪ 과제발굴 — 산출물에서 다음 과제 후보를 추린다. (소유: mfg-reporter)

사규 §3-0 과제발굴팀의 반려 기준을 코드로 옮긴 것이다.

  ① 근거를 산출물의 인용 문장 + 파일 경로로 대지 못하면 반려
  ② 보유 데이터로 답할 수 있는지 태그 단위로 판정하지 않았으면 반려
  ③ 후보가 4개 이상이면 반려 — 대표가 고를 수 있는 수로 줄인다

**질문을 발명하지 않는다.** 후보는 두 곳에서만 나온다:
  · 직전 리포트가 스스로 '미측정·미실행·이관'이라 적은 문장 (모드 B — 기존 디벨롭)
  · 반복 평가가 드러낸 완전 미탐·취약 결함모드 (모드 A — 신규 발굴)
근거를 못 대는 후보는 만들지 않는다. 후보가 0개면 0개로 보고한다.

산출: data/discovery/candidates.json  (오피스 화면이 이 파일을 읽는다)
실행: python -m src.report.discover
"""

import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import DATA_DIR, EDA_DIR, MODELS_DIR, REPORTS_DIR  # noqa: E402

KST = timezone(timedelta(hours=9))
MAX_CANDIDATES = 3           # 사규 §3-0 반려 기준 ③
WEAK_RECALL = 0.8
DISCOVERY_DIR = DATA_DIR / "discovery"

# 이월 표지 — 리포트가 스스로 '아직 안 했다'고 적은 문장만 잡는다
CARRYOVER = re.compile(r"미측정|미실행|미연동|이관한다|이관 대상")


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(_PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def latest_report() -> Path | None:
    """가장 최근 데일리 리포트. 없으면 None."""
    reports = sorted(REPORTS_DIR.glob("daily_*.md"))
    return reports[-1] if reports else None


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def carryover_lines(report: Path) -> list[dict]:
    """리포트에서 이월 문장을 인용으로 추출 — 모드 B 후보의 근거."""
    try:
        lines = report.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for i, raw in enumerate(lines, 1):
        line = raw.strip().lstrip("-· ").strip()
        if not line or not CARRYOVER.search(line):
            continue
        if line.startswith(("|", "<!--", "_이 표")):
            continue
        out.append({"line_no": i, "quote": line[:160]})
    return out


def carryover_subject(quote: str) -> str:
    """이월 문장에서 '무엇이' 미측정/이관인지 주어부만 뽑는다.

    이월 표지는 대개 문장 끝에 온다("…온라인 운영 시 동일 성능은 미측정.").
    앞머리를 자르면 엉뚱한 제목이 나오므로, 표지가 든 절을 고르고 표지 앞을 쓴다.
    """
    clean = re.sub(r"[*`]", "", quote).strip()
    parts = [p for p in re.split(r"[.·—]", clean) if p.strip()]
    seg = next((p for p in parts if CARRYOVER.search(p)), clean).strip()
    m = re.search(r"(.+?)\s*(?:은|는|이|가)?\s*(?:미측정|미실행|미연동|이관)", seg)
    subject = (m.group(1) if m else seg).strip(" ,()")
    subject = re.sub(r"^(다만|단|또한|그리고)\s+", "", subject)
    return subject[:34]


def weak_modes(model: str | None = None) -> list[dict]:
    """반복 평가에서 완전 미탐·취약으로 드러난 결함모드 — 모드 A 후보의 근거."""
    rep = _load_json(MODELS_DIR / "repeat_eval.json")
    cov = rep.get("coverage") or {}
    if not cov:
        return []
    model = model or next(iter(cov))
    sig = (_load_json(EDA_DIR / "fault_signature.json") or {}).get("signatures", {})
    out = []
    for fid, c in sorted(cov.get(model, {}).items(), key=lambda kv: int(kv[0])):
        if c.get("n_zero", 0) == 0 and c.get("recall_mean", 1.0) >= WEAK_RECALL:
            continue
        tags = [t["tag"] for t in (sig.get(str(fid), {}).get("tags") or [])[:2]]
        out.append({
            "fault_id": int(fid),
            "n_seeds": c["n_seeds"],
            "n_zero": c.get("n_zero", 0),
            "recall_mean": c["recall_mean"],
            "tags": tags,                      # 반려 기준 ② — 태그 단위 판정
            "max_shift": (sig.get(str(fid), {}) or {}).get("max_abs_shift"),
        })
    return out


def build_candidates(report: Path) -> dict:
    """모드별 후보를 만든다. 근거 없는 후보는 만들지 않는다."""
    src_report = _rel(report)
    rep_src = _rel(MODELS_DIR / "repeat_eval.json")

    # ── 모드 A: 취약 결함모드 → 감시 수단 설계 과제 ──
    mode_a = []
    for m in sorted(weak_modes(), key=lambda x: (x["recall_mean"], -x["n_zero"]))[:MAX_CANDIDATES]:
        miss = f"{m['n_seeds']}회 평가 중 {m['n_zero']}회 완전 미탐" if m["n_zero"] else \
               f"평균 재현율 {m['recall_mean']:.2f}"
        shift = f" · 최대 이동량 {m['max_shift']:.1f}σ" if m.get("max_shift") is not None else ""
        mode_a.append({
            "label": f"IDV {m['fault_id']} 대체 감시 수단 설계",
            "note": f"근거: {miss}(평균 {m['recall_mean']:.2f}){shift}",
            "question": f"IDV {m['fault_id']}을 재구성오차 외 어떤 신호로 감시할 수 있는가",
            "tags": " · ".join(m["tags"]) if m["tags"] else "태그 미판정",
            "source": rep_src,
        })

    # ── 모드 B: 리포트가 스스로 적은 이월 문장 → 개선 과제 ──
    mode_b, seen = [], set()
    for c in carryover_lines(report):
        subject = carryover_subject(c["quote"])
        if not subject or subject in seen:
            continue
        seen.add(subject)
        mode_b.append({
            "label": f"{subject} 해소",
            "note": f'근거: "{c["quote"][:90]}…"',
            "question": "이 항목을 처리하면 어느 수치가 얼마나 바뀌는가",
            "tags": "이월 대장",
            "source": f"{src_report}:{c['line_no']}",
        })
        if len(mode_b) >= MAX_CANDIDATES:
            break

    return {
        "generated": datetime.now(KST).isoformat(timespec="seconds"),
        "source_report": src_report,
        "max_candidates": MAX_CANDIDATES,
        "note": (
            "산출물에서 추출한 후보다. 질문을 발명하지 않으며, 근거 문장과 파일 경로가 "
            "없는 후보는 만들지 않는다. 비용비·결정권자 등 사람이 정할 값은 비워 둔다."
        ),
        "blanks_for_owner": [
            "오탐 1건 : 미탐 1건 비용비",
            "이 분석으로 무엇을 결정하는가 + 결정권자",
            "대상 태그의 조작 권한자",
        ],
        "A": mode_a,
        "B": mode_b,
    }


def run() -> dict:
    """후보 산출 → data/discovery/candidates.json."""
    report = latest_report()
    if report is None:
        print("[discover] 리포트가 없어 후보를 만들 수 없습니다 — 먼저 daily_report 실행")
        return {}

    result = build_candidates(report)
    DISCOVERY_DIR.mkdir(parents=True, exist_ok=True)
    out = DISCOVERY_DIR / "candidates.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[discover] 근거 리포트: {result['source_report']}")
    for mode in ("A", "B"):
        items = result[mode]
        print(f"  모드 {mode}: 후보 {len(items)}개")
        for c in items:
            print(f"    · {c['label']}  ({c['source']})")
        if not items:
            print("      근거를 찾지 못했습니다 — 후보를 지어내지 않습니다")
    print(f"[discover] 저장 → {out}")
    return result


if __name__ == "__main__":
    run()

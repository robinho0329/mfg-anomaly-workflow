"""⓪ 과제발굴 — 보유한 모든 깃허브 저장소에서 다음 과제 후보를 추린다. (소유: mfg-reporter)

사규 §3-0 과제발굴팀의 반려 기준을 코드로 옮긴 것이다.

  ① 근거를 산출물의 인용 문장 + 파일 경로로 대지 못하면 반려
  ② 보유 데이터로 답할 수 있는지 태그 단위로 판정하지 않았으면 반려
  ③ 후보가 4개 이상이면 반려 — 대표가 고를 수 있는 수로 줄인다

**질문을 발명하지 않는다.** 후보는 관측된 사실에서만 나온다:
  · 리포트·문서가 스스로 '미측정·미실행·이관'이라 적은 문장
  · 반복 평가가 드러낸 완전 미탐·취약 결함모드 (그 형식을 가진 저장소만)
  · 코드에 남은 TODO/FIXME, 열린 이슈, 실패한 CI, 오래 멈춘 저장소
근거를 못 대는 후보는 만들지 않는다. 후보가 0개면 0개로 보고한다.

산출: data/discovery/candidates.json  (오피스 화면이 이 파일을 읽는다)
실행:
    python -m src.report.discover                 # 전체 저장소 스캔
    python -m src.report.discover --repo NAME     # 한 저장소만
    python -m src.report.discover --no-remote     # gh 호출 없이 로컬만
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import DATA_DIR, EDA_DIR, MODELS_DIR, REPORTS_DIR  # noqa: E402

KST = timezone(timedelta(hours=9))
MAX_CANDIDATES = 3            # 사규 §3-0 반려 기준 ③
WEAK_RECALL = 0.8
STALE_DAYS = 30               # 이 기간 커밋이 없으면 '멈춘 저장소'로 본다
DISCOVERY_DIR = DATA_DIR / "discovery"
WORKSPACE = _PROJECT_ROOT.parent

CARRYOVER = re.compile(r"미측정|미실행|미연동|이관한다|이관 대상")
# 표지가 '주장'이 아니라 '인용·규칙'인 경우 — 사규의 "'미측정'이라고 쓴다" 같은 문장.
# 이걸 거르지 않으면 규칙 문서가 통째로 이월 항목이 된다.
RULE_LIKE = re.compile(
    r"['\"‘“]\s*(?:미측정|미실행|미연동)|"           # 따옴표로 인용된 표지
    r"(?:미측정|미실행|미연동)\s*['\"’”]|"
    r"(?:이라고|라고|으로|로)\s*(?:쓴|적|표시|기록|남긴|보고)|"  # "~라고 쓴다"
    r"(?:쓴다|적는다|표시한다|기록한다|남긴다)"
)
TODO_MARK = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b")
CODE_EXT = {".py", ".js", ".ts", ".ipynb", ".sql", ".sh", ".yml", ".yaml"}
SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache", "data"}

CARRYOVER_VERB = [
    (re.compile(r"미측정"), "측정"), (re.compile(r"미실행"), "실행"),
    (re.compile(r"미연동"), "연동"), (re.compile(r"이관"), "착수"),
]


# ── 공통 유틸 ─────────────────────────────────────────────
def _git(repo: Path, *args, timeout: int = 15) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=repo, capture_output=True,
                             text=True, timeout=timeout)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _gh_json(*args, timeout: int = 40):
    try:
        out = subprocess.run(["gh", *args], capture_output=True, text=True, timeout=timeout)
        return json.loads(out.stdout) if out.returncode == 0 and out.stdout.strip() else None
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return None


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def find_repos(root: Path | None = None) -> list[dict]:
    """지정 폴더(기본: 워크스페이스)에서 origin 리모트를 가진 저장소를 모두 찾는다.

    CI 러너는 저장소들을 임시 폴더에 clone 하므로 root 를 바꿔 부른다.
    """
    root = root or WORKSPACE
    repos = []
    if not root.exists():
        return repos
    for path in sorted(p for p in root.iterdir() if p.is_dir()):
        if not (path / ".git").exists():
            continue
        remote = _git(path, "remote", "get-url", "origin")
        if not remote:
            continue
        remote = remote.rstrip("/").removesuffix(".git")
        repos.append({
            "name": remote.split("/")[-1],
            "path": path,
            "remote": remote,
            "commit": _git(path, "rev-parse", "--short", "HEAD"),
            "last_commit": _git(path, "log", "-1", "--format=%cI"),
        })
    return repos


# ── 근거 수집 ─────────────────────────────────────────────
def carryover_subject(quote: str) -> str:
    """이월 문장에서 '무엇이' 미측정/이관인지 주어부만 뽑는다(표지는 문장 끝에 온다)."""
    clean = re.sub(r"[*`'\"‘’“”]", "", quote).strip()
    parts = [p for p in re.split(r"[.·—:]", clean) if p.strip()]
    seg = next((p for p in parts if CARRYOVER.search(p)), clean).strip()
    m = re.search(r"(.+?)\s*(?:은|는|이|가)?\s*(?:미측정|미실행|미연동|이관)", seg)
    subject = (m.group(1) if m else seg).strip(" ,()")
    subject = re.sub(r"^(다만|단|또한|그리고|그러나)\s+", "", subject)
    # 조각난 주어(따옴표 잔해·조사만 남은 것)는 버린다 — 라벨이 뜻을 잃는다
    return subject[:34] if len(subject) >= 4 else ""


def carryover_verb(quote: str) -> str:
    for pat, verb in CARRYOVER_VERB:
        if pat.search(quote):
            return verb
    return "처리"


def scan_markdown(repo: Path) -> list[dict]:
    """저장소의 md 문서에서 이월 문장을 인용으로 수집."""
    out = []
    for md in sorted(repo.rglob("*.md")):
        if any(part in SKIP_DIRS for part in md.parts):
            continue
        try:
            lines = md.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        rel = md.relative_to(repo).as_posix()
        in_fence = False
        for i, raw in enumerate(lines, 1):
            if raw.lstrip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:                       # 코드블록 = 규칙·예시. 이월이 아니다
                continue
            line = raw.strip().lstrip("-·> ").strip()
            if not line or line.startswith(("|", "<!--")) or not CARRYOVER.search(line):
                continue
            if RULE_LIKE.search(line):         # "'미측정'이라고 쓴다" 류 제외
                continue
            out.append({"file": rel, "line_no": i, "quote": line[:160]})
    return out


def scan_todos(repo: Path, limit: int = 40) -> list[dict]:
    """코드에 남은 TODO/FIXME. 주석이 아니어도 잡히지만 근거로는 충분하다."""
    out = []
    for src in sorted(repo.rglob("*")):
        if len(out) >= limit:
            break
        if src.suffix not in CODE_EXT or not src.is_file():
            continue
        if any(part in SKIP_DIRS for part in src.parts):
            continue
        try:
            lines = src.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        rel = src.relative_to(repo).as_posix()
        for i, raw in enumerate(lines, 1):
            if TODO_MARK.search(raw):
                out.append({"file": rel, "line_no": i, "quote": raw.strip()[:140]})
                break          # 파일당 1건 — 같은 파일이 후보를 독식하지 않게
    return out


def weak_modes(repo: Path) -> list[dict]:
    """반복 평가 형식을 가진 저장소에서만 — 완전 미탐·취약 결함모드."""
    if repo != _PROJECT_ROOT:
        return []
    rep = _load_json(MODELS_DIR / "repeat_eval.json")
    cov = rep.get("coverage") or {}
    if not cov:
        return []
    model = next(iter(cov))
    sig = (_load_json(EDA_DIR / "fault_signature.json") or {}).get("signatures", {})
    out = []
    for fid, c in sorted(cov.get(model, {}).items(), key=lambda kv: int(kv[0])):
        if c.get("n_zero", 0) == 0 and c.get("recall_mean", 1.0) >= WEAK_RECALL:
            continue
        out.append({
            "fault_id": int(fid), "n_seeds": c["n_seeds"], "n_zero": c.get("n_zero", 0),
            "recall_mean": c["recall_mean"],
            "tags": [t["tag"] for t in (sig.get(str(fid), {}).get("tags") or [])[:2]],
            "max_shift": (sig.get(str(fid), {}) or {}).get("max_abs_shift"),
        })
    return out


def remote_signals(name: str) -> dict:
    """열린 이슈·최근 CI 결과. gh 가 없거나 실패하면 빈 dict."""
    sig = {}
    issues = _gh_json("issue", "list", "--repo", name, "--state", "open",
                      "--limit", "5", "--json", "number,title")
    if issues:
        sig["issues"] = issues
    runs = _gh_json("run", "list", "--repo", name, "--limit", "3",
                    "--json", "conclusion,displayTitle,workflowName")
    if runs:
        failed = [r for r in runs if r.get("conclusion") == "failure"]
        if failed:
            sig["failed_run"] = failed[0]
    return sig


# ── 후보 조립 ─────────────────────────────────────────────
def candidates_for(repo: dict, use_remote: bool) -> dict:
    """저장소 1개의 후보. 모드 A(신규 발굴) / B(기존 디벨롭)로 나눈다."""
    path, name = repo["path"], repo["name"]
    mode_a, mode_b = [], []

    # 모드 A — 아직 답 못한 질문(취약 결함모드 · 열린 이슈 · 실패한 CI · 멈춘 저장소)
    for m in sorted(weak_modes(path), key=lambda x: (x["recall_mean"], -x["n_zero"]))[:MAX_CANDIDATES]:
        miss = (f"{m['n_seeds']}회 평가 중 {m['n_zero']}회 완전 미탐"
                if m["n_zero"] else f"평균 재현율 {m['recall_mean']:.2f}")
        shift = f" · 최대 이동량 {m['max_shift']:.1f}σ" if m.get("max_shift") is not None else ""
        tags = " · ".join(m["tags"]) or "미판정"
        mode_a.append({
            "label": f"IDV {m['fault_id']} 대체 감시 수단 설계",
            "note": f"근거: {miss}(평균 {m['recall_mean']:.2f}){shift}",
            "question": f"IDV {m['fault_id']}을 재구성오차 외 어떤 신호로 감시할 수 있는가",
            "tags": tags, "source": "data/models/repeat_eval.json",
            "action": f"IDV {m['fault_id']}을 관리도·인터록·순회점검 중 무엇으로 덮을지 정하고, "
                      f"감시 태그({tags})의 관리 한계를 설계한다",
        })

    remote = remote_signals(name) if use_remote else {}
    for iss in (remote.get("issues") or [])[:MAX_CANDIDATES - len(mode_a)]:
        mode_a.append({
            "label": f"이슈 #{iss['number']} 처리",
            "note": f"근거: 열린 이슈 \"{iss['title'][:70]}\"",
            "question": "이 이슈를 닫으려면 무엇을 바꿔야 하는가",
            "tags": "GitHub 이슈", "source": f"{repo['remote']}/issues/{iss['number']}",
            "action": f"이슈 #{iss['number']} 를 해결하고 닫는다",
        })

    if repo.get("last_commit") and len(mode_a) < MAX_CANDIDATES:
        try:
            last = datetime.fromisoformat(repo["last_commit"])
            days = (datetime.now(last.tzinfo) - last).days
        except ValueError:
            days = None
        if days is not None and days >= STALE_DAYS:
            mode_a.append({
                "label": f"{name} 재개 여부 결정",
                "note": f"근거: 마지막 커밋 {last:%Y-%m-%d} — {days}일 멈춤",
                "question": "이 저장소를 계속 갈 것인가, 아카이브할 것인가",
                "tags": "저장소 상태", "source": f"git log -1 ({repo['commit']})",
                "action": "재개한다면 다음 과제 1개를 정하고, 아니면 README에 중단 사유를 남긴다",
            })

    # 모드 B — 이미 적어둔 이월 항목(문서 미측정/이관 · TODO · 실패 CI)
    seen = set()
    for c in scan_markdown(path):
        subject = carryover_subject(c["quote"])
        if not subject or subject in seen:
            continue
        seen.add(subject)
        verb = carryover_verb(c["quote"])
        mode_b.append({
            "label": f"{subject} {verb}",
            "note": f'근거: "{c["quote"][:90]}…"',
            "question": "이 항목을 처리하면 어느 수치가 얼마나 바뀌는가",
            "tags": "이월 대장", "source": f"{c['file']}:{c['line_no']}",
            "action": f"{c['file']} {c['line_no']}행이 '{verb} 필요'로 남긴 항목을 처리하고, "
                      f"처리 후 그 문장을 문서에서 지울 수 있는 상태로 만든다",
        })
        if len(mode_b) >= MAX_CANDIDATES:
            break

    if remote.get("failed_run") and len(mode_b) < MAX_CANDIDATES:
        fr = remote["failed_run"]
        mode_b.append({
            "label": f"{fr.get('workflowName','CI')} 실패 복구",
            "note": f"근거: 최근 실행 실패 — \"{(fr.get('displayTitle') or '')[:60]}\"",
            "question": "무엇 때문에 실패했고 무엇을 고쳐야 통과하는가",
            "tags": "CI", "source": f"{repo['remote']}/actions",
            "action": "실패 로그를 읽고 원인을 고친 뒤 재실행해 통과를 확인한다",
        })

    for c in scan_todos(path)[:MAX_CANDIDATES - len(mode_b)]:
        mode_b.append({
            "label": f"{c['file'].split('/')[-1]} TODO 정리",
            "note": f'근거: "{c["quote"][:80]}"',
            "question": "이 TODO를 처리하면 무엇이 좋아지는가",
            "tags": "코드 주석", "source": f"{c['file']}:{c['line_no']}",
            "action": f"{c['file']} {c['line_no']}행의 TODO를 처리하거나, 불필요하면 근거와 함께 지운다",
        })

    return {
        "project": {"name": name, "remote": repo["remote"], "commit": repo["commit"],
                    "last_commit": repo.get("last_commit")},
        "A": mode_a[:MAX_CANDIDATES],
        "B": mode_b[:MAX_CANDIDATES],
    }


def run(only: str | None = None, use_remote: bool = True,
        root: Path | None = None, out_path: Path | None = None) -> dict:
    """전체 저장소 스캔 → candidates.json."""
    repos = find_repos(root)
    if only:
        repos = [r for r in repos if r["name"] == only]
    if not repos:
        print(f"[discover] 대상 저장소가 없습니다{f' (--repo {only})' if only else ''}")
        return {}

    projects = []
    for r in repos:
        c = candidates_for(r, use_remote)
        if c["A"] or c["B"]:
            projects.append(c)

    # 화면 기본값 — 후보가 가장 많은 저장소를 먼저 보여준다
    projects.sort(key=lambda p: len(p["A"]) + len(p["B"]), reverse=True)
    default = projects[0] if projects else {"project": None, "A": [], "B": []}

    result = {
        "generated": datetime.now(KST).isoformat(timespec="seconds"),
        "scanned_repos": [r["name"] for r in repos],
        "note": ("보유 저장소 전체를 스캔해 뽑은 후보다. 질문을 발명하지 않으며, "
                 "근거 문장과 파일 경로가 없는 후보는 만들지 않는다. "
                 "비용비·결정권자 등 사람이 정할 값은 비워 둔다."),
        "blanks_for_owner": ["오탐 1건 : 미탐 1건 비용비",
                             "이 분석으로 무엇을 결정하는가 + 결정권자",
                             "대상 태그의 조작 권한자"],
        "projects": projects,
        # 하위호환 — 오피스가 아직 단일 프로젝트 키를 읽는다
        "project": default["project"], "A": default["A"], "B": default["B"],
        "source_report": (default["project"] or {}).get("name", "-"),
    }

    out = out_path or (DISCOVERY_DIR / "candidates.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[discover] 저장소 {len(repos)}개 스캔 · 후보가 있는 저장소 {len(projects)}개")
    for p in projects:
        print(f"  ── {p['project']['name']}  (A {len(p['A'])} / B {len(p['B'])})")
        for mode in ("A", "B"):
            for c in p[mode]:
                print(f"      [{mode}] {c['label']}  ← {c['source']}")
    if not projects:
        print("  후보를 찾지 못했습니다 — 지어내지 않습니다")
    print(f"[discover] 저장 → {out}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="보유 저장소 전체에서 과제 후보 발굴")
    ap.add_argument("--repo", default=None, help="특정 저장소만 스캔")
    ap.add_argument("--no-remote", action="store_true", help="gh 호출 생략(이슈·CI 미조회)")
    ap.add_argument("--root", default=None, help="저장소들이 모인 폴더(기본: 워크스페이스)")
    ap.add_argument("--out", default=None, help="산출 파일 경로")
    a = ap.parse_args()
    run(only=a.repo, use_remote=not a.no_remote,
        root=Path(a.root) if a.root else None,
        out_path=Path(a.out) if a.out else None)


if __name__ == "__main__":
    main()

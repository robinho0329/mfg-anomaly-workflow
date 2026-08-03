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

from config.settings import DATA_DIR  # noqa: E402

KST = timezone(timedelta(hours=9))
MAX_CANDIDATES = 3            # 사규 §3-0 반려 기준 ③
WEAK_RECALL = 0.8
STALE_DAYS = 30               # 이 기간 커밋이 없으면 '멈춘 저장소'로 본다
# 대상에서 빼는 저장소. 대표가 "디벨롭 한계"로 판단해 제외한 것 —
# 코드가 임의로 정한 게 아니므로 여기 적어 근거를 남긴다.
EXCLUDE_REPOS = {"mfg-anomaly-workflow"}
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
    subject = re.sub(r"^(다만|단|또한|그리고|그러나)\s+", "", subject).rstrip(" ,·")
    # 주제 조사 뒤는 목적구다("임계값 재보정은 다음 사이클 과제로" → "임계값 재보정").
    # 그대로 두면 라벨이 "…과제로 착수" 처럼 문장이 안 닫힌다.
    topic = re.split(r"(?:은|는)\s+", subject, maxsplit=1)[0].strip()
    if len(topic) >= 4:
        subject = topic
    # 부정어가 주어 끝에 걸리면 라벨에서 잘려 뜻이 뒤집힌다.
    # "…전부 미실행" 의 주어를 "…전부" 로 자르면 실행된 것처럼 읽힌다 — 만들지 않는다.
    if re.search(r"(전부|모두|아직|여전히)$", subject) or subject.endswith(("미", "불", "안", "못")):
        return ""
    # 조각난 주어(따옴표 잔해·조사만 남은 것)는 버린다 — 라벨이 뜻을 잃는다
    return subject[:34] if len(subject) >= 4 else ""


# 숫자 읽기의 끝소리로 목적격 조사를 고른다(12=십이→를, 3=삼→을).
_DIGIT_TAIL = {"0": "을", "1": "을", "3": "을", "6": "을", "7": "을", "8": "을",
               "2": "를", "4": "를", "5": "를", "9": "를"}


def _eul(n: int) -> str:
    """받침 여부에 맞는 목적격 조사."""
    s = str(n)
    if len(s) > 1 and s.endswith("0"):   # 10, 20 → 십, 이십 (ㅂ 받침)
        return "을"
    return _DIGIT_TAIL.get(s[-1], "을")


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


def _notebook_todos(path: Path, rel: str) -> list[dict]:
    """노트북은 JSON이라 원문 줄 번호가 사람에게 무의미하고, 출력 셀의 base64
    이미지에 우연히 TODO 문자열이 섞여 가짜 후보가 된다. 소스 셀만 본다."""
    try:
        nb = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except (json.JSONDecodeError, OSError):
        return []
    out = []
    for idx, cell in enumerate(nb.get("cells", []), 1):
        if cell.get("cell_type") not in ("code", "markdown"):
            continue
        src = "".join(cell.get("source", []))
        for line in src.splitlines():
            if TODO_MARK.search(line) and not _looks_like_blob(line):
                out.append({"file": rel, "line_no": f"셀 {idx}", "quote": line.strip()[:140]})
                break
    return out


def _looks_like_blob(line: str) -> bool:
    """base64·해시 같은 긴 무의미 문자열인가. 근거로 인용할 수 없다."""
    s = line.strip()
    if len(s) > 200:
        return True
    longest = max((len(w) for w in s.split()), default=0)
    return longest > 60


def scan_todos(repo: Path, limit: int = 40) -> list[dict]:
    """코드에 남은 TODO/FIXME. 인용할 수 없는 줄(바이너리·base64)은 제외한다."""
    out = []
    for src in sorted(repo.rglob("*")):
        if len(out) >= limit:
            break
        if src.suffix not in CODE_EXT or not src.is_file():
            continue
        if any(part in SKIP_DIRS for part in src.parts):
            continue
        rel = src.relative_to(repo).as_posix()
        if src.suffix == ".ipynb":
            out.extend(_notebook_todos(src, rel)[:1])   # 파일당 1건
            continue
        try:
            lines = src.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for i, raw in enumerate(lines, 1):
            if TODO_MARK.search(raw) and not _looks_like_blob(raw):
                out.append({"file": rel, "line_no": i, "quote": raw.strip()[:140]})
                break          # 파일당 1건 — 같은 파일이 후보를 독식하지 않게
    return out


def weak_modes(repo: Path) -> list[dict]:
    """반복 평가 산출물을 가진 저장소에서 — 완전 미탐·취약 결함모드.

    경로를 저장소 기준으로 읽는다. 전역 상수(MODELS_DIR)를 쓰면 CI 러너처럼
    다른 위치에 clone 한 저장소를 스캔할 때 자기 자신만 보게 된다.
    """
    rep = _load_json(repo / "data" / "models" / "repeat_eval.json")
    cov = rep.get("coverage") or {}
    if not cov:
        return []
    # 운영에 쓰는 모델의 수치여야 한다. next(iter()) 는 딕셔너리 첫 키를 집어
    # 쓰지 않는 모델의 미탐률로 현장 인터록을 제안하게 만든다.
    model = _operating_model(repo, cov)
    if model is None:
        return []          # 운영 모델을 특정할 수 없으면 후보를 만들지 않는다
    sig = (_load_json(repo / "data" / "eda" / "fault_signature.json") or {}).get("signatures", {})
    out = []
    for fid, c in sorted(cov.get(model, {}).items(), key=lambda kv: int(kv[0])):
        # 점추정만 보면 n_seeds 가 4~11 로 제각각인 모드를 같은 자로 재게 된다.
        # 표준오차를 얹은 상한이 기준 아래일 때만 취약으로 본다.
        n, mean = max(c.get("n_seeds", 1), 1), c.get("recall_mean", 1.0)
        upper = mean + c.get("recall_std", 0.0) / n ** 0.5
        if c.get("n_zero", 0) == 0 and upper >= WEAK_RECALL:
            continue
        out.append({
            "model": model,
            "fault_id": int(fid), "n_seeds": c["n_seeds"], "n_zero": c.get("n_zero", 0),
            "recall_mean": c["recall_mean"], "recall_std": c.get("recall_std", 0.0),
            "tags": [t["tag"] for t in (sig.get(str(fid), {}).get("tags") or [])[:2]],
            "max_shift": (sig.get(str(fid), {}) or {}).get("max_abs_shift"),
        })
    return out


def _operating_model(repo: Path, cov: dict) -> str | None:
    """운영에 쓰는 모델명. comparison.json 의 split·overall 과 대조해 고른다.

    ① comparison.json 에 기록된 모델 중 coverage 에도 있는 것
    ② 그중 리포트가 기본으로 쓰는 모델(F1 최고가 아니라 실제 탐지에 쓰인 것)
    특정할 수 없으면 None — 임의로 고르지 않는다.
    """
    comp = _load_json(repo / "data" / "models" / "comparison.json")
    scores = repo / "data" / "models" / "scores.parquet"
    overall = {m.get("model") for m in comp.get("overall", []) if m.get("model")}
    shared = [m for m in cov if m in overall]
    if not shared:
        return None
    if len(shared) == 1:
        return shared[0]
    # 운영 스코어링(scores.parquet)을 만든 모델을 기본으로 본다. 그 정보가
    # 산출물에 없으면 config 를 읽고, 그것도 없으면 특정 불가로 둔다.
    settings = repo / "config" / "settings.py"
    if settings.exists() and scores.exists():
        try:
            for line in settings.read_text(encoding="utf-8").splitlines():
                if line.startswith("DEFAULT_MODEL"):
                    name = line.split("=", 1)[1].strip().strip('"\'')
                    if name in shared:
                        return name
        except OSError:
            pass
    return None


ABS_PATH = re.compile(r"[A-Za-z]:[\\/]{1,2}Users[\\/]{1,2}")
SRC_EXT = {".py", ".ipynb"}


def repo_langs(repo: Path) -> set[str]:
    """저장소가 실제로 담고 있는 언어. 없는 언어의 관행을 요구하지 않기 위해 쓴다."""
    langs = set()
    for src in repo.rglob("*"):
        if not src.is_file() or any(p in SKIP_DIRS for p in src.parts):
            continue
        if src.suffix == ".py":
            langs.add("py")
        elif src.suffix == ".ipynb":
            langs.add("py")
        elif src.suffix in (".js", ".ts", ".jsx", ".tsx"):
            langs.add("js")
        if {"py", "js"} <= langs:
            break
    return langs


def scan_health(repo: Path) -> list[dict]:
    """파일의 존재·부재로 판정하는 근거. 문서 어휘에 의존하지 않는다.

    각 항목은 경로를 인용으로 대므로 사규 §3-0 반려 기준 ①을 충족한다.
    없는 것을 "없다"고만 적고, 왜 필요한지는 대표가 판단한다.
    """
    found = []
    langs = repo_langs(repo)

    # ① 절대경로 하드코딩 — 남이 클론하면 그 줄에서 멈춘다
    hits = []
    for src in repo.rglob("*"):
        if len(hits) >= 3:
            break
        if src.suffix not in SRC_EXT or not src.is_file():
            continue
        if any(part in SKIP_DIRS for part in src.parts):
            continue
        try:
            text = src.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if ABS_PATH.search(line) and not _looks_like_blob(line):
                hits.append({"file": src.relative_to(repo).as_posix(),
                             "line_no": i, "quote": line.strip()[:120]})
                break
    if hits:
        found.append({
            "key": "abs_path",
            "label": "절대경로 하드코딩 제거",
            "note": f'근거: {len(hits)}개 파일에 로컬 절대경로 — 예 "{hits[0]["quote"][:70]}"',
            "question": "이 저장소를 다른 PC에서 클론하면 어디서 멈추는가",
            "source": f'{hits[0]["file"]}:{hits[0]["line_no"]}',
            "action": "절대경로를 pathlib 기준 상대경로로 바꾸고, 클론 직후 실행되는지 확인한다",
            "blanks": [],
        })

    # ② 회귀 테스트 부재 — 고쳐도 안 깨졌는지 확인할 방법이 없다.
    #    실행 코드가 없는 저장소(문서·정적 페이지)에는 해당하지 않는다.
    if langs and not any((repo / d).is_dir() for d in ("tests", "test")):
        found.append({
            "key": "no_tests",
            "label": "회귀 테스트 도입",
            "note": "근거: tests/ 디렉토리 없음 — 변경이 기존 동작을 깼는지 확인할 수단이 없다",
            "question": "어느 함수부터 테스트로 고정할 것인가",
            "source": "tests/ (없음)",
            "action": "핵심 함수 3개에 pytest 를 붙이고 CI 에서 돌게 한다",
            "blanks": ["먼저 고정할 함수 3개"],
        })

    # ③ CI 부재 — 깨진 상태로 푸시돼도 아무도 모른다.
    #    검증할 코드가 없으면 CI 도 확인할 것이 없다.
    if langs and not (repo / ".github" / "workflows").is_dir():
        found.append({
            "key": "no_ci",
            "label": "CI 워크플로우 추가",
            "note": "근거: .github/workflows 없음 — 푸시가 무엇도 검증하지 않는다",
            "question": "푸시마다 무엇을 자동으로 확인할 것인가",
            "source": ".github/workflows (없음)",
            "action": "임포트 확인·테스트 실행 최소 CI 를 추가하고 배지를 README 에 붙인다",
            "blanks": [],
        })

    # ④ 의존성 명세 부재 — 재현 불가. 언어별로 파일 이름이 다르고,
    #    해당 언어가 없으면 요구 자체가 성립하지 않는다.
    dep_files = {
        "py": ("requirements.txt", "pyproject.toml", "environment.yml", "Pipfile"),
        "js": ("package.json",),
    }
    missing_dep = [
        names[0] for lang, names in dep_files.items()
        if lang in langs and not any((repo / f).exists() for f in names)
    ]
    if missing_dep:
        found.append({
            "key": "no_deps",
            "label": "의존성 명세 작성",
            "note": f"근거: {' · '.join(missing_dep)} 없음 — 환경을 재현할 수 없다",
            "question": "이 저장소를 돌리려면 무엇이 필요한가",
            "source": f"{missing_dep[0]} (없음)",
            "action": "실제 임포트를 근거로 의존성을 고정 버전으로 적는다",
            "blanks": [],
        })

    # ⑤ README 에 실행 방법 없음 — 남이 못 돌린다
    readme = next((repo / n for n in ("README.md", "readme.md") if (repo / n).exists()), None)
    if readme:
        try:
            body = readme.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            body = ""
        if not re.search(r"```|실행|사용법|Usage|python\s+\w", body):
            found.append({
                "key": "no_howto",
                "label": "README 실행 방법 추가",
                "note": f"근거: {readme.name} 에 실행 예시(코드블록·python 명령)가 없다",
                "question": "클론한 사람이 첫 명령으로 무엇을 쳐야 하는가",
                "source": readme.name,
                "action": "클론 → 설치 → 실행 세 줄을 코드블록으로 적고 직접 따라해 확인한다",
                "blanks": [],
            })
    else:
        found.append({
            "key": "no_readme",
            "label": "README 작성",
            "note": "근거: README.md 없음 — 저장소를 열어도 무엇인지 알 수 없다",
            "question": "이 저장소는 무엇이고 무엇을 보여주는가",
            "source": "README.md (없음)",
            "action": "한 줄 정체성·데이터·결과·실행 방법을 적는다",
            "blanks": [],
        })
    return found


def remote_signals(remote: str) -> dict:
    """열린 이슈·최근 CI 결과.

    gh 는 OWNER/REPO 형식을 요구한다. 저장소 이름만 넘기면 전 호출이 실패하고,
    빈 dict 를 돌려주면 "이슈 0건"과 구별되지 않는다 — 실패를 상태로 남긴다.
    """
    slug = "/".join(remote.rstrip("/").split("/")[-2:]) if remote else ""
    if not slug or "/" not in slug:
        return {"status": "미연동", "why": f"remote 형식 불명: {remote!r}"}
    sig: dict = {"status": "연동", "slug": slug}
    issues = _gh_json("issue", "list", "--repo", slug, "--state", "open",
                      "--limit", "5", "--json", "number,title")
    if issues is None:
        sig["status"] = "미연동"
        sig["why"] = "gh issue list 실패(인증·권한·네트워크)"
        return sig
    sig["issues"] = issues
    runs = _gh_json("run", "list", "--repo", slug, "--limit", "3",
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
                if m["n_zero"] else f"{m['n_seeds']}회 평가")
        shift = f" · 최대 이동량 {m['max_shift']:.1f}σ" if m.get("max_shift") is not None else ""
        band = f"±{m.get('recall_std', 0.0):.2f}"
        tags = " · ".join(m["tags"]) or "미판정"
        kind = "대체 감시 수단 설계" if m["n_zero"] else "탐지율 개선"
        mode_a.append({
            "label": f"IDV {m['fault_id']} {kind}",
            # 순위는 비용비를 모른 채 매긴 것이다 — 그 사실을 카드에 남긴다
            "note": (f"근거: {miss} · 평균 재현율 {m['recall_mean']:.2f}{band}"
                     f"({m['n_seeds']}회, {m['model']}){shift}"
                     " · 순위는 오탐:미탐 비용비를 가정하지 않은 나열이다"),
            "question": f"IDV {m['fault_id']}{_eul(m['fault_id'])} 재구성오차 외 "
                        f"어떤 신호로 감시할 수 있는가",
            "tags": tags, "source": "data/models/repeat_eval.json",
            "action": f"IDV {m['fault_id']}{_eul(m['fault_id'])} 관리도·인터록·순회점검 중 "
                      f"무엇으로 덮을지 정하고, 감시 태그({tags})의 관리 한계를 설계한다",
            # 조작 권한자는 조작변수(xmv)가 걸린 과제에만 해당한다. 측정변수(xmeas)만이면
            # 감시 주기·1차 조치자를 묻는 게 맞다 — 안 쓰는 빈칸은 판단을 흐린다.
            "blanks": (["오탐 1건 : 미탐 1건 비용비", "결정권자"]
                       + (["대상 태그의 조작 권한자", "변경폭과 롤백 조건"]
                          if any(x.startswith("xmv") for x in m["tags"])
                          else ["감시 주기", "경보 1차 조치자"])),
        })

    remote = remote_signals(repo["remote"]) if use_remote else {}
    for iss in (remote.get("issues") or [])[:MAX_CANDIDATES - len(mode_a)]:
        mode_a.append({
            "label": f"이슈 #{iss['number']} 처리",
            "note": f"근거: 열린 이슈 \"{iss['title'][:70]}\"",
            "question": "이 이슈를 닫으려면 무엇을 바꿔야 하는가",
            "tags": "GitHub 이슈", "source": f"{repo['remote']}/issues/{iss['number']}",
            "action": f"이슈 #{iss['number']} 를 해결하고 닫는다", "blanks": [],
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
                "blanks": ["재개 / 아카이브 중 무엇인가"],
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
            # 원문에 없는 문자열을 따옴표로 인용하지 않는다(절대규칙 ③).
            "action": f"{c['file']}:{c['line_no']} 가 {verb} 대상으로 남겨둔 항목을 처리하고, "
                      f"처리 후 그 문장을 문서에서 지울 수 있는 상태로 만든다",
            "blanks": ["측정 기준 — 어느 지표가 얼마나 바뀌면 해소로 보는가"],
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
            "action": "실패 로그를 읽고 원인을 고친 뒤 재실행해 통과를 확인한다", "blanks": [],
        })

    for h in scan_health(path):
        if len(mode_b) >= MAX_CANDIDATES:
            break
        mode_b.append({k: v for k, v in h.items() if k != "key"} | {"tags": "저장소 위생"})

    for c in scan_todos(path)[:max(MAX_CANDIDATES - len(mode_b), 0)]:
        mode_b.append({
            "label": f"{c['file'].split('/')[-1]} TODO 정리",
            "note": f'근거: "{c["quote"][:80]}"',
            "question": "이 TODO를 처리하면 무엇이 좋아지는가",
            "tags": "코드 주석", "source": f"{c['file']}:{c['line_no']}",
            "action": f"{c['file']} {c['line_no']}행의 TODO를 처리하거나, 불필요하면 근거와 함께 지운다", "blanks": [],
        })

    return {
        "project": {"name": name, "remote": repo["remote"], "commit": repo["commit"],
                    "last_commit": repo.get("last_commit"),
                    # 받는 사람이 바로 받을 수 있게 — 이름·커밋만으로는 못 움직인다
                    "clone": f"git clone {repo['remote']} && git checkout {repo['commit']}"},
        "A": mode_a[:MAX_CANDIDATES],
        "B": mode_b[:MAX_CANDIDATES],
    }


def run(only: str | None = None, use_remote: bool = True,
        root: Path | None = None, out_path: Path | None = None) -> dict:
    """전체 저장소 스캔 → candidates.json."""
    repos = find_repos(root)
    if only:
        repos = [r for r in repos if r["name"] == only]
    else:
        repos = [r for r in repos if r["name"] not in EXCLUDE_REPOS]
    if not repos:
        print(f"[discover] 대상 저장소가 없습니다{f' (--repo {only})' if only else ''}")
        return {}

    projects = []
    for r in repos:
        c = candidates_for(r, use_remote)
        if c["A"] or c["B"]:
            projects.append(c)

    # 저장소 정렬은 고정(이름순). 후보 수로 정렬하면 사규대로 '미측정'을 성실히
    # 적는 저장소가 항상 1순위로 환류되는 자기강화 루프가 생긴다.
    projects.sort(key=lambda p: (p["project"] or {}).get("name", ""))
    default = projects[0] if projects else {"project": None, "A": [], "B": []}

    result = {
        "generated": datetime.now(KST).isoformat(timespec="seconds"),
        "scanned_repos": [r["name"] for r in repos],
        # 후보 0개의 뜻을 구별한다 — 부채가 없는 것과 볼 파일이 없는 것은 다르다
        "unavailable": {
            r["name"]: [
                p for p in ("data/models/repeat_eval.json", "data/eda/fault_signature.json")
                if not (r["path"] / p).exists()
            ]
            for r in repos
        },
        "note": ("보유 저장소 전체를 스캔해 뽑은 후보다. 질문을 발명하지 않으며, "
                 "근거 문장과 파일 경로가 없는 후보는 만들지 않는다. "
                 "비용비·결정권자 등 사람이 정할 값은 비워 둔다."),
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

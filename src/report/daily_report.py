"""④ 데일리 현황 리포트 — 워크플로우 산출물을 모아 Markdown으로 요약. (소유: mfg-reporter)

Claude API 호출 없이, 기존 산출물 로드 기반(무거운 학습/재계산 없음)으로 다음을 수집:
  1. 회귀 테스트(pytest) 통과 여부
  2. 수집 통계(총행수·정상/결함·시각범위)
  3. 최신 탐지 요약(이상 건수·비율·기본 모델 성능)
  4. 마일스톤 진행률(데이터 규모 게이트)

리포트는 '산출물을 읽어서 요약'하므로, 리포트 생성 이후 산출물이 다시 만들어지면
리포트의 수치는 조용히 낡는다. 이를 막기 위해 근거 산출물의 지문(수정시각·해시)을
리포트에 기록하고, 나중에 대조해 낡음을 드러낸다.

산출: reports/daily_YYYYMMDD.md (+ 콘솔 출력)
실행:
    python -m src.report.daily_report            # 리포트 생성
    python -m src.report.daily_report --check    # 기존 리포트 낡음 감사(생성 안 함)
"""

import argparse
import hashlib
import json
import re
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

# ── 프로젝트 루트를 import 경로에 추가(스크립트 직접 실행 대비) ──
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import (  # noqa: E402
    COLLECT_TABLE,
    DB_PATH,
    DEFAULT_MODEL,
    EDA_DIR,
    MODELS_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
    SAMPLE_INTERVAL_MIN,
    SCORE_SMOOTH_WINDOW,
)

KST = timezone(timedelta(hours=9))
MODEL_LABELS = {"lstm_ae": "LSTM-AE", "vae": "VAE", "transformer_ae": "Transformer-AE"}

# 저탐 판정 기준 — 재현율이 이 값 미만이면 '탐지 취약'으로 본다.
WEAK_RECALL = 0.8

# 데이터 규모 마일스톤 (게이트) — (target_rows, 설명)
# 설명에 수치를 박아두면 데이터가 늘어나도 문구가 안 따라오므로 넣지 않는다.
# '안정화/다양성' 류 표현은 사규 §2 금칙어라 게이트 명칭으로 쓴다.
MILESTONES = [
    (300, "딥러닝 AE 학습 게이트(정상 시퀀스 확보)"),
    (885, "현재 비교 평가 기준"),
    (2000, "결함 IDV 확장 평가 게이트"),
    (5000, "운영급 데이터 규모"),
]

# ── 낡음 판정 ─────────────────────────────────────────────
# 리포트가 근거로 삼는 산출물. 이 중 하나라도 리포트 생성 이후 바뀌면
# 리포트의 수치는 더 이상 현재 상태가 아니다.
SOURCE_ARTIFACTS = (
    MODELS_DIR / "comparison.json",
    MODELS_DIR / "scores.parquet",
    PROCESSED_DIR / "clean.parquet",
    EDA_DIR / "fault_tests.json",
    EDA_DIR / "summary_stats.parquet",
    DB_PATH,
)

PROVENANCE_BEGIN = "<!-- provenance:begin"
PROVENANCE_END = "provenance:end -->"


def _rel(path: Path) -> str:
    """프로젝트 루트 기준 상대 경로(구분자는 / 로 통일)."""
    try:
        return path.resolve().relative_to(_PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def fingerprint(path: Path) -> dict:
    """산출물 1개의 지문 — 존재여부·수정시각·크기·내용 해시(앞 12자)."""
    if not path.exists():
        return {"exists": False}
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        stat = path.stat()
    except OSError as exc:
        return {"exists": True, "error": str(exc)}
    return {
        "exists": True,
        "mtime": datetime.fromtimestamp(stat.st_mtime, KST).isoformat(timespec="seconds"),
        "bytes": stat.st_size,
        "sha256_12": digest,
    }


def collect_provenance() -> dict:
    """근거 산출물 전체의 지문 맵 {상대경로: 지문}."""
    return {_rel(p): fingerprint(p) for p in SOURCE_ARTIFACTS}


def read_recorded_provenance(report_path: Path) -> dict | None:
    """기존 리포트에 기록된 지문 블록을 읽는다. 없거나 깨졌으면 None."""
    if not report_path.exists():
        return None
    try:
        text = report_path.read_text(encoding="utf-8")
    except OSError:
        return None
    pattern = re.escape(PROVENANCE_BEGIN) + r"\s*(\{.*?\})\s*" + re.escape(PROVENANCE_END)
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def diff_provenance(recorded: dict, current: dict) -> tuple[list[str], list[str]]:
    """기록된 지문 대비 현재 산출물 비교 → (검증불가, 낡음).

    검증불가: 산출물이 지금 없어 대조 자체가 불가능한 경우. 예를 들어 CI 러너는
        커밋되지 않은 산출물(.gitignore 대상)을 볼 수 없다 — 이건 '낡음'이 아니라
        '판단 보류'다. 수치가 틀렸다는 근거가 없으므로 게이트를 막지 않는다.
    낡음: 산출물이 실제로 존재하는데 내용 해시가 기록과 다른 경우. 리포트 수치가
        더 이상 현재 상태가 아니라는 확정 근거이므로 게이트를 막는다.
    """
    unverifiable: list[str] = []
    stale: list[str] = []
    for rel, now in current.items():
        was = recorded.get(rel)
        if was is None:
            continue  # 추적 대상이 나중에 추가된 경우 — 낡음 근거로 삼지 않는다
        if was.get("exists") and not now.get("exists"):
            unverifiable.append(f"`{rel}` — 지금 없어 대조 불가(미커밋 산출물일 수 있음)")
        elif not was.get("exists") and now.get("exists"):
            stale.append(f"`{rel}` — 새로 생성됨 ({now.get('mtime')})")
        elif was.get("sha256_12") != now.get("sha256_12"):
            stale.append(
                f"`{rel}` — 내용 변경 ({was.get('mtime')} → {now.get('mtime')})"
            )
    return unverifiable, stale


def check_report(report_path: Path, current: dict) -> tuple[list[str], list[str]]:
    """리포트 1개 검사 → (경고, 오류).

    경고: 지문 기록이 없어 판정 불가(가드 도입 이전) — CI 게이트를 막지 않는다.
    오류: 지문이 현재 산출물과 불일치 — 실제로 낡은 리포트, 게이트 차단 대상.
    """
    recorded = read_recorded_provenance(report_path)
    if recorded is None:
        return (["지문 기록이 없어 대조 불가 — 가드 도입 이전 리포트(게이트 미적용)"], [])
    return diff_provenance(recorded, current)


def audit_reports(current: dict) -> dict[Path, tuple[list[str], list[str]]]:
    """reports/ 의 모든 데일리 리포트를 감사 → {경로: (경고, 오류)} (문제 있는 것만)."""
    findings: dict[Path, tuple[list[str], list[str]]] = {}
    for path in sorted(REPORTS_DIR.glob("daily_*.md")):
        warns, errors = check_report(path, current)
        if warns or errors:
            findings[path] = (warns, errors)
    return findings


# ── 산출물 요약 ───────────────────────────────────────────
def run_pytest() -> tuple[bool, str]:
    """pytest 실행 → (성공여부, 마지막 출력 라인)."""
    py = _PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    py = py if py.exists() else Path(sys.executable)
    try:
        result = subprocess.run(
            [str(py), "-m", "pytest", "-q", "--no-header"],
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=600,
        )
        last = (result.stdout.strip().splitlines() or ["(no output)"])[-1]
        return result.returncode == 0, last
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return False, f"pytest 실행 실패: {exc}"


def collect_stats() -> dict:
    """stream.db 수집 통계(총행수·정상/결함·시각범위). 없으면 빈 통계."""
    if not DB_PATH.exists():
        return {"total": 0}
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql(f"SELECT timestamp, fault_id FROM {COLLECT_TABLE}", conn)
    except (pd.errors.DatabaseError, sqlite3.OperationalError):
        return {"total": 0}
    if df.empty:
        return {"total": 0}
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    n_normal = int((df["fault_id"] == 0).sum())
    fault_modes = sorted(int(f) for f in df.loc[df["fault_id"] != 0, "fault_id"].unique())
    return {
        "total": len(df),
        "normal": n_normal,
        "fault": len(df) - n_normal,
        "fault_modes": fault_modes,
        "first": ts.min(),
        "last": ts.max(),
    }


def _load_comparison() -> dict:
    """comparison.json 로드. 없거나 깨졌으면 빈 dict."""
    comp_path = MODELS_DIR / "comparison.json"
    if not comp_path.exists():
        return {}
    try:
        return json.loads(comp_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def detection_summary() -> dict:
    """최신 탐지 요약(scores.parquet) + 기본 모델 성능(comparison.json)."""
    out: dict = {}
    scores_path = MODELS_DIR / "scores.parquet"
    if scores_path.exists():
        s = pd.read_parquet(scores_path)
        out["samples"] = len(s)
        out["anomalies"] = int(s["is_anomaly"].sum())
        out["ratio"] = float(s["is_anomaly"].mean()) if len(s) else 0.0
        if "split" in s.columns:
            out["split_ratios"] = {
                seg: float((grp != 0).mean())
                for seg, grp in s.groupby("split")["fault_id"]
            }

    comp = _load_comparison()
    overall = comp.get("overall", [])
    best = next((m for m in overall if m.get("model") == DEFAULT_MODEL), None)
    if best:
        out["model"] = MODEL_LABELS.get(best["model"], best["model"])
        for key in ("precision", "recall", "f1", "roc_auc", "pr_auc", "fp", "fn"):
            out[key] = best.get(key)
        # 자명 베이스라인 — 홀드아웃 유병률 기준(전건 알람 F1, 무작위 랭킹 AUC)
        pos = int(best.get("tp") or 0) + int(best.get("fn") or 0)
        tot = pos + int(best.get("fp") or 0) + int(best.get("tn") or 0)
        if tot:
            prev = pos / tot
            out["eval_rows"] = tot
            out["prevalence"] = prev
            out["baseline_f1"] = 2 * prev / (1 + prev)
    out["split"] = comp.get("split") or {}
    return out


def fault_coverage(model: str = DEFAULT_MODEL) -> dict:
    """결함모드별 재현율 → 미탐/저탐/완탐 분류.

    fault_id 0(정상)은 '탐지되면 안 되는' 구간이라 재현율 0이 정상 동작이다.
    따라서 미탐 판정에서 반드시 제외한다.
    """
    comp = _load_comparison()
    rows = [
        r
        for r in comp.get("per_fault", [])
        if r.get("model") == model and int(r.get("fault_id", 0)) != 0
    ]
    if not rows:
        return {}
    missed = sorted(int(r["fault_id"]) for r in rows if float(r.get("recall", 0.0)) <= 0.0)
    weak = sorted(
        (int(r["fault_id"]), float(r["recall"]))
        for r in rows
        if 0.0 < float(r.get("recall", 0.0)) < WEAK_RECALL
    )
    # 구간을 [0 / (0,0.8) / [0.8,1.0) / 1.0] 으로 닫는다 — 빠지는 모드가 없어야 한다
    partial = sorted(
        (int(r["fault_id"]), float(r["recall"]))
        for r in rows
        if WEAK_RECALL <= float(r.get("recall", 0.0)) < 1.0
    )
    full = sorted(int(r["fault_id"]) for r in rows if float(r.get("recall", 0.0)) >= 1.0)
    n_classified = len(missed) + len(weak) + len(partial) + len(full)
    return {"missed": missed, "weak": weak, "partial": partial, "full": full,
            "n_modes": len(rows), "n_classified": n_classified}


def milestone_progress(total: int) -> list[dict]:
    """데이터 규모 마일스톤 진행률."""
    rows = []
    for target, desc in MILESTONES:
        pct = min(total / target * 100, 100) if target else 0
        rows.append(
            {"desc": desc, "progress": f"{total:,}/{target:,}", "pct": pct, "met": total >= target}
        )
    return rows


# ── 렌더링 ────────────────────────────────────────────────
def render_report(
    *,
    today: datetime,
    tests_ok: bool,
    tests_summary: str,
    acc: dict,
    det: dict,
    coverage: dict,
    milestones: list[dict],
    provenance: dict,
    warnings: list[str],
) -> str:
    """수집·테스트·탐지·마일스톤을 Markdown 문자열로 조립."""
    db_src = _rel(DB_PATH)
    scores_src = _rel(MODELS_DIR / "scores.parquet")
    comp_src = _rel(MODELS_DIR / "comparison.json")

    L: list[str] = []
    L.append(f"# 제조 공정 이상탐지 워크플로우 — 데일리 리포트 {today:%Y-%m-%d} (KST)")
    L.append("")
    L.append(f"> 자동 생성: {today:%Y-%m-%d %H:%M KST} · `python -m src.report.daily_report`")
    L.append("> 포지셔닝: 수집 → 전처리/EDA → 딥러닝 이상탐지 → 리포트 자동화 워크플로우")
    L.append("")

    if warnings:
        L.append("> ⚠️ **생성 중 산출물이 변경되었습니다. 아래 수치는 신뢰할 수 없습니다.**")
        for w in warnings:
            L.append(f"> - {w}")
        L.append("> 재실행하세요: `python -m src.report.daily_report`")
        L.append("")

    # 0. 근거 산출물
    L.append("## 0. 근거 산출물")
    L.append("")
    L.append("| 파일 | 수정시각 | 크기 | 해시 |")
    L.append("|---|---|---|---|")
    for rel, fp in provenance.items():
        if not fp.get("exists"):
            L.append(f"| `{rel}` | — | — | _없음_ |")
        elif "error" in fp:
            L.append(f"| `{rel}` | — | — | _읽기 실패_ |")
        else:
            L.append(
                f"| `{rel}` | {fp['mtime']} | {fp['bytes']:,} B | `{fp['sha256_12']}` |"
            )
    L.append("")
    L.append(
        "_이 표보다 산출물이 새로우면 아래 수치는 낡은 것입니다. "
        "확인: `python -m src.report.daily_report --check`_"
    )
    L.append("")

    # 1. 회귀 테스트
    L.append("## 1. 회귀 테스트")
    L.append(f"- {'PASS' if tests_ok else 'FAIL'} · `{tests_summary}` · _출처: 본 리포트 생성 시 pytest 실측_")
    L.append("")

    # 2. 수집 통계
    L.append("## 2. 수집 통계")
    if acc["total"] == 0:
        L.append("- _수집 데이터 없음 — `python -m src.collect.scheduler` 로 적재 필요_")
    else:
        ratio = acc["normal"] / acc["fault"] if acc["fault"] else float("nan")
        L.append(f"- 총 적재: **{acc['total']:,}행** (정상 {acc['normal']:,} / 결함 {acc['fault']:,})")
        if acc["fault"]:
            L.append(
                f"- 정상:결함 ≈ {ratio:.1f} : 1 · 결함 IDV {len(acc['fault_modes'])}종 {acc['fault_modes']}"
            )
        if pd.notna(acc.get("first")):
            L.append(f"- 시각 범위: {acc['first']:%Y-%m-%d %H:%M} ~ {acc['last']:%Y-%m-%d %H:%M}")
        L.append(f"- _출처: `{db_src}`_")
    L.append("")

    # 3. 탐지 요약
    L.append("## 3. 이상탐지 요약")
    if not det:
        L.append("- _탐지 산출 없음 — `python -m src.models.detect` 필요_")
    else:
        if "samples" in det:
            L.append(
                f"- 운영 스코어링(전 구간): 시퀀스 **{det['samples']:,}** · 이상 탐지 **{det['anomalies']:,}건** "
                f"({det['ratio'] * 100:.1f}%) · _출처: `{scores_src}`_"
            )
        if "model" in det:
            sp = det.get("split") or {}
            if sp.get("boundary_time"):
                L.append(
                    f"- 분할: 재생 시간축 앞 {sp['train_ratio'] * 100:.0f}% 적합 · "
                    f"경계 {sp['boundary_time'][:16]} · purge {sp['purge_steps']}스텝 · "
                    f"홀드아웃 평가 {sp['eval_seqs']:,}행 — **아래 지표는 홀드아웃 구간만**"
                )
                L.append(
                    "- 재생 시간축은 원본 TEP 파일 순서(정상 전량 → 결함 전량)가 아니라, "
                    "결함 에피소드(연속 48행) 통짜 + 정상 48행 청크를 seed=42로 셔플해 구성한 순서다"
                    "(`src/collect/tep_loader.py` `_interleave_fault_blocks`). 에피소드가 경계에서 "
                    "쪼개지지 않아 샘플 단위 누수는 없으나, 지표는 이 블록 순서 1회 추첨에 조건부다."
                )
                if det.get("split_ratios"):
                    r = det["split_ratios"]
                    parts = " / ".join(
                        f"{seg} {r[seg]:.3f}" for seg in ("train", "purge", "eval") if seg in r
                    )
                    L.append(f"- 구간별 결함 비율: {parts} · _출처: `{scores_src}`_")
            else:
                L.append("- 분할: 없음(무분할 폴백) — 아래 지표는 낙관 편향 가능")
            L.append(
                f"- 기본 모델 **{det['model']}**: F1 {det['f1']:.3f} · ROC-AUC {det['roc_auc']:.3f} · "
                f"PR-AUC {det['pr_auc']:.3f} (P {det['precision']:.3f} / R {det['recall']:.3f})"
            )
            L.append(
                f"- 오탐(FP) {int(det['fp'])} · 미탐(FN) {int(det['fn'])} · _출처: `{comp_src}`_"
            )
            if det.get("prevalence") is not None:
                prev, bf1 = det["prevalence"], det["baseline_f1"]
                beats = det["f1"] > bf1 and det["roc_auc"] > 0.5 and det["pr_auc"] > prev
                L.append(
                    f"- 자명 베이스라인 대비(홀드아웃 {det['eval_rows']:,}행 · 유병률 {prev:.3f}): "
                    f"전건 알람 F1 {bf1:.3f} · 무작위 랭킹 ROC-AUC 0.500 / PR-AUC {prev:.3f} → "
                    + ("**세 축 모두 상회**" if beats else "**⚠️ 베이스라인 미달 축 존재**")
                )
            L.append(
                "- 반복 표준편차(±): 미반영 — 모델 시드 1회 × 블록 순서 시드 1회 실행. "
                "방법론 검수 유보 항목."
            )
            L.append(
                f"- 점수 평활 rolling median(window={SCORE_SMOOTH_WINDOW}, center=True) — "
                f"각 시점이 뒤 {SCORE_SMOOTH_WINDOW // 2}스텝"
                f"({SCORE_SMOOTH_WINDOW // 2 * SAMPLE_INTERVAL_MIN}분)을 참조. 온라인 운영 시 동일 성능은 미측정."
            )

        # 한계 — 하드코딩 금지. comparison.json 의 결함모드별 재현율에서 산출한다.
        if coverage:
            L.append("")
            L.append(f"**한계 (정직 표기) — `{comp_src}` 결함모드별 재현율에서 산출**")
            evaluated = set(coverage["missed"]) | {i for i, _ in coverage["weak"]} \
                | {i for i, _ in coverage["partial"]} | set(coverage["full"])
            uneval = sorted(set(acc.get("fault_modes") or []) - evaluated)
            if uneval:
                modes = " · ".join(f"IDV {i}" for i in uneval)
                L.append(
                    f"- **홀드아웃 미배정으로 미평가: {modes} ({len(uneval)}종)** — "
                    "IDV당 에피소드가 1개뿐이라 경계 반대편 결함모드는 이번 분할에서 평가 불가. "
                    "아래 진술은 평가된 모드에 한하며, 미평가 모드의 미탐 여부는 미측정이다."
                )
            if coverage["missed"]:
                modes = " · ".join(f"IDV {i}" for i in coverage["missed"])
                L.append(f"- 완전 미탐(재현율 0): {modes} → 대체 감시 수단 필요")
            else:
                L.append(
                    f"- 완전 미탐(재현율 0) 없음 — 평가된 {coverage['n_modes']}종에 한한 진술. "
                    "이전 사이클의 완전 미탐 모드가 미평가 목록에 있다면 해소가 아니라 미평가다."
                )
            if coverage["weak"]:
                modes = " · ".join(f"IDV {i} {r:.2f}" for i, r in coverage["weak"])
                L.append(f"- 탐지 취약(재현율 < {WEAK_RECALL:.1f}): {modes}")
            if coverage["partial"]:
                modes = " · ".join(f"IDV {i} {r:.2f}" for i, r in coverage["partial"])
                L.append(f"- 부분 탐지(재현율 {WEAK_RECALL:.1f}~1.00 미만): {modes}")
            if coverage["full"]:
                modes = " · ".join(f"IDV {i}" for i in coverage["full"])
                L.append(f"- 완전 탐지(재현율 1.00): {modes}")
            L.append(
                f"- 평가 대상 결함모드 {coverage['n_modes']}종 (정상 구간 제외) · "
                f"분류 합계 {coverage['n_classified']}종"
                + ("" if coverage["n_classified"] == coverage["n_modes"] else " ⚠️ 불일치")
            )
    L.append("")

    # 4. 마일스톤
    L.append("## 4. 데이터 규모 마일스톤")
    for m in milestones:
        mark = "달성" if m["met"] else f"{m['pct']:.0f}%"
        L.append(f"- [{mark}] {m['desc']} ({m['progress']})")
    L.append(f"- _출처: `{db_src}` 총행수 기준_")
    L.append("")

    L.append("---")
    L.append(
        "_본 리포트 생성 과정에서는 재학습·재계산을 하지 않는다(산출물 로드 요약). "
        "모델 자체의 최근 학습 시각은 §0 `comparison.json`·`scores.parquet` 수정시각을 따른다._"
    )
    L.append("")
    L.append(PROVENANCE_BEGIN)
    L.append(json.dumps(provenance, ensure_ascii=False, indent=2))
    L.append(PROVENANCE_END)
    L.append("")
    return "\n".join(L)


# ── 실행 ──────────────────────────────────────────────────
def build() -> Path:
    """데일리 리포트 생성 → reports/daily_YYYYMMDD.md 경로 반환."""
    today = datetime.now(KST)

    # 생성 전 지문. pytest 가 수 분 걸리므로 그 사이 산출물이 바뀔 수 있다.
    prov_before = collect_provenance()

    # 기존 리포트 낡음 감사 — 덮어쓰기 전에 콘솔로 드러낸다.
    findings = audit_reports(prov_before)
    if findings:
        print("[daily_report] 기존 리포트 감사:")
        for path, (warns, errors) in findings.items():
            print(f"  - {_rel(path)}")
            for w in warns:
                print(f"      [경고] {w}")
            for e in errors:
                print(f"      [낡음] {e}")

    tests_ok, tests_summary = run_pytest()
    acc = collect_stats()
    det = detection_summary()
    coverage = fault_coverage()
    milestones = milestone_progress(acc["total"])

    # 생성 후 지문과 대조 — 읽는 도중 산출물이 갱신·소실됐으면 리포트에 경고를 박는다.
    prov_after = collect_provenance()
    gone, changed = diff_provenance(prov_before, prov_after)
    warnings = changed + gone

    md = render_report(
        today=today,
        tests_ok=tests_ok,
        tests_summary=tests_summary,
        acc=acc,
        det=det,
        coverage=coverage,
        milestones=milestones,
        provenance=prov_after,
        warnings=warnings,
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"daily_{today:%Y%m%d}.md"
    out.write_text(md, encoding="utf-8")

    print(md)
    print(f"[daily_report] 생성 완료 → {out}")
    if warnings:
        print("[daily_report] 경고: 생성 중 산출물이 변경됨. 재실행 권장.")
    return out


def check() -> int:
    """기존 리포트가 현재 산출물 기준으로 낡았는지 감사.

    실제 낡음(지문 불일치)이 있을 때만 1을 반환한다 — CI 머지 게이트용.
    가드 도입 이전 리포트(지문 없음)는 경고만 출력하고 게이트를 막지 않는다.
    """
    current = collect_provenance()
    findings = audit_reports(current)
    has_error = False
    if not findings:
        print("[daily_report --check] 모든 리포트가 현재 산출물과 일치합니다.")
        return 0
    for path, (warns, errors) in findings.items():
        print(f"  - {_rel(path)}")
        for w in warns:
            print(f"      [보류] {w}")
        for e in errors:
            print(f"      [낡음] {e}")
            has_error = True
    if has_error:
        print("[daily_report --check] 낡은 리포트 존재 — 재생성 필요: python -m src.report.daily_report")
        return 1
    print("[daily_report --check] 실제 낡음 없음(경고만) — 게이트 통과.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="데일리 현황 리포트 생성/감사")
    parser.add_argument(
        "--check",
        action="store_true",
        help="리포트를 만들지 않고 기존 리포트의 낡음 여부만 검사한다",
    )
    args = parser.parse_args()
    if args.check:
        sys.exit(check())
    build()


if __name__ == "__main__":
    main()

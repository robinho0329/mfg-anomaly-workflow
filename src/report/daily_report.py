"""④ 데일리 현황 리포트 — 워크플로우 산출물을 모아 Markdown으로 요약. (소유: mfg-reporter)

Claude API 호출 없이, 기존 산출물 로드 기반(무거운 학습/재계산 없음)으로 다음을 수집:
  1. 회귀 테스트(pytest) 통과 여부
  2. 수집 통계(총행수·정상/결함·시각범위)
  3. 최신 탐지 요약(이상 건수·비율·기본 모델 성능)
  4. 마일스톤 진행률(데이터 규모 게이트)

산출: reports/daily_YYYYMMDD.md (+ 콘솔 출력)
실행: python -m src.report.daily_report
"""

import json
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
    MODELS_DIR,
    REPORTS_DIR,
)

KST = timezone(timedelta(hours=9))
DEFAULT_MODEL = "vae"
MODEL_LABELS = {"lstm_ae": "LSTM-AE", "vae": "VAE", "transformer_ae": "Transformer-AE"}

# 데이터 규모 마일스톤 (게이트) — (target_rows, 설명)
MILESTONES = [
    (300, "딥러닝 AE 학습 안정화(정상 시퀀스 확보)"),
    (885, "현재 비교 평가 기준(정상690/결함195)"),
    (2000, "결함 IDV 다양성 확장 평가"),
    (5000, "운영급 데이터 규모"),
]


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


def detection_summary() -> dict:
    """최신 탐지 요약(scores.parquet) + 기본 모델 성능(comparison.json)."""
    out: dict = {}
    scores_path = MODELS_DIR / "scores.parquet"
    if scores_path.exists():
        s = pd.read_parquet(scores_path)
        out["samples"] = len(s)
        out["anomalies"] = int(s["is_anomaly"].sum())
        out["ratio"] = float(s["is_anomaly"].mean()) if len(s) else 0.0

    comp_path = MODELS_DIR / "comparison.json"
    if comp_path.exists():
        comp = json.loads(comp_path.read_text(encoding="utf-8"))
        overall = comp.get("overall", [])
        best = next((m for m in overall if m.get("model") == DEFAULT_MODEL), None)
        if best:
            out["model"] = MODEL_LABELS.get(best["model"], best["model"])
            out["precision"] = best.get("precision")
            out["recall"] = best.get("recall")
            out["f1"] = best.get("f1")
            out["roc_auc"] = best.get("roc_auc")
            out["pr_auc"] = best.get("pr_auc")
            out["fp"] = best.get("fp")
            out["fn"] = best.get("fn")
    return out


def milestone_progress(total: int) -> list[dict]:
    """데이터 규모 마일스톤 진행률."""
    rows = []
    for target, desc in MILESTONES:
        pct = min(total / target * 100, 100) if target else 0
        rows.append(
            {"desc": desc, "progress": f"{total}/{target}", "pct": pct, "met": total >= target}
        )
    return rows


def render_report(
    *,
    today: datetime,
    tests_ok: bool,
    tests_summary: str,
    acc: dict,
    det: dict,
    milestones: list[dict],
) -> str:
    """수집·테스트·탐지·마일스톤을 Markdown 문자열로 조립."""
    L: list[str] = []
    L.append(f"# 제조 공정 이상탐지 워크플로우 — 데일리 리포트 {today:%Y-%m-%d} (KST)")
    L.append("")
    L.append(f"> 자동 생성: {today:%Y-%m-%d %H:%M KST} · `python -m src.report.daily_report`")
    L.append("> 포지셔닝: 수집 → 전처리/EDA → 딥러닝 이상탐지 → 리포트 자동화 워크플로우")
    L.append("")

    # 1. 회귀 테스트
    L.append("## 1. 회귀 테스트")
    L.append(f"- {'PASS' if tests_ok else 'FAIL'} · `{tests_summary}`")
    L.append("")

    # 2. 수집 통계
    L.append("## 2. 수집 통계")
    if acc["total"] == 0:
        L.append("- _수집 데이터 없음 — `python -m src.collect.scheduler` 로 적재 필요_")
    else:
        ratio = acc["normal"] / acc["fault"] if acc["fault"] else float("nan")
        L.append(f"- 총 적재: **{acc['total']:,}행** (정상 {acc['normal']:,} / 결함 {acc['fault']:,})")
        if acc["fault"]:
            L.append(f"- 정상:결함 ≈ {ratio:.1f} : 1 · 결함 IDV {len(acc['fault_modes'])}종 {acc['fault_modes']}")
        if pd.notna(acc.get("first")):
            L.append(f"- 시각 범위: {acc['first']:%Y-%m-%d %H:%M} ~ {acc['last']:%Y-%m-%d %H:%M}")
    L.append("")

    # 3. 탐지 요약
    L.append("## 3. 최신 이상탐지 요약")
    if not det:
        L.append("- _탐지 산출 없음 — `python -m src.models.detect` 필요_")
    else:
        if "samples" in det:
            L.append(
                f"- 평가 시퀀스: **{det['samples']:,}** · 이상 탐지 **{det['anomalies']:,}건** "
                f"({det['ratio'] * 100:.1f}%)"
            )
        if "model" in det:
            L.append(
                f"- 기본 모델 **{det['model']}**: F1 {det['f1']:.3f} · ROC-AUC {det['roc_auc']:.3f} · "
                f"PR-AUC {det['pr_auc']:.3f} (P {det['precision']:.3f} / R {det['recall']:.3f})"
            )
            L.append(f"- 오탐(FP) {int(det['fp'])} · 미탐(FN) {int(det['fn'])}")
            L.append("- 한계(정직 표기): IDV 1·4·8은 재구성오차가 정상과 구별 불가 → AE 원리상 미탐.")
    L.append("")

    # 4. 마일스톤
    L.append("## 4. 데이터 규모 마일스톤")
    for m in milestones:
        mark = "달성" if m["met"] else f"{m['pct']:.0f}%"
        L.append(f"- [{mark}] {m['desc']} ({m['progress']})")
    L.append("")

    L.append("---")
    L.append("_본 리포트는 기존 산출물 로드 기반 자동 요약입니다(재학습/재계산 없음)._")
    L.append("")
    return "\n".join(L)


def build() -> Path:
    """데일리 리포트 생성 → reports/daily_YYYYMMDD.md 경로 반환."""
    today = datetime.now(KST)
    tests_ok, tests_summary = run_pytest()
    acc = collect_stats()
    det = detection_summary()
    milestones = milestone_progress(acc["total"])

    md = render_report(
        today=today,
        tests_ok=tests_ok,
        tests_summary=tests_summary,
        acc=acc,
        det=det,
        milestones=milestones,
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"daily_{today:%Y%m%d}.md"
    out.write_text(md, encoding="utf-8")

    print(md)
    print(f"[daily_report] 생성 완료 → {out}")
    return out


if __name__ == "__main__":
    build()

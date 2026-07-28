"""③ 딥러닝 이상탐지 — 시드 반복 평가. (소유: mfg-model)

단일 실행 지표는 '모델 시드 1회 × 블록 순서 시드 1회'의 이중 단일 추첨이다.
블록 순서 시드를 바꾸면 홀드아웃에 배정되는 결함모드 구성 자체가 바뀌므로,
반복 실행 하나로 두 가지를 동시에 얻는다:
  (1) 지표 변동폭 — 평균 ± 표준편차
  (2) 결함모드 커버리지 — 회차를 합치면 대부분의 IDV가 한 번 이상 평가된다

stream.db는 건드리지 않는다. 원본 CSV에서 시드별 재생 순서를 만들고 전처리를
메모리에서 적용하므로, 커밋된 산출물·리포트 지문에 영향을 주지 않는다.

산출: data/models/repeat_eval.json
실행:
    python -m src.models.repeat_eval                      # 기본 5시드
    python -m src.models.repeat_eval --seeds 42,43,44     # 시드 지정
    python -m src.models.repeat_eval --models vae         # 모델 한정(빠름)
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import (  # noqa: E402
    MODELS_DIR,
    RANDOM_STATE,
    RAW_DIR,
    SAMPLE_INTERVAL_MIN,
    SEQ_LEN,
)
from src.collect.tep_loader import load_raw_tep  # noqa: E402
from src.models.compare import _metrics_overall, _metrics_per_fault  # noqa: E402
from src.models.detect import MODEL_REGISTRY, fit_and_score  # noqa: E402
from src.models.tf_seed import set_active_seed  # noqa: E402
from src.pipeline.preprocess import clean  # noqa: E402

# 기본 반복 시드 — 42는 프로젝트 표준 시드, 이후는 연속값(결정적)
DEFAULT_SEEDS = [RANDOM_STATE + i for i in range(5)]
METRIC_KEYS = ("f1", "roc_auc", "pr_auc", "precision", "recall")


def build_frame(seed: int) -> pd.DataFrame:
    """시드별 재생 순서로 원본을 읽어 전처리까지 마친 프레임(메모리 전용)."""
    real = load_raw_tep(RAW_DIR, seed=seed)
    if real is None or real.empty:
        return pd.DataFrame()
    df = real.copy()
    # 재생 시간축 부여 — 실제 공정 시각이 아니라 블록 순서에 따른 연속 스탬프
    df["timestamp"] = pd.date_range(
        "2026-01-01", periods=len(df), freq=f"{SAMPLE_INTERVAL_MIN}min"
    )
    ordered = ["timestamp", "fault_id"] + [
        c for c in df.columns if c not in ("timestamp", "fault_id")
    ]
    return clean(df[ordered])


def eval_one(df: pd.DataFrame, model_name: str) -> tuple[dict, list] | None:
    """모델 1개를 홀드아웃 구간에서 평가 → (전체지표, 결함모드별)."""
    out = fit_and_score(df, model_name=model_name)
    if out.empty:
        return None
    ev = out[out["split"] == "eval"]
    if ev.empty:
        return None
    y_true = (ev["fault_id"] != 0).astype(int).to_numpy()
    y_pred = ev["is_anomaly"].to_numpy()
    score = ev["anomaly_score"].to_numpy()
    m = _metrics_overall(y_true, y_pred, score)
    m["model"] = model_name
    m["eval_rows"] = int(len(ev))
    # 홀드아웃에 결함(양성)이 없으면 F1·AUC가 정의되지 않는다. 이때 F1=0은
    # '성능이 0'이 아니라 '평가 불가'이므로 집계에서 제외해야 한다.
    m["degenerate"] = bool(y_true.sum() == 0)
    pf = [r | {"model": model_name} for r in _metrics_per_fault(ev["fault_id"].to_numpy(), y_pred)]
    return m, pf


def _paired_all(per_seed: list[dict], models: list[str]) -> dict:
    """주요 지표별 대응 비교. 지표마다 순위가 엇갈릴 수 있어 함께 본다.

    F1은 임계값 보정에 의존하고 ROC/PR-AUC는 임계값과 무관한 순위 품질이다.
    둘이 다른 모델을 가리키면 '판별력 차이'가 아니라 '임계값 보정 차이'일 수 있다.
    """
    return {k: _paired(per_seed, models, key=k) for k in ("f1", "roc_auc", "pr_auc")}


def _paired(per_seed: list[dict], models: list[str], key: str = "f1") -> list[dict]:
    """모델 간 대응(paired) 비교.

    세 모델은 회차마다 **동일 분할**에서 평가되므로, 모델 차이를 판단할 때
    비교 대상은 주변 표준편차가 아니라 회차별 차이의 표준편차다. 주변 편차의
    대부분은 '이번 분할이 쉬웠나 어려웠나'라는 공통 성분이고 차이에서 상쇄된다.
    """
    by_seed = {
        s["seed"]: {
            o["model"]: float(o[key])
            for o in s["overall"]
            if not o.get("degenerate") and o.get(key) == o.get(key)  # NaN 제외
        }
        for s in per_seed
    }
    out: list[dict] = []
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            diffs = [v[a] - v[b] for v in by_seed.values() if a in v and b in v]
            if len(diffs) < 2:
                continue
            sd = statistics.stdev(diffs)
            out.append({
                "a": a, "b": b, "n": len(diffs),
                "mean_diff": statistics.fmean(diffs),
                "std_diff": sd,
                "se_diff": sd / len(diffs) ** 0.5,
                "wins_a": sum(1 for d in diffs if d > 0),
                "wins_b": sum(1 for d in diffs if d < 0),
            })
    return out


def _summarize(per_seed: list[dict], models: list[str]) -> dict:
    """회차별 결과 → 모델별 평균±표준편차 + 결함모드 커버리지."""
    summary: dict = {}
    n_degenerate = sum(
        1 for s in per_seed for o in s["overall"] if o.get("degenerate")
    ) // max(len(models), 1)
    for name in models:
        all_runs = [o for s in per_seed for o in s["overall"] if o["model"] == name]
        # 평가 불가 회차(홀드아웃에 결함 0건) 제외 — 모든 지표를 같은 표본에서 계산한다
        runs = [o for o in all_runs if not o.get("degenerate")]
        if not runs:
            continue
        stats = {"n_runs": len(runs), "n_excluded": len(all_runs) - len(runs)}
        for key in METRIC_KEYS:
            vals = [float(r[key]) for r in runs if r.get(key) == r.get(key)]  # NaN 제외
            if not vals:
                continue
            stats[f"{key}_mean"] = statistics.fmean(vals)
            stats[f"{key}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
            stats[f"{key}_min"] = min(vals)
            stats[f"{key}_max"] = max(vals)
        summary[name] = stats

    # 커버리지 — 결함모드별로 몇 회차에서 평가됐고 재현율이 어땠는지(정상 0 제외)
    coverage: dict = {}
    for name in models:
        by_fault: dict[int, list[float]] = {}
        for s in per_seed:
            for r in s["per_fault"]:
                if r["model"] != name or int(r["fault_id"]) == 0:
                    continue
                by_fault.setdefault(int(r["fault_id"]), []).append(float(r["recall"]))
        coverage[name] = {
            str(fid): {
                "n_seeds": len(vals),
                "recall_mean": statistics.fmean(vals),
                "recall_std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "recall_min": min(vals),
                # 완전 미탐이 몇 회차에서 났는지 — 평균만 보면 가려진다.
                # 단일 분할에서 '완전 미탐 없음'이어도 다른 추첨에서는 0일 수 있다.
                "n_zero": sum(1 for v in vals if v <= 0.0),
            }
            for fid, vals in sorted(by_fault.items())
        }
    return {
        "summary": summary,
        "coverage": coverage,
        "paired": _paired_all(per_seed, models),
        "n_degenerate_seeds": n_degenerate,
    }


def recompute(path: Path | None = None) -> dict:
    """저장된 회차 결과로 요약만 다시 계산한다(재학습 없음)."""
    path = path or (MODELS_DIR / "repeat_eval.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    models = data.get("models") or list(MODEL_REGISTRY)
    # 구버전 결과 보정 — 양성이 0건이면 평가 불가 회차로 표시
    for s in data.get("per_seed", []):
        for o in s.get("overall", []):
            o.setdefault("degenerate", int(o.get("tp", 0)) + int(o.get("fn", 0)) == 0)
    data.update(_summarize(data.get("per_seed", []), models))
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[repeat] 요약 재계산 완료 → {path}")
    return data


def run(seeds: list[int] | None = None, models: list[str] | None = None) -> dict:
    """시드별로 재생 순서·모델 초기화를 바꿔 반복 평가 → repeat_eval.json 저장."""
    seeds = seeds or DEFAULT_SEEDS
    models = models or list(MODEL_REGISTRY)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    per_seed: list[dict] = []
    for i, seed in enumerate(seeds, 1):
        set_active_seed(seed)  # 모델 초기화 시드도 회차별로 변경
        df = build_frame(seed)
        if len(df) < SEQ_LEN + 10:
            print(f"[repeat] seed={seed} 데이터 부족({len(df)}행) — 건너뜀")
            continue
        overall, per_fault, split_meta = [], [], None
        for name in models:
            res = eval_one(df, name)
            if res is None:
                print(f"[repeat] seed={seed} {name}: 홀드아웃 없음 — 건너뜀")
                continue
            m, pf = res
            overall.append(m)
            per_fault.extend(pf)
            print(
                f"[repeat {i}/{len(seeds)}] seed={seed} {name:>15}: "
                f"F1 {m['f1']:.3f} ROC-AUC {m['roc_auc']:.3f} PR-AUC {m['pr_auc']:.3f}"
            )
        eval_modes = sorted({int(r["fault_id"]) for r in per_fault if int(r["fault_id"]) != 0})
        per_seed.append(
            {"seed": seed, "overall": overall, "per_fault": per_fault, "eval_modes": eval_modes}
        )
        print(f"[repeat {i}/{len(seeds)}] seed={seed} 홀드아웃 결함모드: {eval_modes}")

    result = {"seeds": seeds, "models": models, "per_seed": per_seed, **_summarize(per_seed, models)}
    out = MODELS_DIR / "repeat_eval.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== 반복 평가 요약 ({len(per_seed)}회차) ===")
    for name, st in result["summary"].items():
        print(
            f"{name:>15}: F1 {st['f1_mean']:.3f} ± {st['f1_std']:.3f} "
            f"[{st['f1_min']:.3f}~{st['f1_max']:.3f}] · "
            f"ROC-AUC {st['roc_auc_mean']:.3f} ± {st['roc_auc_std']:.3f}"
        )
    for name in models:
        modes = sorted(int(k) for k in result["coverage"].get(name, {}))
        print(f"{name:>15}: 평가된 결함모드 {len(modes)}종 {modes}")
    print(f"[repeat] 저장 → {out}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="시드 반복 평가(± 표기 + 커버리지 확보)")
    parser.add_argument("--seeds", type=str, default="",
                        help="쉼표 구분 시드 목록(기본: 42~46)")
    parser.add_argument("--models", type=str, default="",
                        help="쉼표 구분 모델명(기본: 전체)")
    parser.add_argument("--recompute", action="store_true",
                        help="재학습 없이 저장된 회차 결과로 요약만 다시 계산")
    args = parser.parse_args()
    if args.recompute:
        recompute()
        return
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()] or None
    models = [m.strip() for m in args.models.split(",") if m.strip()] or None
    run(seeds=seeds, models=models)


if __name__ == "__main__":
    main()

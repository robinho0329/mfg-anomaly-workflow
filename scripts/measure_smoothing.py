"""평활 방향이 성능에 미치는 영향 측정 — '온라인 성능 미측정'을 지우기 위한 실측.

center=True 는 각 시점이 뒤 window//2 스텝을 참조한다. 오프라인 평가에서는 유리하지만
온라인 운영에서는 미래를 못 보므로 재현되지 않는다. 얼마나 유리했는지를 숫자로 남긴다.

같은 분할·같은 시드에서 평활 방향만 바꿔 대응 비교한다 — 분할 난이도 공통 성분이
상쇄되므로 주변 편차보다 훨씬 작은 차이도 잡힌다.

산출: data/models/smoothing_effect.json
실행: python scripts/measure_smoothing.py [--seeds 42,43,44]
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.settings import MODELS_DIR, RANDOM_STATE  # noqa: E402
from src.models.compare import _metrics_overall  # noqa: E402
from src.models.detect import MODEL_REGISTRY, fit_and_score  # noqa: E402
from src.models.repeat_eval import build_frame  # noqa: E402
from src.models.tf_seed import set_active_seed  # noqa: E402
import src.models.scoring as scoring  # noqa: E402

METRICS = ("f1", "roc_auc", "pr_auc", "recall", "precision")


def evaluate(df, model_name: str, center: bool) -> dict | None:
    """평활 방향을 지정해 홀드아웃 지표를 낸다."""
    # 전역을 바꾸는 대신 인자로 넘긴다 — 기본 인자는 정의 시점에 고정되므로
    # 모듈 전역만 바꾸면 반영되지 않는다(첫 측정이 전부 0.0000 이 나온 원인).
    out = fit_and_score(df, model_name=model_name, smooth_center=center)
    if out.empty:
        return None
    ev = out[out["split"] == "eval"]
    if ev.empty or (ev["fault_id"] != 0).sum() == 0:
        return None
    y = (ev["fault_id"] != 0).astype(int).to_numpy()
    return _metrics_overall(y, ev["is_anomaly"].to_numpy(), ev["anomaly_score"].to_numpy())


def run(seeds: list[int], models: list[str]) -> dict:
    rows = []
    for seed in seeds:
        set_active_seed(seed)
        df = build_frame(seed)
        for name in models:
            a = evaluate(df, name, center=True)     # 오프라인(미래 참조)
            b = evaluate(df, name, center=False)    # 온라인 안전(후행)
            if not a or not b:
                continue
            rows.append({"seed": seed, "model": name,
                         "center": {k: a[k] for k in METRICS},
                         "trailing": {k: b[k] for k in METRICS}})
            print(f"  seed {seed} {name:>15}: F1 중심 {a['f1']:.3f} → 후행 {b['f1']:.3f} "
                  f"({b['f1'] - a['f1']:+.3f})")

    summary = {}
    for k in METRICS:
        diffs = [r["trailing"][k] - r["center"][k] for r in rows
                 if r["trailing"][k] == r["trailing"][k] and r["center"][k] == r["center"][k]]
        if len(diffs) < 2:
            continue
        sd = statistics.stdev(diffs)
        summary[k] = {
            "n": len(diffs),
            "mean_diff": statistics.fmean(diffs),      # 후행 − 중심 (음수면 후행이 손해)
            "std_diff": sd,
            "se_diff": sd / len(diffs) ** 0.5,
            "worse": sum(1 for d in diffs if d < 0),
        }

    result = {"seeds": seeds, "models": models, "per_run": rows, "summary": summary,
              "note": ("center=True 는 각 시점이 미래 스텝을 참조해 온라인에서 재현 불가하다. "
                       "같은 분할에서 방향만 바꾼 대응 비교이므로 분할 난이도는 상쇄된다.")}
    out = MODELS_DIR / "smoothing_effect.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 후행 − 중심 (대응 비교) ===")
    for k, s in summary.items():
        det = "차이 검출" if abs(s["mean_diff"]) > 2 * s["se_diff"] else "차이 미검출"
        print(f"  {k:>10}: {s['mean_diff']:+.4f} ± SE {s['se_diff']:.4f} "
              f"(n={s['n']}, 손해 {s['worse']}회) → {det}")
    print(f"[smoothing] 저장 → {out}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="평활 방향 영향 측정")
    ap.add_argument("--seeds", default=",".join(str(RANDOM_STATE + i) for i in range(3)))
    ap.add_argument("--models", default="")
    a = ap.parse_args()
    run([int(s) for s in a.seeds.split(",") if s.strip()],
        [m.strip() for m in a.models.split(",") if m.strip()] or list(MODEL_REGISTRY))


if __name__ == "__main__":
    main()

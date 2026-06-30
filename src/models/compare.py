"""③ 딥러닝 이상탐지 — 모델 비교 평가. (소유: mfg-model)

LSTM-AE vs VAE vs Transformer-AE를 동일 데이터/전처리/점수 로직으로 학습·평가한다.
정답은 fault_id != 0(이상=양성). 전체/결함 IDV별 precision/recall/f1 + ROC-AUC/PR-AUC를
산출하고, ROC·PR 커브 PNG를 저장한다.

산출(스키마 유지, mfg-reporter 의존):
  - comparison.parquet           : 모델별 P/R/F1 + AUC + 혼동행렬
  - comparison_per_fault.parquet : 모델·결함 IDV별 탐지율
  - comparison.json              : 위 둘 + ROC/PR 커브 좌표
  - roc_pr_curves.png            : ROC·PR 커브 시각화

실행: python -m src.models.compare
"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from config.settings import MODELS_DIR, PROCESSED_DIR, SEQ_LEN
from src.models.detect import MODEL_REGISTRY, fit_and_score


def _load_clean() -> pd.DataFrame:
    path = PROCESSED_DIR / "clean.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _metrics_overall(y_true: np.ndarray, y_pred: np.ndarray, score: np.ndarray) -> dict:
    """이상=양성 기준 전체 P/R/F1 + ROC-AUC/PR-AUC."""
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    # 양/음성이 모두 존재할 때만 AUC 정의됨(graceful)
    has_both = len(set(y_true.tolist())) == 2
    roc_auc = float(roc_auc_score(y_true, score)) if has_both else float("nan")
    pr_auc = float(average_precision_score(y_true, score)) if has_both else float("nan")
    return {
        "precision": p, "recall": r, "f1": f,
        "roc_auc": roc_auc, "pr_auc": pr_auc,
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
        "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
    }


def _metrics_per_fault(fault_id: np.ndarray, y_pred: np.ndarray) -> list:
    """결함 IDV별 탐지율(recall). 정상(fault_id=0) 행은 오탐(FP) 수/비율."""
    rows = []
    for fid in sorted(set(fault_id) - {0}):
        mask = fault_id == fid
        n = int(mask.sum())
        detected = int((y_pred[mask] == 1).sum())
        rows.append({
            "fault_id": int(fid), "n_rows": n, "detected": detected,
            "recall": detected / n if n else 0.0,
        })
    normal_mask = fault_id == 0
    fp = int((y_pred[normal_mask] == 1).sum())
    n_norm = int(normal_mask.sum())
    rows.append({
        "fault_id": 0, "n_rows": n_norm, "detected": fp,
        "recall": fp / n_norm if n_norm else 0.0,  # 정상 행은 false positive rate
    })
    return rows


def _plot_curves(curves: dict, overall: pd.DataFrame) -> None:
    """모델별 ROC·PR 커브를 한 PNG로 저장(matplotlib 미설치 시 graceful 스킵)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[compare] matplotlib 미설치 — 커브 PNG 생략")
        return

    auc_map = overall.set_index("model")[["roc_auc", "pr_auc"]].to_dict("index")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for name, cv in curves.items():
        ax1.plot(cv["fpr"], cv["tpr"], label=f"{name} (AUC={auc_map[name]['roc_auc']:.3f})")
        ax2.plot(cv["recall"], cv["precision"], label=f"{name} (AP={auc_map[name]['pr_auc']:.3f})")
    ax1.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax1.set(title="ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax2.set(title="Precision-Recall Curve", xlabel="Recall", ylabel="Precision")
    for ax in (ax1, ax2):
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("TEP Anomaly Detection - Model Comparison (885 rows, fault=positive)")
    fig.tight_layout()
    fig.savefig(MODELS_DIR / "roc_pr_curves.png", dpi=120)
    plt.close(fig)
    print(f"[compare] 커브 저장: {MODELS_DIR / 'roc_pr_curves.png'}")


def run() -> pd.DataFrame:
    """모든 등록 모델을 평가하고 비교표/커브를 저장."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = _load_clean()
    if len(df) < SEQ_LEN + 10:
        print(f"[compare] 데이터 부족({len(df)}행)")
        return pd.DataFrame()

    overall_rows, per_fault_rows, curves = [], [], {}
    for name in MODEL_REGISTRY:
        print(f"[compare] 학습/평가: {name} ...")
        out = fit_and_score(df, model_name=name)
        if out.empty:
            continue
        y_true = (out["fault_id"] != 0).astype(int).to_numpy()
        y_pred = out["is_anomaly"].to_numpy()
        score = out["anomaly_score"].to_numpy()
        fault_id = out["fault_id"].to_numpy()

        m = _metrics_overall(y_true, y_pred, score)
        m["model"] = name
        overall_rows.append(m)
        for pf in _metrics_per_fault(fault_id, y_pred):
            pf["model"] = name
            per_fault_rows.append(pf)

        if len(set(y_true.tolist())) == 2:
            fpr, tpr, _ = roc_curve(y_true, score)
            prec, rec, _ = precision_recall_curve(y_true, score)
            curves[name] = {
                "fpr": fpr.tolist(), "tpr": tpr.tolist(),
                "precision": prec.tolist(), "recall": rec.tolist(),
            }
        print(
            f"    P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
            f"ROC-AUC={m['roc_auc']:.3f} PR-AUC={m['pr_auc']:.3f} (FP={m['fp']})"
        )

    overall = pd.DataFrame(overall_rows)[
        ["model", "precision", "recall", "f1", "roc_auc", "pr_auc", "tp", "fp", "fn", "tn"]
    ]
    per_fault = pd.DataFrame(per_fault_rows)[
        ["model", "fault_id", "n_rows", "detected", "recall"]
    ]

    overall.to_parquet(MODELS_DIR / "comparison.parquet", engine="pyarrow", index=False)
    per_fault.to_parquet(MODELS_DIR / "comparison_per_fault.parquet", engine="pyarrow", index=False)
    (MODELS_DIR / "comparison.json").write_text(
        json.dumps({
            "n_rows": int(len(df)),
            "overall": overall.to_dict(orient="records"),
            "per_fault": per_fault.to_dict(orient="records"),
            "curves": curves,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _plot_curves(curves, overall)

    print("\n=== 전체 비교 ===")
    print(overall.to_string(index=False))
    print("\n=== 결함 IDV별(fault_id=0 행은 오탐율) ===")
    print(per_fault.to_string(index=False))
    best = overall.sort_values("f1", ascending=False).iloc[0]["model"]
    print(f"\n[compare] 최고 F1 모델: {best}")
    return overall


if __name__ == "__main__":
    run()

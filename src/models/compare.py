"""③ 딥러닝 이상탐지 — 모델 비교 평가. (소유: mfg-model)

LSTM-AE vs VAE vs Transformer-AE를 동일 데이터/전처리/임계값 로직으로 학습·평가한다.
정답은 fault_id != 0(이상=양성). 전체 및 결함 IDV별 precision/recall/f1을 산출해
data/models/comparison.parquet (+ comparison.json)으로 저장한다.

실행: python -m src.models.compare
"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from config.settings import MODELS_DIR, PROCESSED_DIR, SEQ_LEN
from src.models.detect import MODEL_REGISTRY, fit_and_score


def _load_clean() -> pd.DataFrame:
    path = PROCESSED_DIR / "clean.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _metrics_overall(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """이상=양성 기준 전체 precision/recall/f1."""
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _metrics_per_fault(fault_id: np.ndarray, y_pred: np.ndarray) -> list:
    """결함 IDV별 탐지율(recall). 각 IDV 행을 양성으로, 정상 대비 평가."""
    rows = []
    normal_mask = fault_id == 0
    for fid in sorted(set(fault_id) - {0}):
        mask = fault_id == fid
        n = int(mask.sum())
        detected = int((y_pred[mask] == 1).sum())
        recall = detected / n if n else 0.0
        rows.append({
            "fault_id": int(fid),
            "n_rows": n,
            "detected": detected,
            "recall": recall,
        })
    # 정상 구간 오탐율도 함께 기록
    fp = int((y_pred[normal_mask] == 1).sum())
    n_norm = int(normal_mask.sum())
    rows.append({
        "fault_id": 0,
        "n_rows": n_norm,
        "detected": fp,
        "recall": fp / n_norm if n_norm else 0.0,  # 정상 행에서는 false positive rate
    })
    return rows


def run() -> pd.DataFrame:
    """모든 등록 모델을 평가하고 비교표를 저장."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = _load_clean()
    if len(df) < SEQ_LEN + 10:
        print(f"[compare] 데이터 부족({len(df)}행)")
        return pd.DataFrame()

    overall_rows = []
    per_fault_rows = []
    for name in MODEL_REGISTRY:
        print(f"[compare] 학습/평가: {name} ...")
        out = fit_and_score(df, model_name=name)
        if out.empty:
            continue
        y_true = (out["fault_id"] != 0).astype(int).to_numpy()
        y_pred = out["is_anomaly"].to_numpy()
        fault_id = out["fault_id"].to_numpy()

        m = _metrics_overall(y_true, y_pred)
        m["model"] = name
        overall_rows.append(m)

        for pf in _metrics_per_fault(fault_id, y_pred):
            pf["model"] = name
            per_fault_rows.append(pf)
        print(
            f"    precision={m['precision']:.3f} recall={m['recall']:.3f} "
            f"f1={m['f1']:.3f} (FP={m['fp']})"
        )

    overall = pd.DataFrame(overall_rows)[
        ["model", "precision", "recall", "f1", "tp", "fp", "fn", "tn"]
    ]
    per_fault = pd.DataFrame(per_fault_rows)[
        ["model", "fault_id", "n_rows", "detected", "recall"]
    ]

    overall.to_parquet(MODELS_DIR / "comparison.parquet", engine="pyarrow", index=False)
    per_fault.to_parquet(MODELS_DIR / "comparison_per_fault.parquet", engine="pyarrow", index=False)
    payload = {
        "overall": overall.to_dict(orient="records"),
        "per_fault": per_fault.to_dict(orient="records"),
    }
    (MODELS_DIR / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== 전체 비교 ===")
    print(overall.to_string(index=False))
    print("\n=== 결함 IDV별(fault_id=0 행은 오탐율) ===")
    print(per_fault.to_string(index=False))
    best = overall.sort_values("f1", ascending=False).iloc[0]["model"]
    print(f"\n[compare] 최고 F1 모델: {best}")
    return overall


if __name__ == "__main__":
    run()

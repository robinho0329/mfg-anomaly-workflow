"""② EDA·통계분석 — 정제 데이터의 탐색적 분석 + 통계 검정 + 시각화. (소유: mfg-eda)

산출(data/eda/):
  summary_stats.parquet  — 변수별 기술통계(인계 계약 유지)
  fault_tests.json       — 정상 vs 결함 검정(인계 계약 유지)
  corr_heatmap.png       — 분산 상위 변수 상관 히트맵
  dist_normal_vs_fault.png — 유의 변수별 정상/결함 분포 비교 박스플롯
  fault_counts.png       — 결함 IDV별 샘플 수 막대 차트
  timeseries_key_vars.png — 주요 변수 시계열 (결함 구간 음영)

대시보드(④)와 PPT가 PNG를 직접 임베드한다.
"""

# ── 표준 라이브러리 ───────────────────────────────────────────
import json
import os

# OpenBLAS 멀티스레드 메모리 할당 오류 방지 (Windows 환경)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# ── 서드파티 ─────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")  # 헤드리스 환경(서버/CI)에서 GUI 창 없이 렌더링
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

# ── 로컬 ─────────────────────────────────────────────────────
from config.settings import EDA_DIR, PROCESS_COLS
from src.pipeline.preprocess import load_stream, clean

# ── 공통 설정 ─────────────────────────────────────────────────
DPI = 110               # 저장 해상도
HEATMAP_TOP_N = 30      # 상관 히트맵에 쓸 분산 상위 변수 수
DIST_TOP_N = 12         # 분포 비교에 표시할 유의 변수 수
TS_TOP_N = 4            # 시계열 플롯에 쓸 키 변수 수
SIG_ALPHA = 0.05        # 유의수준 α

# ── 색상 팔레트 (한글 폰트 미사용; 영문·숫자 라벨만 사용) ─────
COLOR_NORMAL = "#4C72B0"   # 정상 파란색
COLOR_FAULT  = "#DD8452"   # 결함 주황색
COLOR_SHADE  = "#FFCCAA"   # 시계열 결함 구간 음영


# ──────────────────────────────────────────────────────────────
# 1. 기술통계
# ──────────────────────────────────────────────────────────────
def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """변수별 기술통계(평균/표준편차/왜도/첨도)."""
    desc = df[PROCESS_COLS].describe().T
    desc["skew"] = df[PROCESS_COLS].skew()
    desc["kurtosis"] = df[PROCESS_COLS].kurtosis()
    return desc


# ──────────────────────────────────────────────────────────────
# 2. 정상 vs 결함 통계 검정
# ──────────────────────────────────────────────────────────────
def fault_tests(df: pd.DataFrame) -> dict:
    """정상(fault_id==0) vs 결함 그룹 간 변수별 t검정·KS검정.

    결함 데이터가 없으면 빈 dict 반환(graceful degradation).
    """
    normal = df[df["fault_id"] == 0]
    faulty = df[df["fault_id"] != 0]
    if normal.empty or faulty.empty:
        return {}
    results = {}
    for c in PROCESS_COLS:
        t_stat, t_p = stats.ttest_ind(normal[c], faulty[c], equal_var=False)
        ks_stat, ks_p = stats.ks_2samp(normal[c], faulty[c])
        results[c] = {
            "t_stat": float(t_stat), "t_p": float(t_p),
            "ks_stat": float(ks_stat), "ks_p": float(ks_p),
        }
    return results


# ──────────────────────────────────────────────────────────────
# 3. 시각화 생성
# ──────────────────────────────────────────────────────────────
def _select_top_by_variance(df: pd.DataFrame, n: int) -> list[str]:
    """분산 기준 상위 n개 공정 변수 컬럼명 반환."""
    variances = df[PROCESS_COLS].var().sort_values(ascending=False)
    return variances.head(n).index.tolist()


def _select_top_by_ks(tests: dict, n: int) -> list[str]:
    """KS 통계량 기준 상위 n개 유의 변수 반환.
    유의 변수가 n개 미만이면 있는 만큼만 반환."""
    if not tests:
        return []
    sig = {k: v for k, v in tests.items() if v["ks_p"] < SIG_ALPHA}
    if not sig:
        sig = tests  # 유의한 게 없으면 전체에서 상위 추출
    top = sorted(sig.items(), key=lambda x: x[1]["ks_stat"], reverse=True)
    return [k for k, _ in top[:n]]


def make_corr_heatmap(df: pd.DataFrame) -> None:
    """상관 히트맵 PNG 저장 (분산 상위 HEATMAP_TOP_N 변수).

    seaborn 없이 matplotlib imshow로 구현 — 의존성 최소화.
    """
    top_cols = _select_top_by_variance(df, HEATMAP_TOP_N)
    corr = df[top_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    # 축 레이블 (짧은 변수명)
    short = [c.replace("xmeas_", "m").replace("xmv_", "v") for c in top_cols]
    ax.set_xticks(range(len(short)))
    ax.set_yticks(range(len(short)))
    ax.set_xticklabels(short, rotation=90, fontsize=7)
    ax.set_yticklabels(short, fontsize=7)

    # 컬러바
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Pearson r", fontsize=9)

    ax.set_title(
        f"Correlation Heatmap — Top {len(top_cols)} Variables by Variance",
        fontsize=11, pad=10
    )
    fig.tight_layout()
    out = EDA_DIR / "corr_heatmap.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[eda] corr_heatmap.png 저장 ({len(top_cols)}변수)")


def make_dist_comparison(df: pd.DataFrame, tests: dict) -> None:
    """유의 변수 상위 DIST_TOP_N개의 정상/결함 박스플롯 비교 PNG 저장."""
    top_cols = _select_top_by_ks(tests, DIST_TOP_N)
    if not top_cols:
        print("[eda] 유의 변수 없음 — dist_normal_vs_fault.png 생략")
        return

    n_cols = 4
    n_rows = int(np.ceil(len(top_cols) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = np.array(axes).flatten()

    normal_data = df[df["fault_id"] == 0]
    faulty_data = df[df["fault_id"] != 0]

    for idx, col in enumerate(top_cols):
        ax = axes[idx]
        short = col.replace("xmeas_", "XMEAS ").replace("xmv_", "XMV ")

        # 두 그룹 박스플롯
        bp = ax.boxplot(
            [normal_data[col].dropna().values, faulty_data[col].dropna().values],
            tick_labels=["Normal", "Fault"],  # matplotlib 3.9+ (labels → tick_labels)
            patch_artist=True,
            widths=0.5,
            medianprops=dict(color="black", linewidth=2),
        )
        bp["boxes"][0].set_facecolor(COLOR_NORMAL)
        bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_facecolor(COLOR_FAULT)
        bp["boxes"][1].set_alpha(0.7)

        # p값 표시
        ks_p = tests.get(col, {}).get("ks_p", float("nan"))
        p_str = f"KS p={ks_p:.2e}" if not np.isnan(ks_p) else ""
        ax.set_title(f"{short}\n{p_str}", fontsize=8)
        ax.set_ylabel("Value", fontsize=7)
        ax.tick_params(labelsize=7)

    # 남는 axes 숨기기
    for idx in range(len(top_cols), len(axes)):
        axes[idx].set_visible(False)

    # 범례
    patch_n = mpatches.Patch(color=COLOR_NORMAL, alpha=0.7, label="Normal (fault_id=0)")
    patch_f = mpatches.Patch(color=COLOR_FAULT, alpha=0.7, label="Fault (fault_id!=0)")
    fig.legend(handles=[patch_n, patch_f], loc="lower right", fontsize=9)

    fig.suptitle(
        f"Normal vs Fault Distribution — Top {len(top_cols)} Significant Variables (KS test)",
        fontsize=11, y=1.01
    )
    fig.tight_layout()
    out = EDA_DIR / "dist_normal_vs_fault.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[eda] dist_normal_vs_fault.png 저장 ({len(top_cols)}변수)")


def make_fault_counts(df: pd.DataFrame) -> None:
    """결함 IDV별 샘플 수 막대 차트 PNG 저장."""
    counts = df["fault_id"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 5))

    # fault_id==0 정상과 결함 구분 색상
    colors = [COLOR_NORMAL if fid == 0 else COLOR_FAULT for fid in counts.index]
    bars = ax.bar(counts.index.astype(str), counts.values, color=colors, edgecolor="white", width=0.7)

    # 막대 위 수치 표시
    for bar, cnt in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            str(cnt),
            ha="center", va="bottom", fontsize=9
        )

    ax.set_xlabel("Fault ID (0 = Normal)", fontsize=10)
    ax.set_ylabel("Sample Count", fontsize=10)
    ax.set_title("Sample Count by Fault ID (IDV)", fontsize=11, pad=8)

    # 범례
    patch_n = mpatches.Patch(color=COLOR_NORMAL, label="Normal (IDV=0)")
    patch_f = mpatches.Patch(color=COLOR_FAULT, label="Fault (IDV 1-20)")
    ax.legend(handles=[patch_n, patch_f], fontsize=9)

    ax.set_ylim(0, counts.max() * 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = EDA_DIR / "fault_counts.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[eda] fault_counts.png 저장 ({len(counts)}개 fault_id)")


def make_timeseries(df: pd.DataFrame, tests: dict) -> None:
    """주요 변수 시계열 플롯 (결함 구간 음영) PNG 저장.

    timestamp 컬럼이 없거나 단조증가하지 않으면 인덱스(행 번호) 축을 사용한다.
    """
    top_cols = _select_top_by_ks(tests, TS_TOP_N)
    if not top_cols:
        # 검정 결과 없으면 분산 상위 변수 사용
        top_cols = _select_top_by_variance(df, TS_TOP_N)
    if not top_cols:
        print("[eda] 시계열 플롯용 변수 없음 — timeseries_key_vars.png 생략")
        return

    # X축: timestamp 또는 행 인덱스
    has_ts = "timestamp" in df.columns and pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    x_vals = df["timestamp"] if has_ts else df.index
    x_label = "Timestamp" if has_ts else "Row Index"

    # 결함 구간 식별 (연속 구간으로 묶기)
    is_fault = df["fault_id"] != 0
    fault_spans: list[tuple] = []
    in_span = False
    start = None
    for i, flt in enumerate(is_fault):
        xv = x_vals.iloc[i] if has_ts else i
        if flt and not in_span:
            start = xv
            in_span = True
        elif not flt and in_span:
            fault_spans.append((start, xv))
            in_span = False
    if in_span:
        fault_spans.append((start, x_vals.iloc[-1] if has_ts else len(df) - 1))

    fig, axes = plt.subplots(len(top_cols), 1, figsize=(14, 3 * len(top_cols)), sharex=True)
    if len(top_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, top_cols):
        short = col.replace("xmeas_", "XMEAS ").replace("xmv_", "XMV ")
        ax.plot(x_vals, df[col].values, color=COLOR_NORMAL, linewidth=0.8, label=short)

        # 결함 구간 음영
        for (s, e) in fault_spans:
            ax.axvspan(s, e, alpha=0.25, color=COLOR_SHADE, zorder=0)

        ax.set_ylabel(short, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    # 마지막 subplot에만 X 레이블
    axes[-1].set_xlabel(x_label, fontsize=9)
    if has_ts:
        fig.autofmt_xdate(rotation=30, ha="right")

    # 공통 제목·범례
    fig.suptitle(
        f"Time Series — Top {len(top_cols)} Key Variables  [shaded: fault region]",
        fontsize=11, y=1.01
    )
    fault_patch = mpatches.Patch(color=COLOR_SHADE, alpha=0.6, label="Fault Region")
    fig.legend(handles=[fault_patch], loc="upper right", fontsize=9)

    fig.tight_layout()
    out = EDA_DIR / "timeseries_key_vars.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[eda] timeseries_key_vars.png 저장 (변수: {top_cols})")


def make_figures(df: pd.DataFrame, tests: dict) -> None:
    """4종 PNG 시각화를 일괄 생성해 EDA_DIR에 저장.

    df가 비어 있으면 조용히 종료(graceful degradation).
    """
    if df.empty:
        print("[eda] 데이터 없음 — PNG 생성 생략")
        return
    make_corr_heatmap(df)
    make_dist_comparison(df, tests)
    make_fault_counts(df)
    make_timeseries(df, tests)


# ──────────────────────────────────────────────────────────────
# 4. 통합 실행
# ──────────────────────────────────────────────────────────────
def run() -> dict:
    """EDA 파이프라인 실행 → 통계 JSON + PNG 일괄 저장."""
    EDA_DIR.mkdir(parents=True, exist_ok=True)

    # clean.parquet 우선 로드 (메인이 리프레시 보장), 없으면 스트림에서 재생성
    clean_path = EDA_DIR.parent / "processed" / "clean.parquet"
    try:
        if clean_path.exists():
            df = pd.read_parquet(clean_path, engine="pyarrow")
        else:
            from src.pipeline.preprocess import load_stream, clean  # 로컬 임포트 재사용
            df = clean(load_stream())
    except Exception as exc:
        print(f"[eda] 데이터 로드 실패: {exc!r} — 빈 DataFrame 사용")
        df = pd.DataFrame(columns=["timestamp", "fault_id", *PROCESS_COLS])

    if df.empty:
        print("[eda] 데이터 없음 — 수집(collect) 먼저 실행 필요")
        return {}

    # ── 기술통계 ──
    summary = summary_stats(df)
    summary.to_parquet(EDA_DIR / "summary_stats.parquet", engine="pyarrow")

    # ── 정상 vs 결함 검정 ──
    tests = fault_tests(df)
    (EDA_DIR / "fault_tests.json").write_text(
        json.dumps(tests, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[eda] 기술통계 {len(summary)}변수 + 검정 {len(tests)}변수 → {EDA_DIR}")

    # ── 시각화 PNG 생성 ──
    make_figures(df, tests)

    return {"summary": summary, "tests": tests}


if __name__ == "__main__":
    run()

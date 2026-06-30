"""④ 포트폴리오 PPT 자동 생성 — 워크플로우 산출물을 슬라이드로 조립. (소유: mfg-reporter)

각 단계 산출물(수집 규모·EDA 통계·이상탐지 성능)을 읽어 python-pptx로 덱을 만든다.
실행: python -m src.report.ppt
"""

import sqlite3

import pandas as pd

from config.settings import COLLECT_TABLE, DB_PATH, MODELS_DIR, REPORTS_DIR


def _collect_count() -> int:
    if not DB_PATH.exists():
        return 0
    with sqlite3.connect(DB_PATH) as conn:
        try:
            return conn.execute(f"SELECT COUNT(*) FROM {COLLECT_TABLE}").fetchone()[0]
        except sqlite3.OperationalError:
            return 0


def _detection_summary() -> dict:
    path = MODELS_DIR / "scores.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    return {
        "samples": len(df),
        "anomalies": int(df["is_anomaly"].sum()),
        "ratio": float(df["is_anomaly"].mean()),
    }


def build() -> "Path":
    """포트폴리오 덱 생성 → reports/portfolio.pptx 반환."""
    from pptx import Presentation
    from pptx.util import Inches, Pt

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    prs = Presentation()

    # 표지
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = "제조 공정 이상탐지 자동화 워크플로우"
    slide.placeholders[1].text = "TEP 다변량 시계열 · 수집→전처리/EDA→딥러닝 이상탐지→리포트"

    # 워크플로우 요약
    det = _detection_summary()
    bullet = slide = prs.slides.add_slide(prs.slide_layouts[1])
    bullet.shapes.title.text = "워크플로우 성과 요약"
    tf = bullet.placeholders[1].text_frame
    tf.text = f"① 수집: 누적 {_collect_count():,}행 (TEP 52변수 스트림)"
    for line in [
        "② 전처리·EDA: 정상/결함 통계 검정 자동화",
        f"③ 딥러닝 이상탐지: 샘플 {det.get('samples', 0):,} / 이상 {det.get('anomalies', 0):,}"
        f" ({det.get('ratio', 0) * 100:.1f}%)",
        "④ 대시보드·PPT: 산출물 자동 임베드",
    ]:
        p = tf.add_paragraph()
        p.text = line
        p.font.size = Pt(18)

    out = REPORTS_DIR / "portfolio.pptx"
    prs.save(out)
    print(f"[ppt] 포트폴리오 덱 생성 → {out}")
    return out


if __name__ == "__main__":
    build()

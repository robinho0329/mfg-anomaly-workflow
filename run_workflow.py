"""전체 워크플로우 오케스트레이터 — 4단계를 순서대로 실행.

순서 의존: collect → pipeline(preprocess→eda) → model → report
실행: python run_workflow.py [--collect-batches N] [--ppt]
"""

import argparse

from src.collect.scheduler import collect_batch
from src.pipeline import eda, preprocess
from src.models import detect


def main() -> None:
    parser = argparse.ArgumentParser(description="제조 이상탐지 워크플로우 실행")
    parser.add_argument("--collect-batches", type=int, default=10,
                        help="수집 배치 횟수(정상/결함 번갈아 주입)")
    parser.add_argument("--ppt", action="store_true", help="PPT 덱까지 생성")
    args = parser.parse_args()

    # ① 수집 — 정상 위주 + 일부 결함 주입으로 학습/검정 데이터 확보
    print("=== ① 수집 ===")
    for i in range(args.collect_batches):
        fault = 0 if i % 4 != 3 else (i % 20 + 1)  # 4회 중 1회 결함
        collect_batch(fault_id=fault)

    # ② 전처리·EDA
    print("=== ② 전처리·EDA ===")
    preprocess.run()
    eda.run()

    # ③ 딥러닝 이상탐지
    print("=== ③ 딥러닝 이상탐지 ===")
    detect.run()

    # ④ 리포트
    if args.ppt:
        print("=== ④ 포트폴리오 PPT ===")
        from src.report import ppt
        ppt.build()

    print("\n워크플로우 완료. 대시보드: streamlit run src/report/dashboard/app.py")


if __name__ == "__main__":
    main()

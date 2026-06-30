# CLAUDE.md — 제조 공정 이상탐지 자동화 워크플로우

> Claude가 이 프로젝트를 이해하고 효과적으로 작업하기 위한 가이드

## 프로젝트 개요

제조 공정(화학공정 TEP, Tennessee Eastman Process) 다변량 시계열에 대한
**딥러닝 기반 이상탐지 자동화 워크플로우**. 포트폴리오용 프로젝트로,
ML 모델링이 핵심이되 **수집 → 전처리/EDA/통계 → 딥러닝 이상탐지 → 대시보드/PPT**
전 과정을 자동화했다는 점을 일관되게 강조한다.

> 전력수급 프로젝트(`power-grid-anomaly-monitor`)와는 **완전히 별개**다.
> 도메인(제조 공정), 데이터(TEP 52변수), 모델(딥러닝 AE 계열)이 모두 다르다.

## 워크플로우 4단계 = 에이전트 4종

```
① 수집(collect) → ② 전처리·EDA(pipeline) → ③ 딥러닝 이상탐지(models) → ④ 대시보드·PPT(report)
   mfg-collector       mfg-eda                    mfg-model                    mfg-reporter
```

각 에이전트는 자기 단계를 **소유하고 유지보수**한다. 파일 겹침이 없어 병렬 안전.

| 에이전트 | 소유 영역 | 산출물 |
|---------|----------|--------|
| `mfg-collector` | `src/collect/`, `data/raw,db/` | `stream.db`, raw 스냅샷 |
| `mfg-eda` | `src/pipeline/`, `data/processed,eda/` | `clean.parquet`, 통계 JSON |
| `mfg-model` | `src/models/`, `data/models/` | `scores.parquet`, scaler |
| `mfg-reporter` | `src/report/`, `reports/` | Streamlit 앱, `portfolio.pptx` |

## 데이터 스키마 (TEP)

- 측정 변수 `xmeas_01..41` (XMEAS), 조작 변수 `xmv_01..11` (XMV) → 총 52개.
- `fault_id`: 0=정상, 1~20=IDV 결함 모드. `timestamp`: 3분 간격.
- 스키마 상수는 `config/settings.py`에 집중. 변경 시 전 단계 동기화 필수.

## 실행

```bash
# 의존성: 로컬 학습은 전체, 배포(대시보드)는 경량
pip install -r requirements-train.txt   # 로컬 전체(TF 포함)
# pip install -r requirements.txt       # Streamlit Cloud 배포용 경량(TF 미포함)

# 전체 워크플로우(수집 12배치 → 전처리/EDA → 탐지 → PPT)
python run_workflow.py --collect-batches 12 --ppt

# 단계별
python -m src.collect.scheduler --rows 20 --fault 0
python -m src.pipeline.preprocess && python -m src.pipeline.eda
python -m src.models.detect
streamlit run src/report/dashboard/app.py

# 테스트
pytest tests/
```

## 순차 의존 / 병렬 규칙

- **순차 필수**: collect → preprocess → eda → model → report (데이터 흐름 순서)
- **병렬 가능**: 서로 다른 단계의 *구조 개선* 작업(스키마 계약만 지키면)
- 스키마(`PROCESS_COLS`) 변경은 모든 단계에 전파 → 단일 작업으로 처리

## 코딩 규칙

- 경로 `pathlib.Path`, 임포트 순서 표준→서드파티→로컬, f-string, `random_state=42`.
- 주석/독스트링 한국어 우선, 식별자 영어. 구체적 예외 타입.
- Parquet 저장은 pyarrow 엔진. 데이터 없을 시 빈 DataFrame(graceful degradation).
- `.venv/`, `data/`, `*.pptx`는 커밋 대상 아님(.gitignore).

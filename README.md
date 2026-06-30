# 🏭 제조 공정 이상탐지 자동화 워크플로우

화학공정(Tennessee Eastman Process) 다변량 시계열에 대한 **딥러닝 이상탐지**를
**수집 → 전처리/EDA/통계 → 모델링 → 대시보드/PPT** 전 과정으로 자동화한 포트폴리오 프로젝트.

## 핵심 메시지

> 모델 하나가 아니라 **데이터가 흘러 들어와 인사이트와 리포트로 나오는 파이프라인 전체**를 자동화했다.
> 각 단계는 전담 에이전트가 소유·유지보수한다.

## 워크플로우

```
① 수집            ② 전처리·EDA·통계        ③ 딥러닝 이상탐지        ④ 대시보드·PPT
TEP 시뮬레이터  →  정제·검정(t/KS)      →  LSTM-AE 재구성오차   →  Streamlit + python-pptx
SQLite 적재        clean.parquet            scores.parquet           portfolio.pptx
```

| 단계 | 모듈 | 담당 에이전트 |
|------|------|--------------|
| ① 수집 | `src/collect/` | `mfg-collector` |
| ② 전처리·EDA | `src/pipeline/` | `mfg-eda` |
| ③ 딥러닝 이상탐지 | `src/models/` | `mfg-model` |
| ④ 대시보드·PPT | `src/report/` | `mfg-reporter` |

## 빠른 시작

```bash
# 로컬 전체 워크플로우(딥러닝 학습 포함) — TensorFlow 등 전체 의존성
pip install -r requirements-train.txt

# 전체 워크플로우 실행
python run_workflow.py --collect-batches 12 --ppt

# 대시보드
streamlit run src/report/dashboard/app.py

# 테스트
pytest tests/
```

> `requirements.txt` 는 **Streamlit Cloud 배포용 경량 의존성**(TF 미포함),
> `requirements-train.txt` 는 **로컬 학습용 전체 의존성**이다.

## Streamlit Cloud 배포

대시보드는 사전계산 산출물(`data/models/scores.parquet` 등)만 읽으므로 **TensorFlow 없이 경량 배포**된다.

1. share.streamlit.io 접속 → GitHub repo 선택
2. **Main file path**: `src/report/dashboard/app.py`
3. Deploy (루트 `requirements.txt` 가 자동 적용 — 경량)

산출물 갱신 시: 로컬에서 워크플로우를 재실행한 뒤
`git add -f data/models/scores.parquet` 로 다시 커밋한다(데이터는 기본 .gitignore 대상).

## 데이터

- **시뮬레이터 스트림**(기본): TEP 구조(52변수, 정상+20결함)를 재현해 '살아있는 수집' 데모.
- **실제 TEP 데이터**: Rieth et al. 2017(Harvard Dataverse)을 `data/raw/`에 드롭하면 그대로 사용.

## 기술 스택

pandas · numpy · scipy · scikit-learn · TensorFlow(LSTM-AE) · Streamlit · python-pptx

## 모델

준지도 이상탐지 — 정상 운전만으로 LSTM 오토인코더 학습 → 재구성오차 임계값으로 이상 판정.
확장 슬롯: VAE / Transformer-AE / USAD (동일 인터페이스로 형제 모듈 추가).

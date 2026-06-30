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
pip install -r requirements.txt

# 전체 워크플로우 실행
python run_workflow.py --collect-batches 10 --ppt

# 대시보드
streamlit run src/report/dashboard/app.py

# 테스트
pytest tests/
```

## 데이터

- **시뮬레이터 스트림**(기본): TEP 구조(52변수, 정상+20결함)를 재현해 '살아있는 수집' 데모.
- **실제 TEP 데이터**: Rieth et al. 2017(Harvard Dataverse)을 `data/raw/`에 드롭하면 그대로 사용.

## 기술 스택

pandas · numpy · scipy · scikit-learn · TensorFlow(LSTM-AE) · Streamlit · python-pptx

## 모델

준지도 이상탐지 — 정상 운전만으로 LSTM 오토인코더 학습 → 재구성오차 임계값으로 이상 판정.
확장 슬롯: VAE / Transformer-AE / USAD (동일 인터페이스로 형제 모듈 추가).

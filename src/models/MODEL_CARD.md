# Model Card — 제조 공정(TEP) 딥러닝 이상탐지

> 워크플로우 ③ 딥러닝 이상탐지 단계(mfg-model) 산출 모델 카드.
> 갱신 기준 데이터: `data/processed/clean.parquet` **1,940행** (실제 TEP 벤치마크). 재현 시드 `RANDOM_STATE=42`.

## 1. 개요

화학공정 Tennessee Eastman Process(TEP)의 52개 다변량 시계열(`xmeas_01..41`, `xmv_01..11`)에 대한
**준지도 이상탐지**. 정상 운전 구간만으로 시퀀스 오토인코더를 학습하고, 정상 대비 재구성오차의
**마할라노비스 거리**가 큰 구간을 이상으로 판정한다(이상=양성, 정답은 `fault_id != 0`).

- 입력: `data/processed/clean.parquet` (mfg-eda 산출, read-only)
- 출력: `data/models/scores.parquet` (`timestamp, fault_id, anomaly_score, threshold, is_anomaly`)
- 비교 산출: `comparison.parquet`, `comparison_per_fault.parquet`, `comparison.json`, `roc_pr_curves.png`

## 2. 데이터 구성 (1,940행 · 실제 TEP 벤치마크)

출처: **Tennessee Eastman Process** 표준 벤치마크(Downs & Vogel 1993, Braatz 배포판)의
정상 운전(`d00`) + 결함 시뮬레이션(`d01..d18`)을 프로젝트 스키마(52변수)로 적재.

| 구분 | 행 수 | 비고 |
|------|------|------|
| 정상(fault_id=0) | 1,460 | `d00`(500, 전치) + `d00_te`(960). 학습/임계값 보정 기준 |
| 결함(IDV) | 480 | 10종: 1, 2, 3, 4, 6, 7, 8, 12, 14, 18 (각 48행) |

- 시퀀스 윈도우로 평가. 정상:결함 ≈ 3.0:1 (정상 위주 구성).
- **합성이 아닌 실제 벤치마크** → 성능 수치가 문헌 baseline과 비교 가능.

## 3. 모델 구조 (3종, 동일 인터페이스)

`build / fit / reconstruction_error / reconstruction_error_vector` 인터페이스를 공유하며,
TensorFlow 지연 임포트로 미설치 환경에서도 import가 깨지지 않는다.

| 모델 | 파일 | 구조 요약 |
|------|------|-----------|
| **LSTM-AE** | `lstm_ae.py` | LSTM 인코더(latent=16, dropout) → RepeatVector → LSTM 디코더 → TimeDistributed Dense |
| **VAE** | `vae.py` | LSTM 인코더 → (μ, logσ²) 재매개변수화 샘플링(β=0.5 KL) → LSTM 디코더 |
| **Transformer-AE** ★기본 | `transformer_ae.py` | 입력 임베딩 + 학습형 위치인코딩 → 멀티헤드 셀프어텐션 인코더 블록 ×2(잔차) → Dense |

- 공통 학습: 순수 정상 시퀀스(윈도우 전체 정상)만 사용, `EarlyStopping`(val_loss, patience=8), dropout=0.1.
- 결정성: `tf_seed.enable_determinism()`(numpy/python/TF 시드 + `enable_op_determinism()`)으로 실행 간 동일 결과.

## 4. 점수·임계값 방법론

1. **점수**: 시퀀스 마지막 스텝의 **변수별 재구성오차 벡터**에 대한 마할라노비스 거리.
   - 정상 오차 벡터의 평균·공분산(Ledoit-Wolf 축소 추정) 기준.
   - 일부 변수만 이상한 결함과, 재구성오차가 정상보다 **비정상적으로 작은** 결함(부호 무관)도 분리.
2. **임계값**: 순수 정상 시퀀스 점수의 분위 `ANOMALY_QUANTILE=0.995` × 여유배수 `THRESHOLD_MARGIN=1.10`.
3. **평활**: 행 점수 rolling median(`SCORE_SMOOTH_WINDOW=5`)으로 단발 스파이크 오탐 억제.

> 마지막 스텝 기준 점수는 결함 직후 회복(경계) 구간의 오탐을 구조적으로 억제한다.
> 윈도우 평균 대신 변수별 오차 + 마할라노비스로 ROC-AUC를 실 TEP에서 0.92 수준까지 확보.

## 5. 평가 결과 (실 TEP 1,940행, 이상=양성)

| 모델 | Precision | Recall | F1 | ROC-AUC | PR-AUC | FP | FN |
|------|-----------|--------|-----|---------|--------|----|----|
| **Transformer-AE** ★ | 0.981 | **0.748** | **0.849** | **0.920** | **0.894** | 7 | 121 |
| VAE | **1.000** | 0.700 | 0.824 | 0.873 | 0.844 | **0** | 144 |
| LSTM-AE | 0.983 | 0.702 | 0.819 | 0.876 | 0.860 | 6 | 143 |

- **기본 모델: Transformer-AE** — F1·ROC-AUC·PR-AUC 최고.
- **VAE**는 오탐(FP) **0건 / 정밀도 1.00** — 오경보 비용이 큰 운영에서 매력적.
- ROC/PR 커브: `data/models/roc_pr_curves.png`.

## 6. 결함 IDV별 탐지율 (Transformer-AE 기준)

| IDV | 1 | 2 | 3 | 4 | 6 | 7 | 8 | 12 | 14 | 18 |
|-----|---|---|---|---|---|---|---|----|----|----|
| 탐지율 | 0.96 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.52 | 1.00 | 0.00 |

- **잘 탐지(≈1.00)**: IDV 1, 2, 4, 6, 7, 8, 14 — 재구성오차에 뚜렷한 패턴 변화(스텝·랜덤 결함).
- **부분 탐지**: IDV 12 (0.52).
- **미탐(0.00)**: IDV 3, 18 — 실제 TEP에서도 잘 알려진 난탐지 결함(정상과 통계적으로 거의 구별 불가).

## 7. 임계값 트레이드오프

- 임계값 여유배수 `THRESHOLD_MARGIN`↑ → 과탐(FP)↓·미탐(FN)↑.
  운영 정책상 과탐 민감하면 margin을 높이고(예: VAE는 이미 FP=0), 미탐 민감하면 낮춘다.
- 미탐 결함(IDV 3/18)은 margin을 낮춰도 잡히지 않으므로, 추가 탐지는
  AE 재구성이 아닌 다른 신호(통계적 관리도, 도메인 규칙)와의 결합이 필요하다.

## 8. 한계 및 주의

- **준지도**: 정상 데이터만으로 학습. 정상에 미관측 운전 모드가 섞이면 오탐 가능.
- **실 TEP 벤치마크**: 표준 Tennessee Eastman Process 시뮬레이션 데이터(학술 벤치마크). 실제 가동 공장의 라이브 스트림은 아님.
- **표본 규모**: 결함당 48행으로 per-fault 탐지율은 표본 변동에 다소 민감(전체 1,940행).
- **AE 한계**: 재구성오차에 흔적을 남기지 않는 결함(IDV 3/18)은 원리적으로 미탐.
- **경계 라벨**: 결함→정상 전환 구간은 라벨이 정상이라도 점수가 일시 상승할 수 있음(평활로 완화).

## 9. 재현 명령

```bash
# 가상환경 + 결정성 환경변수
cd /c/Users/xcv54/workspace/mfg-anomaly-workflow
export TF_CPP_MIN_LOG_LEVEL=3 PYTHONHASHSEED=42

# 탐지(기본 Transformer-AE) → scores.parquet
./.venv/Scripts/python.exe -m src.models.detect

# 모델 비교 → comparison*.parquet/json + roc_pr_curves.png
./.venv/Scripts/python.exe -m src.models.compare

# 단위 테스트
./.venv/Scripts/python.exe -m pytest tests/test_models.py -q
```

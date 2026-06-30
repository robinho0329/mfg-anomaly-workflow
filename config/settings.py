"""프로젝트 전역 설정 — 경로 상수 / 하이퍼파라미터 / TEP 스키마 집중 관리.

워크플로우 4단계(collect → pipeline → models → report)가 모두 이 모듈을 참조한다.
경로·스키마를 바꿀 때는 여기만 수정하고, 영향받는 단계 에이전트에게 인계한다.
"""

from pathlib import Path

# ── 기준 경로 ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"            # ① 수집 원본 (Parquet/CSV)
PROCESSED_DIR = DATA_DIR / "processed"  # ② 전처리 산출 (Parquet)
EDA_DIR = DATA_DIR / "eda"            # ② EDA/통계 산출 (PNG/JSON)
MODELS_DIR = DATA_DIR / "models"      # ③ 학습 모델/스코어 (H5/joblib/Parquet)
DB_PATH = DATA_DIR / "db" / "stream.db"  # ① 수집 스트림 적재 SQLite
REPORTS_DIR = BASE_DIR / "reports"    # ④ 대시보드/PPT 산출

# ── 재현성 ────────────────────────────────────────────────
RANDOM_STATE = 42

# ── TEP(Tennessee Eastman Process) 스키마 ─────────────────
# 측정 변수 XMEAS 1~41, 조작 변수 XMV 1~11 → 총 52개 공정 변수.
N_XMEAS = 41
N_XMV = 11
XMEAS_COLS = [f"xmeas_{i:02d}" for i in range(1, N_XMEAS + 1)]
XMV_COLS = [f"xmv_{i:02d}" for i in range(1, N_XMV + 1)]
PROCESS_COLS = XMEAS_COLS + XMV_COLS  # 52개 피처

# 결함 모드 IDV(Disturbance) 1~20 (0 = 정상 운전)
N_FAULTS = 20
FAULT_IDS = list(range(0, N_FAULTS + 1))

# 샘플링 주기(분) — TEP 표준은 3분 간격
SAMPLE_INTERVAL_MIN = 3

# ── 수집(collect) 설정 ────────────────────────────────────
COLLECT_BATCH_ROWS = 20      # 스케줄러 1회 실행 시 적재 행 수
COLLECT_TABLE = "tep_stream"  # SQLite 적재 테이블명

# ── 모델(models) 설정 ─────────────────────────────────────
SEQ_LEN = 20                 # 시계열 윈도우 길이(스텝)
AE_LATENT_DIM = 16           # 오토인코더 잠재 차원
AE_EPOCHS = 80               # 최대 epoch (조기종료로 자동 단축)
AE_BATCH = 64
AE_VAL_SPLIT = 0.15          # 학습 중 검증 분할 비율(조기종료 기준)
AE_PATIENCE = 8              # EarlyStopping 인내(epoch)
AE_DROPOUT = 0.1             # 과적합 억제용 드롭아웃

# 임계값/점수 산정 ─ 과탐 억제 핵심 영역
ANOMALY_QUANTILE = 0.995     # 순수 정상 재구성오차 분위수 → 임계값
THRESHOLD_MARGIN = 1.15      # 분위 임계값에 곱하는 안전 여유 배수(노이즈 마진)
SCORE_SMOOTH_WINDOW = 5      # 행 점수 평활(rolling median) 윈도우(과탐 억제)

# 신규 모델 하이퍼파라미터
VAE_BETA = 0.5               # VAE KL 가중치(β-VAE)
TRANSFORMER_HEADS = 4        # Transformer-AE 멀티헤드 어텐션 헤드 수
TRANSFORMER_FF_DIM = 64      # Transformer-AE 피드포워드 차원

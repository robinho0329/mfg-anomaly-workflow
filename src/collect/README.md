# ① 수집 단계 (mfg-collector)

TEP(Tennessee Eastman Process) 52변수 다변량 스트림을 생성/적재하는 수집 단계.
시뮬레이터 합성 또는 `data/raw/` 실데이터를 SQLite(`data/db/stream.db`)에 누적한다.

## 구성

| 파일 | 역할 |
|------|------|
| `simulator.py` | TEP 유사 정상/결함(IDV) 스트림 합성 (seed 고정 재현성) |
| `tep_loader.py` | 실제 TEP 데이터 드롭인 어댑터(컬럼 매핑/결측보간) |
| `scheduler.py` | 배치 수집·SQLite 적재·연속 timestamp·결정적 백필 |

## 출력 스키마 (인계 계약 — 변경 금지)

```
timestamp, fault_id, xmeas_01..41, xmv_01..11
```

## 사용법

```bash
# 1회 배치(정상 20행)
python -m src.collect.scheduler --rows 20 --fault 0

# 결함 IDV 주입(예: IDV 6)
python -m src.collect.scheduler --rows 20 --fault 6

# 결정적 백필(BACKFILL_PLAN 기반 DB 확대 — 정상:결함 ≈ 3.5:1, IDV 다양)
python -m src.collect.scheduler --backfill
```

## 실데이터 드롭인 (tep_loader)

`data/raw/` 에 실제 TEP 데이터(예: Rieth et al. 2017, Harvard Dataverse)를 두면
시뮬레이터 대신 **그 데이터를 우선 적재**한다(없으면 시뮬레이터 폴백).

- 지원 포맷: CSV / Parquet. 파일명 글롭은 `config.settings.TEP_REAL_GLOBS`
  (예: `TEP_FaultFree_Training.csv`, `TEP_Faulty_Training.csv`).
  시뮬레이터 스냅샷(`snapshot_*`)은 실데이터로 취급하지 않는다.
- 컬럼 매핑(대소문자/zero-pad 유연): `faultNumber→fault_id`,
  `xmeas_1..41→xmeas_01..41`, `xmv_1..11→xmv_01..11`.
- 결측/문자값은 숫자 변환 후 변수별 중앙값으로 보간. 포맷 불일치 시 graceful 폴백.
- 실데이터는 오프셋(`collect_meta.real_offset`) 기반으로 배치마다 이어서 소진된다.

## 상시 수집 스케줄러 등록

스크립트는 venv 파이썬으로 **1회 배치 수집**을 돈다. OS 스케줄러로 주기 실행한다.

### Windows — Task Scheduler (schtasks)

```bat
:: 매 30분마다 정상 20행 수집 (절대경로 사용)
schtasks /Create /TN "mfg-collect" /SC MINUTE /MO 30 ^
  /TR "C:\Users\xcv54\workspace\mfg-anomaly-workflow\.venv\Scripts\python.exe -m src.collect.scheduler --rows 20 --fault 0" ^
  /SD 2026/06/30 /ST 00:00

:: 등록 확인 / 즉시 실행 / 삭제
schtasks /Query /TN "mfg-collect"
schtasks /Run   /TN "mfg-collect"
schtasks /Delete /TN "mfg-collect" /F
```

> 주의: Task Scheduler는 작업 디렉터리를 프로젝트 루트로 보장하지 않는다.
> "시작 위치(start-in)"가 필요하면 GUI(작업 스케줄러)에서
> 시작 위치를 `C:\Users\xcv54\workspace\mfg-anomaly-workflow` 로 지정하거나,
> `cmd /c "cd /d <루트> && .venv\Scripts\python.exe -m src.collect.scheduler ..."` 형태로 감싼다.

### Linux/macOS — cron (참고용)

```cron
# 매 30분마다 정상 20행 수집
*/30 * * * * cd /path/to/mfg-anomaly-workflow && ./.venv/bin/python -m src.collect.scheduler --rows 20 --fault 0
```

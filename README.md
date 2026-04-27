# 해양사고 위험 예측

공개 해양사고 통계, 격자 데이터, 해양 기상 데이터를 결합해 격자×시간 단위 사고 위험도를 예측하는 데모 프로젝트입니다. 슈어소프트테크 인턴십 중 해양경찰청 AI융복합 과제로 수행한 작업을, 회사 보안상 사내 코드·데이터를 외부에 공개할 수 없어 공개 데이터 기반으로 가볍게 재현했습니다. 데이터 적재부터 피처 생성, LightGBM 학습, SHAP 분석, FastAPI API, Streamlit 대시보드까지 하나의 흐름으로 구성했습니다.

## 개요

| 항목 | 내용 |
| --- | --- |
| 분야 | 해양 안전 위험 예측 |
| 과제 | 공간 격자와 시간 단위 사고 발생 확률 예측 |
| 데이터 | 공개 사고 통계, level-4 격자, 해양 기상 API |
| 모델 | LightGBM 분류 모델 |
| 설명가능성 | SHAP 기반 feature attribution |
| 서빙 | FastAPI, Docker, Streamlit |
| 역할 | 슈어소프트테크 인턴십 중 해양경찰청 AI융복합 과제로 수행한 작업을, 회사 보안상 사내 코드·데이터를 외부에 공개할 수 없어 공개 데이터 기반으로 가볍게 재현한 데모 |

## 파이프라인

```text
사고 기록 + 격자 데이터 + 기상 API
  -> 공간 / 시간 정규화
  -> 격자×시간 피처 테이블 생성
  -> LightGBM 학습
  -> SHAP 분석
  -> FastAPI 예측 API
  -> Streamlit 위험도 대시보드
```

## 저장소 구성

```text
.
├── configs/
│   └── default.yaml
├── notebooks/
│   └── 01_demo_pipeline.ipynb
├── src/marine_accident_risk/
│   ├── data/        # 사고, 격자, 기상 API loader
│   ├── features/    # 피처 생성
│   ├── models/      # 학습과 추론
│   ├── api/         # FastAPI app
│   └── dashboard/   # Streamlit app
├── Dockerfile
└── pyproject.toml
```

## 실행

```bash
pip install -e .

export NMPNT_SERVICE_KEY="<marine-weather-api-key>"

python -m marine_accident_risk.cli weather-fetch \
  --start 2024-01-01 --end 2024-01-31 \
  --stations 994401578,994401588,994401597

python -m marine_accident_risk.cli build-dataset \
  --accidents data/raw/accidents.xlsx \
  --grid data/raw/level4.csv \
  --bbox 34.5,128.5,35.5,129.5

python -m marine_accident_risk.cli train --config configs/default.yaml
uvicorn marine_accident_risk.api.app:app --port 8000
streamlit run src/marine_accident_risk/dashboard/app.py
```

## Multi-threshold evaluation

운영 시 알림 정책을 정하려면 점수 임계값별 trade-off 를 함께 보아야 한다고 보고, OOF 예측을 활용해 임계값 0.1~0.9 구간의 precision / recall / specificity / F1 / Youden's J / 비용 가중치(FN=5, FP=1)를 일괄로 계산했습니다. 결과는 `reports/threshold/` 에 정리됩니다 (`threshold_report.md`, `threshold_table.csv`, `roc_pr_curves.png`).

| 기준 | threshold | precision | recall | F1 |
| --- | --- | --- | --- | --- |
| F1 최대 | 0.30 | 0.682 | 0.748 | 0.714 |
| Youden's J 최대 | 0.20 | 0.576 | 0.913 | 0.706 |
| Cost-weighted (FN>FP) | 0.20 | 0.576 | 0.913 | 0.706 |

OOF AUC 0.946 / PR-AUC 0.742 기준으로, F1 만 보면 0.30 부근이 균형점이지만 사고 미탐 비용이 오탐 비용보다 큰 운영 환경(FN cost 5x)에서는 0.20 이 비용 합을 가장 낮게 만들어 알림 임계값 후보로 적절합니다. 더 보수적인 운영(오경보 억제)이 필요한 경우 0.30~0.40 구간에서 precision 0.68~0.74 수준을 확보할 수 있어 정책에 따라 선택지를 두는 것을 가정합니다.

실행:

```bash
python scripts/run_threshold_analysis.py --config configs/default.yaml \
    --cost-fn 5 --cost-fp 1
```

## 공개 범위

원본 데이터, 전처리 산출물, 학습된 모델, SHAP cache는 포함하지 않습니다.

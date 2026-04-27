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

## 공개 범위

원본 데이터, 전처리 산출물, 학습된 모델, SHAP cache는 포함하지 않습니다.

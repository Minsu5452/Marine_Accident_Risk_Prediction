# Marine Accident Risk Prediction

> 격자 × 시간 단위로 한반도 인근 해역의 해양사고 발생 확률을 예측하는 데이터 파이프라인 + 모델 + 서빙 데모.

## ⚠️ 공개 범위 (Disclaimer)

본 저장소는 **재직 중 수행한 정부 대형 R&D 과제 (AI융복합)** 의 일부 흐름을 **본인이 별도로 재현한 데모**입니다.

- 회사·고객사 자산(원본 코드, 내부 데이터, 모델 가중치, 평가 수치)은 보안 정책상 **공개하지 않습니다**.
- 본 레포에서 사용하는 데이터·코드·모델은 모두 **공개 OPEN API + 공개 GIS 격자 + 공개 사고 통계** 만으로 구성된 재현 환경입니다.
- 실제 프로젝트는 더 큰 규모(다채널 외부 API, 통항량/AIS, 격자별 SHAP, 행정구역 단위 대시보드 등)이며, 본 레포는 그중 핵심 흐름만 일부 시연합니다.

## Snapshot

| 항목 | 내용 |
| --- | --- |
| 도메인 | 해양 안전, 해양사고 위험 예측 |
| 과제 | 격자 × 1시간 단위 해양사고 발생 확률 추정 |
| 데이터 | 해양수산부 GIS 기반 해양사고 (2017–2024) + 0.025° 격자 (level4) + 측위정보원 OPEN API 기상 |
| 모델 | LightGBM 분류, 격자/기상/시간/선박용도 피처 |
| 설명가능성 | SHAP value 기반 격자별 위험요인 분해 |
| 배포 | FastAPI 추론 API + Docker 이미지 + 격자 대시보드 (Streamlit) |
| 원본 프로젝트 | SureSoftTech AX응용기술팀 / 정부 R&D (민·군·경 협력) 2-1 세부과제 |
| 본인 역할 (원본) | 데이터 수집 자동화, 통계/SHAP 분석, 예측 모델 개발, FastAPI/Docker 배포 |

## 핵심 흐름

```
[NMPNT 측위정보원 OPEN API]   [GIS 사고 xlsx]   [level4 격자 csv]
        │                           │                     │
        └──── ETL (시간 단위 리샘플) ─┴── 격자 매핑 ─────────┘
                                    │
                              feature build
                          (기상×격자×시간×선박)
                                    │
                          LightGBM 학습 / SHAP
                                    │
                FastAPI ──── Docker ──── 격자 대시보드
```

## Repository Structure

```
src/marine_accident_risk/
  data/         # OPEN API 클라이언트, xlsx/csv 로더, 격자 매퍼
  features/     # 시간/기상/선박 파생 피처
  models/       # LightGBM 학습/추론, SHAP
  api/          # FastAPI 앱
  dashboard/    # Streamlit 격자 위험도 시각화
notebooks/      # 단계별 데모 노트북 (EDA → 학습 → SHAP → 추론)
configs/        # YAML 설정 (격자 영역, 학습 하이퍼, API 엔드포인트)
scripts/        # 데이터 수집/학습/추론 CLI 진입점
data/           # raw / processed / external (gitignored)
```

## 실행 (재현 시)

```bash
# 1) 의존성
pip install -e .

# 2) 환경변수
export NMPNT_SERVICE_KEY="<측위정보원 인증키>"

# 3) 한 달치 기상 수집 (예시)
python -m marine_accident_risk.cli weather-fetch \
  --start 2024-01-01 --end 2024-01-31 \
  --stations 994401578,994401588,994401597

# 4) 격자×시간 학습 데이터셋 생성
python -m marine_accident_risk.cli build-dataset \
  --accidents data/raw/accidents.xlsx \
  --grid data/raw/level4.csv \
  --bbox 34.5,128.5,35.5,129.5  # 부산 인근 데모

# 5) 학습
python -m marine_accident_risk.cli train --config configs/default.yaml

# 6) 추론 API
uvicorn marine_accident_risk.api.app:app --port 8000

# 7) 대시보드
streamlit run src/marine_accident_risk/dashboard/app.py
```

## Public Scope / Out of Scope

**포함**
- 공개 OPEN API 기반 1시간 단위 기상 ETL
- GIS 사고 데이터 → 격자 매핑 + 시간 정규화
- LightGBM 학습/검증/추론 루프
- SHAP 기반 격자별 위험요인 설명
- FastAPI 추론 + Docker
- Streamlit 간이 대시보드

**제외 (보안/규모)**
- 회사 자산 코드, 내부 데이터, 내부 가중치, 평가 수치
- AIS 통항량 (실프로젝트 사용, 본 레포는 mock)
- 행정구역(경찰서 관할 등) 단위 집계
- 격자 전영역 학습 (본 레포는 데모 격자만 사용; 부산 인근 bbox)

## Links
- 원본 활동: SureSoftTech AX응용기술팀 (2025.06–2025.11)
- 사용 OPEN API
  - 국립해양측위정보원 해양기상 정보서비스
  - 해양수산부 GIS 기반 해양사고 통계

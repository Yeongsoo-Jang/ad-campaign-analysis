다음은 업데이트된 README.md 내용입니다:

```markdown
# 광고 캠페인 효과 분석 대시보드

## 프로젝트 개요
이 프로젝트는 광고 캠페인 데이터를 분석하고 시각화하는 대시보드를 제공합니다. 머신러닝 기반 회귀 분석을 통해 ROAS(투자수익률)에 영향을 미치는 주요 요인을 파악하고, 예산 최적화 기능을 통해 효과적인 광고 캠페인 전략을 수립할 수 있습니다.

## 주요 기능
- **데이터 시각화**: 다양한 광고 지표의 시계열 트렌드 분석
- **회귀 분석**: ROAS에 영향을 미치는 주요 요인 식별 및 예측 모델 구축
- **그룹별 성과 비교**: 플랫폼, 크리에이티브 유형, 타겟팅별 성과 비교
- **예산 최적화**: 캠페인별 권장 예산 조정 비율 및 신뢰도 점수 제공
- **ROAS 예측 모델**: 입력값에 따른 ROAS 예측 및 최적 설정 제안
- **사용자 정의 KPI**: 커스텀 KPI 생성 및 추적 기능
- **알림 시스템**: 중요 지표 임계값 모니터링 및 알림
- **HTML 보고서 생성**: 자동화된 HTML 보고서 생성 및 이메일 전송
- **API 연동**: 광고 플랫폼 API 연동 기능 (Google Ads, Facebook Ads, Naver 광고)

## 시스템 구조
```
ad-campaign-analysis/
├── app.py                  # 메인 대시보드 애플리케이션
├── config/
│   └── config.yaml         # 시스템 설정 파일
├── data/
│   └── campaign_data.csv   # 광고 캠페인 데이터
├── logs/                   # 로그 파일 디렉토리
├── models/                 # 훈련된 모델 저장 디렉토리
├── reports/                # 생성된 보고서 저장 디렉토리
├── templates/              # 보고서 템플릿 디렉토리
├── alerts/                 # 알림 기록 저장 디렉토리
├── kpis/                   # KPI 정의 및 기록 디렉토리
├── feedback/               # 사용자 피드백 저장 디렉토리
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # 데이터 로드 및 API 연동
│   ├── preprocessing.py    # 데이터 전처리
│   ├── modeling.py         # 머신러닝 모델 훈련 및 예측
│   ├── reporter.py         # HTML 보고서 생성 및 전송
│   ├── alert_system.py     # 알림 시스템
│   ├── custom_kpi.py       # 사용자 정의 KPI 관리
│   ├── budget_optimizer.py # 예산 최적화 
│   ├── ab_test.py          # A/B 테스트 분석
│   └── insight_generator.py # AI 인사이트 생성
└── requirements.txt        # 필요한 패키지 목록
```

## 설치 방법
1. 이 저장소를 클론합니다:
   ```
   git clone https://github.com/your-username/ad-campaign-analysis.git
   cd ad-campaign-analysis
   ```
2. 필요한 패키지를 설치합니다:
   ```
   pip install -r requirements.txt
   ```
3. 설정 파일을 확인합니다:
   ```
   mkdir -p config
   # config.yaml 파일 생성 또는 편집
   ```

## 사용 방법
1. 대시보드를 실행합니다:
   ```
   streamlit run app.py
   ```
2. 웹 브라우저에서 다음 주소로 접속합니다:
   ```
   http://localhost:8501
   ```
3. 대시보드에서 다음 기능들을 사용할 수 있습니다:
   - 데이터 필터링 (날짜, 플랫폼, 캠페인, 타겟 등)
   - 시계열 분석 및 시각화
   - 회귀 분석 및 예측 모델링
   - 캠페인별 예산 최적화 추천
   - 사용자 정의 KPI 관리
4. 모든 모듈은 기본적으로 import 되어 있으며, HTML 보고서 생성 기능이 활성화되어 있습니다:
   ```python
   # app.py에 다음 import가 포함되어 있음
   from src.modeling import ModelTrainer
   from src.reporter import ReportGenerator  # HTML 보고서 생성 기능
   from src.alert_system import AlertSystem
   from src.custom_kpi import CustomKPI
   from src.budget_optimizer import BudgetOptimizer
   from src.preprocessing import DataPreprocessor
   from src.ab_test import ABTestAnalyzer
   from src.insight_generator import InsightGenerator
   ```

## 데이터 전처리 기능
데이터 품질을 향상시키기 위해 다음 전처리 기능이 자동으로 적용됩니다:
- **이상치 제거**: IQR(사분위 범위) 기반 이상치 탐지 및 처리
- **정규화**: Robust 스케일링 방법으로 수치형 특성 정규화
- **파생변수 생성**: 효율성 지표(ROI), 복합 지표(유효 전환율), 시간 기반 특성 등 자동 생성

## 예산 최적화 기능
이 대시보드의 핵심 기능 중 하나는 캠페인별 예산 최적화입니다:
- **권장 예산 조정 비율**: 각 캠페인의 ROAS 효율성을 기반으로 예산 조정 비율을 제안합니다.
- **신뢰도 점수**: 데이터 포인트 수, 지출 규모, 성과 안정성 등을 고려한 종합 신뢰도 점수를 제공합니다.
- **직관적 시각화**: 색상 구분과 명확한 수치로 쉽게 의사결정에 활용할 수 있습니다.

## ROAS 예측 및 최적 설정 제안
대시보드는 머신러닝 모델을 통해 지출, 노출, 클릭수 등의 입력값에 따른 ROAS를 예측하고, 다음과 같은 최적 설정을 제안합니다:
- 최적 플랫폼 (예: Naver)
- 최적 크리에이티브 유형 (예: 텍스트)
- 최적 타겟 연령 (예: 35-44)
- 최적 타겟 성별 (예: 남성)
- 최적 캠페인 (예: 캠페인_10)

## AI 인사이트 생성
데이터 분석 결과를 바탕으로 자동화된 인사이트를 제공합니다:
- 가장 효율적인 캠페인과 개선이 필요한 캠페인 식별
- 시간에 따른 성과 추세 파악
- 상관관계 분석 및 주요 영향 요인 도출
- 이상치 및 특이점 감지

## HTML 보고서 생성
대시보드에서 직접 HTML 형식의 보고서를 생성하고 다운로드할 수 있습니다:
- 주요 지표, 시계열 트렌드, 플랫폼별 성과 등 포함할 섹션 선택 가능
- 캠페인 상세 보고서 또는 전체 요약 보고서 중 선택 가능
- 생성된 보고서는 다운로드하여 공유 가능

## Requirements
최신 requirements.txt 파일 내용:
```
streamlit==1.30.0
pandas==2.1.3
numpy==1.26.3
plotly==5.18.0
statsmodels==0.14.1
scikit-learn==1.3.2
xgboost==2.0.3
pyyaml==6.0.1
jinja2==3.1.2
requests==2.31.0
matplotlib==3.8.2
seaborn==0.13.0
# 이메일 송신 (선택사항)
# 설정에 SMTP 서버 정보 필요
```

## 주의사항
- PDF 생성 기능은 제거되고 HTML 보고서 생성 기능만 지원됩니다.
- 모든 모듈은 기본적으로 활성화되어 있으며, 데이터 전처리가 자동으로 적용됩니다.
- 사용자 피드백을 저장하려면 feedback 디렉토리가 필요합니다 (자동 생성됨).

## 라이선스
이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.
```

이 README는 다음 사항이 업데이트되었습니다:
1. PDF 보고서 생성 대신 HTML 보고서 생성만 지원한다는 점
2. 데이터 전처리가 자동으로 적용된다는 점
3. AI 인사이트 생성 및 최적 설정 제안 (최적 캠페인 포함) 기능 추가
4. 피드백 디렉토리 추가
5. 시스템 구조에 추가 모듈 표시 (preprocessing.py, budget_optimizer.py, ab_test.py, insight_generator.py)
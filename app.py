import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.formula.api import ols
import yaml
import os 
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import logging
# src 모듈 import
from src.modeling import ModelTrainer
from src.reporter import ReportGenerator  
from src.alert_system import AlertSystem
from src.custom_kpi import CustomKPI
from src.budget_optimizer import BudgetOptimizer  # 예산 최적화 모듈
from src.preprocessing import DataPreprocessor  # 데이터 전처리 모듈
from src.ab_test import ABTestAnalyzer  # A/B 테스트 분석 모듈
from src.insight_generator import InsightGenerator

# 로그 디렉토리 확인 및 생성
if not os.path.exists("logs"):
    os.makedirs("logs")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("app")

# 로그 기록 예시
logger.info("앱이 시작되었습니다.")

# 스크립트 파일의 디렉토리 경로
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE_PATH = os.path.join(SCRIPT_DIR, 'config/config.yaml')

# 디렉토리 생성
for dir_path in ['logs', 'models', 'reports', 'templates', 'alerts', 'kpis']:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logger.info(f"디렉토리 생성됨: {dir_path}")


# 페이지 설정
st.set_page_config(
    page_title="광고 캠페인 분석 대시보드",
    page_icon="📊",
    layout="wide"
)

# 스크립트 파일의 디렉토리 경로
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE_PATH = os.path.join(SCRIPT_DIR, 'config/config.yaml')

# 설정 파일 로드
@st.cache_data
def load_config():
    try:
        with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        st.warning("설정 파일을 찾을 수 없습니다. 기본 설정을 사용합니다.")
        return {"data_path": "data/campaign_data.csv"}

# 사이드바에 파일 업로드 기능 추가
st.sidebar.header("데이터 소스")
data_source = st.sidebar.radio(
    "데이터 소스 선택",
    options=["기본 데이터", "CSV 업로드"]
)

# 데이터 로드 함수 수정
@st.cache_data
def load_data(file_path=None, uploaded_file=None, apply_preprocessing=True):
    # 업로드된 파일이 있는 경우
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    # 파일 경로가 제공된 경우
    elif file_path is not None:
        df = pd.read_csv(file_path)
    else:
        st.error("데이터를 찾을 수 없습니다.")
        return pd.DataFrame()
    
    # 날짜 형식 변환
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # 원본 데이터 보존
    original_df = df.copy()
    
    # 데이터 전처리 적용 (옵션)
    if apply_preprocessing:
        preprocessor = DataPreprocessor()
        df, preprocessing_info = preprocessor.preprocess_data(
            df,
            remove_outliers=True,
            outlier_method='iqr',
            normalize=True,
            normalization_method='robust',
            create_features=True
        )
        
        # 전처리 정보 저장
        st.session_state.preprocessing_info = preprocessing_info
    
    # 원본 데이터 저장
    st.session_state.original_df = original_df
    
    return df

# 데이터 로드
try:
    # 설정 파일 로드
    config = load_config()
    
    if data_source == "기본 데이터":
        # 기존 방식으로 데이터 로드
        try:
            df = load_data(config['data_path'])
            st.sidebar.success(f"기본 데이터 로드 성공: {config['data_path']}")
        except FileNotFoundError:
            st.error("데이터 파일을 찾을 수 없습니다. config.yaml 파일에 지정된 경로를 확인해주세요.")
            st.stop()
    else:
        # 파일 업로드 기능
        uploaded_file = st.sidebar.file_uploader("CSV 파일을 업로드하세요", type=["csv"])
        
        if uploaded_file is not None:
            # 업로드된 파일로 데이터 로드
            df = load_data(uploaded_file=uploaded_file)
            st.sidebar.success("파일 업로드 성공!")
            
            # 업로드된 데이터 정보 표시
            with st.sidebar.expander("업로드된 데이터 정보"):
                st.write(f"**행**: {df.shape[0]}, **열**: {df.shape[1]}")
                
                # 열 데이터 타입 요약
                dtype_counts = df.dtypes.value_counts().to_dict()
                dtype_summary = ", ".join([f"{count}개의 {dtype}" for dtype, count in dtype_counts.items()])
                st.write(f"**데이터 타입**: {dtype_summary}")
        else:
            st.error("CSV 파일을 업로드하세요.")
            st.stop()
except Exception as e:
    st.error(f"데이터 로드 중 오류가 발생했습니다: {e}")
    st.stop()

try:
    # 설정 파일 로드
    config = load_config()
    
    # 데이터 로드
    df = load_data(config['data_path'])
    
    # 타이틀과 소개
    st.title(config['dashboard']['title'])
    st.markdown(config['dashboard']['description'])
    
    # 사이드바 필터
    st.sidebar.header("필터")
    
    # 날짜 범위 선택
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    date_range = st.sidebar.date_input(
        "날짜 범위",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        mask = (df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)
        filtered_df = df.loc[mask]
    else:
        filtered_df = df
    
    # 플랫폼 선택
    platforms = ['전체'] + sorted(df['platform'].unique().tolist())
    selected_platform = st.sidebar.selectbox("플랫폼", platforms)
    
    if selected_platform != '전체':
        filtered_df = filtered_df[filtered_df['platform'] == selected_platform]
    
    # 캠페인 선택
    campaigns = ['전체'] + sorted(filtered_df['campaign_name'].unique().tolist())
    selected_campaign = st.sidebar.selectbox("캠페인", campaigns)
    
    if selected_campaign != '전체':
        filtered_df = filtered_df[filtered_df['campaign_name'] == selected_campaign]
    
    # 타겟 연령 선택
    ages = ['전체'] + sorted(df['target_age'].unique().tolist())
    selected_age = st.sidebar.selectbox("타겟 연령", ages)
    
    if selected_age != '전체':
        filtered_df = filtered_df[filtered_df['target_age'] == selected_age]
    
    # 타겟 성별 선택
    genders = ['전체'] + sorted(df['target_gender'].unique().tolist())
    selected_gender = st.sidebar.selectbox("타겟 성별", genders)
    
    if selected_gender != '전체':
        filtered_df = filtered_df[filtered_df['target_gender'] == selected_gender]
    
    # 크리에이티브 유형 선택
    creative_types = ['전체'] + sorted(df['creative_type'].unique().tolist())
    selected_creative = st.sidebar.selectbox("크리에이티브 유형", creative_types)
    
    if selected_creative != '전체':
        filtered_df = filtered_df[filtered_df['creative_type'] == selected_creative]
    
    # 주요 지표 표시
    st.header("주요 지표")
    
    if hasattr(st.session_state, 'original_df'):
        # 원본 데이터에서 필터링된 부분을 가져와 집계
        original_filtered = st.session_state.original_df.copy()
        
        # 동일한 필터 적용
        if len(date_range) == 2:
            start_date, end_date = date_range
            original_filtered = original_filtered[(original_filtered['date'].dt.date >= start_date) & 
                                              (original_filtered['date'].dt.date <= end_date)]
        
        if selected_platform != '전체':
            original_filtered = original_filtered[original_filtered['platform'] == selected_platform]
        
        if selected_campaign != '전체':
            original_filtered = original_filtered[original_filtered['campaign_name'] == selected_campaign]
        
        if selected_age != '전체':
            original_filtered = original_filtered[original_filtered['target_age'] == selected_age]
        
        if selected_gender != '전체':
            original_filtered = original_filtered[original_filtered['target_gender'] == selected_gender]
            
        if selected_creative != '전체':
            original_filtered = original_filtered[original_filtered['creative_type'] == selected_creative]
        
        # 원본 데이터로 지표 계산
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            spend_sum_orig = original_filtered['spend'].sum()
            daily_budget_sum_orig = original_filtered['daily_budget'].sum()
            delta_orig = f"{spend_sum_orig / daily_budget_sum_orig:.1%} (예산 대비)" if daily_budget_sum_orig > 0 else "N/A"
            st.metric(label="총 지출 (원본)", value=f"₩{spend_sum_orig:,.0f}", delta=delta_orig)
        with col2:
            st.metric(label="총 수익 (원본)", value=f"₩{original_filtered['revenue'].sum():,.0f}", delta=None)
        with col3:
            revenue_sum_orig = original_filtered['revenue'].sum()
            spend_sum_orig_for_roas = original_filtered['spend'].sum()
            avg_roas_orig = revenue_sum_orig / spend_sum_orig_for_roas if spend_sum_orig_for_roas > 0 else 0
            st.metric(label="평균 ROAS (원본)", value=f"{avg_roas_orig:.2f}", delta=None)
        with col4:
            total_conversions_orig = original_filtered['conversions'].sum()
            spend_sum_orig_for_cpa = original_filtered['spend'].sum()
            cost_per_conversion_orig = spend_sum_orig_for_cpa / total_conversions_orig if total_conversions_orig > 0 else 0
            st.metric(label="전환당 비용 (CPA) (원본)", value=f"₩{cost_per_conversion_orig:,.0f}", delta=None)
    else:
        # 기존 코드 사용 (세션 저장소에 원본 데이터가 없는 경우, filtered_df 사용)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            spend_sum_filt = filtered_df['spend'].sum()
            daily_budget_sum_filt = filtered_df['daily_budget'].sum()
            delta_filt = f"{spend_sum_filt / daily_budget_sum_filt:.1%} (예산 대비)" if daily_budget_sum_filt > 0 else "N/A"
            st.metric(label="총 지출", value=f"₩{spend_sum_filt:,.0f}", delta=delta_filt)
        with col2:
            st.metric(label="총 수익", value=f"₩{filtered_df['revenue'].sum():,.0f}", delta=None)
        with col3:
            revenue_sum_filt = filtered_df['revenue'].sum()
            spend_sum_filt_for_roas = filtered_df['spend'].sum()
            avg_roas_filt = revenue_sum_filt / spend_sum_filt_for_roas if spend_sum_filt_for_roas > 0 else 0
            st.metric(label="평균 ROAS", value=f"{avg_roas_filt:.2f}", delta=None)
        with col4:
            total_conversions_filt = filtered_df['conversions'].sum()
            spend_sum_filt_for_cpa = filtered_df['spend'].sum()
            cost_per_conversion_filt = spend_sum_filt_for_cpa / total_conversions_filt if total_conversions_filt > 0 else 0
            st.metric(label="전환당 비용 (CPA)", value=f"₩{cost_per_conversion_filt:,.0f}", delta=None)
    
    # 인사이트 섹션 추가
    st.header("AI 인사이트")

    # 인사이트 생성
    with st.spinner("인사이트 생성 중..."):
        insight_generator = InsightGenerator()
        insights = insight_generator.generate_insights(filtered_df, target_col='roas', top_n=5)
        
        # 인사이트 표시
        for i, insight in enumerate(insights):
            st.info(f"**인사이트 {i+1}:** {insight}")

    # 새로고침 버튼
    if st.button("인사이트 새로고침"):
        st.experimental_rerun()
    
    # 예산 최적화 섹션
    st.header("예산 최적화")

    # 예산 최적화 모듈 초기화
    budget_optimizer = BudgetOptimizer()

    # 필터링된 데이터로 예산 최적화 분석 수행
    if not filtered_df.empty:
        campaign_perf = budget_optimizer.analyze_and_visualize(filtered_df)
    else:
        st.warning("데이터가 없습니다. 필터 조건을 변경해보세요.")
    
    # 전처리 정보 표시 섹션
    if hasattr(st.session_state, 'preprocessing_info'):
        st.header("데이터 전처리 정보")
        
        preprocessing_info = st.session_state.preprocessing_info
        
        # 적용된 전처리 단계
        st.subheader("적용된 전처리 단계")
        for step in preprocessing_info['steps_applied']:
            st.write(f"- {step}")
        
        # 이상치 정보
        if 'outliers' in preprocessing_info:
            st.subheader("이상치 처리 정보")
            outliers_info = preprocessing_info['outliers']
            
            st.write(f"이상치 탐지 방법: {outliers_info['method']}")
            st.write(f"처리된 이상치 수: {outliers_info['total_outliers_handled']}")
            
            # 컬럼별 이상치 수
            col1, col2 = st.columns(2)
            with col1:
                st.write("컬럼별 이상치 수:")
                for col, count in outliers_info['outliers_by_column'].items():
                    st.write(f"- {col}: {count}")
            
            with col2:
                # 이상치 비율 시각화
                if outliers_info['outliers_by_column']:
                    import plotly.express as px
                    outlier_df = pd.DataFrame({
                        'column': list(outliers_info['outliers_by_column'].keys()),
                        'count': list(outliers_info['outliers_by_column'].values())
                    })
                    
                    fig = px.bar(
                        outlier_df, 
                        x='column', 
                        y='count',
                        title='컬럼별 이상치 수',
                        labels={'column': '컬럼', 'count': '이상치 수'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # 파생변수 정보
        if 'features' in preprocessing_info and preprocessing_info['features']['created_features']:
            st.subheader("생성된 파생변수")
            for feature in preprocessing_info['features']['created_features']:
                st.write(f"- {feature}")
    
    
    
    # 사용자 정의 KPI 관리
    st.sidebar.header("사용자 정의 KPI")

    # KPI 정의
    with st.sidebar.expander("KPI 관리"):
        custom_kpi = CustomKPI()
        
        # 새 KPI 추가
        with st.form("add_kpi_form"):
            kpi_name = st.text_input("KPI 이름")
            kpi_formula = st.text_input("KPI 공식 (예: revenue / spend)")
            kpi_description = st.text_input("KPI 설명")
            
            if st.form_submit_button("KPI 추가"):
                if kpi_name and kpi_formula:
                    success = custom_kpi.add_kpi(
                        name=kpi_name,
                        formula=kpi_formula,
                        description=kpi_description
                    )
                    if success:
                        st.success(f"KPI '{kpi_name}'가 추가되었습니다.")
        
        # 현재 KPI 목록 확인
        all_kpis = custom_kpi.get_kpi_definitions()
        if all_kpis:
            st.write("### 정의된 KPI 목록")
            for name, kpi_def in all_kpis.items():
                st.write(f"**{name}**: {kpi_def['formula']} - {kpi_def.get('description', '')}")

    # KPI 계산 및 표시
    if st.sidebar.checkbox("사용자 정의 KPI 계산", value=True):
        if not filtered_df.empty:
            # KPI 계산
            kpi_df = custom_kpi.calculate_kpis(filtered_df)
            
            # 집계된 KPI 표시
            st.header("사용자 정의 KPI")
            aggregated_kpis = custom_kpi.aggregate_kpis(kpi_df)
            
            # 데이터프레임으로 표시
            st.dataframe(aggregated_kpis)
        else:
            st.warning("데이터가 없습니다. 필터 조건을 변경해보세요.")
    
    # 알림 시스템
    if st.sidebar.checkbox("알림 시스템 활성화", value=False):
        st.header("알림 시스템")
        
        alert_system = AlertSystem()
        
        # 알림 임계값 설정
        with st.expander("알림 임계값 설정"):
            with st.form("alert_thresholds_form"):
                # 몇 가지 주요 임계값 설정 UI
                roas_min = st.number_input("최소 ROAS 임계값", min_value=0.0, value=1.0, step=0.1)
                ctr_min = st.number_input("최소 CTR 임계값 (%)", min_value=0.0, value=1.0, step=0.1) / 100
                spend_change = st.number_input("지출 변화율 임계값 (%)", min_value=5.0, value=20.0, step=5.0) / 100
                
                if st.form_submit_button("임계값 저장"):
                    thresholds = {
                        'roas_min_threshold': roas_min,
                        'ctr_min_threshold': ctr_min,
                        'spend_change_threshold': spend_change
                    }
                    alert_system.set_thresholds(thresholds)
                    st.success("알림 임계값이 저장되었습니다.")
        
        # 알림 확인
        if st.button("알림 확인"):
            with st.spinner("알림 조건 확인 중..."):
                alerts = alert_system.check_alerts(filtered_df, period='daily')
                
                if alerts:
                    st.warning(f"{len(alerts)}개의 알림이 발생했습니다.")
                    
                    # 알림 표시
                    for i, alert in enumerate(alerts):
                        alert_type = alert['type']
                        metric = alert['metric']
                        value = alert['value']
                        threshold = alert.get('threshold', '')
                        
                        # 알림 유형에 따른 설명 텍스트
                        type_text = {
                            'metric_decrease': "지표 감소",
                            'metric_increase': "지표 증가",
                            'metric_below_min': "최소 임계값 미만",
                            'metric_above_max': "최대 임계값 초과",
                            'metric_anomaly': "이상치 감지"
                        }.get(alert_type, alert_type)
                        
                        # 지표 한글명
                        metric_text = {
                            'spend': "지출",
                            'impressions': "노출 수",
                            'clicks': "클릭 수",
                            'conversions': "전환 수",
                            'revenue': "수익",
                            'ctr': "클릭률 (CTR)",
                            'cvr': "전환율 (CVR)",
                            'roas': "ROAS",
                            'cpc': "클릭당 비용 (CPC)",
                            'cpa': "전환당 비용 (CPA)"
                        }.get(metric, metric)
                        
                        st.info(f"**알림 {i+1}:** {type_text} - {metric_text} ({value:.2f})")
                else:
                    st.success("발생한 알림이 없습니다.")
        
        # 알림 기록 확인
        alert_history = alert_system.load_alert_history()
        if alert_history:
            with st.expander("알림 기록"):
                st.json(alert_history)
    
    # 모델 훈련 및 저장
    if st.button("모델 훈련 및 저장"):
        with st.spinner("모델 훈련 중..."):
            # 모델 훈련기 초기화
            trainer = ModelTrainer()
            
            # 데이터 전처리
            X_train, X_test, y_train, y_test, preprocessor, feature_names = trainer.preprocess_data(filtered_df)
            
            # 모델 훈련
            models = trainer.train_models(X_train, y_train, preprocessor, feature_names)
            
            # 모델 평가
            evaluation = trainer.evaluate_models(X_test, y_test)
            
            # 최고 성능 모델 저장
            if trainer.best_model:
                model_path = trainer.save_model()
                st.success(f"모델이 저장되었습니다: {model_path}")
                
            # 평가 결과 표시
            st.write("### 모델 평가 결과")
            st.dataframe(evaluation)

    if st.button("보고서 생성"):
        # 보고서 생성 섹션
        st.header("보고서 생성")

        col1, col2 = st.columns(2)

        with col1:
            report_type = st.selectbox(
                "보고서 유형",
                options=["요약 보고서", "캠페인 상세 보고서"],
                index=0
            )

        with col2:
            include_sections = st.multiselect(
                "포함할 섹션",
                options=["주요 지표", "시계열 트렌드", "플랫폼별 성과", "크리에이티브별 성과", "타겟팅별 성과"],
                default=["주요 지표", "시계열 트렌드", "플랫폼별 성과"]
            )

        if st.button("HTML 보고서 생성"):
            with st.spinner("보고서 생성 중..."):
                try:
                    # 보고서 생성기 초기화
                    from src.reporter import ReportGenerator
                    generator = ReportGenerator()
                    
                    # HTML 보고서 생성
                    report_path = generator.create_html_report(filtered_df, include_sections=include_sections)
                    
                    if report_path:
                        st.success(f"보고서가 생성되었습니다: {report_path}")
                        
                        # 다운로드 버튼 제공
                        with open(report_path, "rb") as file:
                            file_name = os.path.basename(report_path)
                            
                            btn = st.download_button(
                                label="보고서 다운로드",
                                data=file,
                            file_name=file_name,
                            mime="text/html"
                        )
                except Exception as e:
                    st.error(f"보고서 생성 중 오류가 발생했습니다: {e}")
            
    # 시계열 트렌드
    st.header("시계열 트렌드")
    
    # 일별 데이터 집계
    daily_data = filtered_df.groupby('date').agg({
        'spend': 'sum',
        'impressions': 'sum',
        'clicks': 'sum',
        'conversions': 'sum',
        'revenue': 'sum'
    }).reset_index()
 
    # 시간 단위 선택
    time_unit = st.radio(
        "시간 단위",
        options=["일별", "주별", "월별"],
        horizontal=True
    )

    # 선택한 시간 단위에 따라 데이터 재집계
    if time_unit == "주별":
        # 주별 데이터 집계
        filtered_df['week'] = filtered_df['date'].dt.isocalendar().week
        filtered_df['year'] = filtered_df['date'].dt.isocalendar().year
        time_data = filtered_df.groupby(['year', 'week']).agg({
            'spend': 'sum',
            'impressions': 'sum',
            'clicks': 'sum',
            'conversions': 'sum',
            'revenue': 'sum'
        }).reset_index()
        time_data['date'] = time_data.apply(lambda x: pd.to_datetime(f"{int(x['year'])}-W{int(x['week'])}-1", format='%Y-W%W-%w'), axis=1)
    elif time_unit == "월별":
        # 월별 데이터 집계
        filtered_df['month'] = filtered_df['date'].dt.to_period('M')
        time_data = filtered_df.groupby('month').agg({
            'spend': 'sum',
            'impressions': 'sum',
            'clicks': 'sum',
            'conversions': 'sum',
            'revenue': 'sum'
        }).reset_index()
        time_data['date'] = time_data['month'].dt.to_timestamp()
    else:
        time_data = daily_data
    
    # 파생 지표 계산
    daily_data['ctr'] = daily_data['clicks'] / daily_data['impressions'] * 100
    daily_data['cvr'] = daily_data['conversions'] / daily_data['clicks'] * 100
    daily_data['roas'] = daily_data['revenue'] / daily_data['spend']
    
    # 시계열 차트를 위한 지표 선택
    metric_options = {
        '지출': 'spend',
        '노출': 'impressions',
        '클릭': 'clicks',
        '전환': 'conversions',
        '수익': 'revenue',
        'CTR (%)': 'ctr',
        'CVR (%)': 'cvr',
        'ROAS': 'roas'
    }
    
    selected_metrics = st.multiselect(
        "표시할 지표 선택",
        options=list(metric_options.keys()),
        default=['지출', 'ROAS']
    )
    
    if selected_metrics:
        fig = go.Figure()
        
        for metric in selected_metrics:
            metric_col = metric_options[metric]
            
            # Y축 위치 결정 (ROAS, CVR, CTR은 오른쪽, 나머지는 왼쪽)
            yaxis = 'y2' if metric in ['ROAS', 'CTR (%)', 'CVR (%)'] else 'y'
            
            fig.add_trace(
                go.Scatter(
                    x=daily_data['date'],
                    y=daily_data[metric_col],
                    mode='lines+markers',
                    name=metric
                )
            )
        
        fig.update_layout(
            title='일별 성과 추이',
            xaxis_title='날짜',
            yaxis_title='금액 (₩)',
            yaxis2=dict(
                title='비율',
                overlaying='y',
                side='right'
            ),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("지표를 선택해주세요.")
    
    # 회귀 분석
    st.header("회귀 분석")
    
    # 독립 변수 선택
    cols1, cols2 = st.columns(2)
    
    with cols1:
        st.subheader("ROAS 영향 요인 분석")
        
        independent_vars = ['지출 (spend)', '노출 (impressions)', '클릭 (clicks)', 
                            '클릭률 (ctr)', '전환 (conversions)', '전환율 (cvr)']
        
        selected_vars = st.multiselect(
            "독립 변수 선택",
            options=independent_vars,
            default=['클릭률 (ctr)', '전환율 (cvr)', '전환 (conversions)']
        )
        
        var_mapping = {
            '지출 (spend)': 'spend',
            '노출 (impressions)': 'impressions',
            '클릭 (clicks)': 'clicks',
            '클릭률 (ctr)': 'ctr',
            '전환 (conversions)': 'conversions',
            '전환율 (cvr)': 'cvr'
        }
        
        if selected_vars:
            # 회귀 분석을 위한 데이터 준비
            regression_df = filtered_df.copy()
            
            # 수식 구성
            formula = "roas ~ " + " + ".join([var_mapping[var] for var in selected_vars])
            
            try:
                # 회귀 모델 구성 및 학습
                model = ols(formula, data=regression_df).fit()
                
                # 결과 표시
                st.write("**회귀 모델 결과:**")
                
                # 주요 지표 표시
                r_squared = model.rsquared
                adj_r_squared = model.rsquared_adj
                f_statistic = model.fvalue
                p_value = model.f_pvalue
                
                st.write(f"R-squared: {r_squared:.4f}")
                st.write(f"Adjusted R-squared: {adj_r_squared:.4f}")
                st.write(f"F-statistic: {f_statistic:.4f}, p-value: {p_value:.6f}")
                
                # 계수 표시
                st.write("**변수별 영향력 (계수):**")
                
                # 계수를 데이터프레임으로 변환
                coef_df = pd.DataFrame({
                    '변수': ['상수항'] + [var for var in selected_vars],
                    '계수': [model.params['Intercept']] + [model.params[var_mapping[var]] for var in selected_vars],
                    'p-value': [model.pvalues['Intercept']] + [model.pvalues[var_mapping[var]] for var in selected_vars]
                })
                
                # 유의성 표시
                coef_df['유의성'] = coef_df['p-value'].apply(
                    lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
                )
                
                # 데이터프레임 표시
                st.dataframe(coef_df.style.format({
                    '계수': '{:.4f}',
                    'p-value': '{:.4f}'
                }))
                
                # 유의성 범례
                st.write("유의성: *** p<0.001, ** p<0.01, * p<0.05")
                
            except Exception as e:
                st.error(f"회귀 분석 중 오류가 발생했습니다: {e}")
        else:
            st.info("독립 변수를 선택해주세요.")
    
    with cols2:
        # 선택한 변수와 ROAS의 산점도
        st.subheader("변수별 ROAS 관계")
        
        if selected_vars:
            scatter_var = st.selectbox(
                "변수 선택",
                options=selected_vars
            )
            
            if scatter_var:
                fig = px.scatter(
                    filtered_df,
                    x=var_mapping[scatter_var],
                    y='roas',
                    color='platform',
                    trendline='ols',
                    labels={
                        var_mapping[scatter_var]: scatter_var,
                        'roas': 'ROAS',
                        'platform': '플랫폼'
                    },
                    title=f'{scatter_var}와 ROAS의 관계'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 상관관계 계산
                corr = filtered_df[[var_mapping[scatter_var], 'roas']].corr().iloc[0, 1]
                st.write(f"**상관계수:** {corr:.4f}")
                
                # 상관관계 해석
                if abs(corr) >= 0.7:
                    st.write("**매우 강한 상관관계**가 있습니다.")
                elif abs(corr) >= 0.5:
                    st.write("**강한 상관관계**가 있습니다.")
                elif abs(corr) >= 0.3:
                    st.write("**중간 정도의 상관관계**가 있습니다.")
                elif abs(corr) >= 0.1:
                    st.write("**약한 상관관계**가 있습니다.")
                else:
                    st.write("**거의 상관관계가 없습니다.**")
        else:
            st.info("독립 변수를 선택해주세요.")
    
    # 플랫폼별/크리에이티브별 성과 비교
    st.header("그룹별 성과 비교")
    
    # 탭 생성
    if 'test_group' in filtered_df.columns:
        tab1, tab2, tab3, tab4 = st.tabs(["플랫폼별", "크리에이티브별", "타겟팅별", "A/B 테스트"])  # A/B 테스트 탭 추가
    else:
        tab1, tab2, tab3 = st.tabs(["플랫폼별", "크리에이티브별", "타겟팅별"])
    
    with tab1:
        # 플랫폼별 성과
        platform_data = filtered_df.groupby('platform').agg({
            'spend': 'sum',
            'impressions': 'sum',
            'clicks': 'sum',
            'conversions': 'sum',
            'revenue': 'sum'
        }).reset_index()
        
        # 파생 지표 계산
        platform_data['ctr'] = platform_data['clicks'] / platform_data['impressions'] * 100
        platform_data['cvr'] = platform_data['conversions'] / platform_data['clicks'] * 100
        platform_data['roas'] = platform_data['revenue'] / platform_data['spend']
        platform_data['cpc'] = platform_data['spend'] / platform_data['clicks']
        platform_data['cpa'] = platform_data['spend'] / platform_data['conversions']
        
        # 바차트로 표시
        st.subheader("플랫폼별 주요 지표")
        
        platform_metric = st.selectbox(
            "지표 선택 (플랫폼별)",
            options=['roas', 'ctr', 'cvr', 'cpc', 'cpa'],
            format_func=lambda x: {
                'roas': 'ROAS',
                'ctr': 'CTR (%)',
                'cvr': 'CVR (%)',
                'cpc': 'CPC (₩)',
                'cpa': 'CPA (₩)'
            }[x]
        )
        
        fig = px.bar(
            platform_data,
            x='platform',
            y=platform_metric,
            color='platform',
            text_auto='.2f',
            labels={
                'platform': '플랫폼',
                platform_metric: {
                    'roas': 'ROAS',
                    'ctr': 'CTR (%)',
                    'cvr': 'CVR (%)',
                    'cpc': 'CPC (₩)',
                    'cpa': 'CPA (₩)'
                }[platform_metric]
            },
            title='플랫폼별 성과 비교'
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # 플랫폼별 성과 데이터 테이블
        st.subheader("플랫폼별 상세 지표")
        
        # 테이블에 표시할 컬럼과 포맷 설정
        display_columns = {
            'platform': '플랫폼',
            'spend': '지출 (₩)',
            'impressions': '노출 수',
            'clicks': '클릭 수',
            'ctr': 'CTR (%)',
            'conversions': '전환 수',
            'cvr': 'CVR (%)',
            'revenue': '수익 (₩)',
            'roas': 'ROAS',
            'cpc': 'CPC (₩)',
            'cpa': 'CPA (₩)'
        }
        
        platform_display = platform_data.copy()
        platform_display.columns = [display_columns.get(col, col) for col in platform_display.columns]
        
        # 숫자 포맷팅
        st.dataframe(platform_display.style.format({
            '지출 (₩)': '{:,.0f}',
            '노출 수': '{:,.0f}',
            '클릭 수': '{:,.0f}',
            'CTR (%)': '{:.2f}',
            '전환 수': '{:,.0f}',
            'CVR (%)': '{:.2f}',
            '수익 (₩)': '{:,.0f}',
            'ROAS': '{:.2f}',
            'CPC (₩)': '{:.0f}',
            'CPA (₩)': '{:,.0f}'
        }))
    
    with tab2:
        # 크리에이티브별 성과
        creative_data = filtered_df.groupby('creative_type').agg({
            'spend': 'sum',
            'impressions': 'sum',
            'clicks': 'sum',
            'conversions': 'sum',
            'revenue': 'sum'
        }).reset_index()
        
        # 파생 지표 계산
        creative_data['ctr'] = creative_data['clicks'] / creative_data['impressions'] * 100
        creative_data['cvr'] = creative_data['conversions'] / creative_data['clicks'] * 100
        creative_data['roas'] = creative_data['revenue'] / creative_data['spend']
        creative_data['cpc'] = creative_data['spend'] / creative_data['clicks']
        creative_data['cpa'] = creative_data['spend'] / creative_data['conversions']
        
        # 바차트로 표시
        st.subheader("크리에이티브 유형별 주요 지표")
        
        creative_metric = st.selectbox(
            "지표 선택 (크리에이티브별)",
            options=['roas', 'ctr', 'cvr', 'cpc', 'cpa'],
            format_func=lambda x: {
                'roas': 'ROAS',
                'ctr': 'CTR (%)',
                'cvr': 'CVR (%)',
                'cpc': 'CPC (₩)',
                'cpa': 'CPA (₩)'
            }[x]
        )
        
        fig = px.bar(
            creative_data,
            x='creative_type',
            y=creative_metric,
            color='creative_type',
            text_auto='.2f',
            labels={
                'creative_type': '크리에이티브 유형',
                creative_metric: {
                    'roas': 'ROAS',
                    'ctr': 'CTR (%)',
                    'cvr': 'CVR (%)',
                    'cpc': 'CPC (₩)',
                    'cpa': 'CPA (₩)'
                }[creative_metric]
            },
            title='크리에이티브 유형별 성과 비교'
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # 크리에이티브별 성과 데이터 테이블
        st.subheader("크리에이티브 유형별 상세 지표")
        
        creative_display = creative_data.copy()
        creative_display.columns = [display_columns.get(col, col) for col in creative_display.columns]
        
        # 숫자 포맷팅
        st.dataframe(creative_display.style.format({
            '지출 (₩)': '{:,.0f}',
            '노출 수': '{:,.0f}',
            '클릭 수': '{:,.0f}',
            'CTR (%)': '{:.2f}',
            '전환 수': '{:,.0f}',
            'CVR (%)': '{:.2f}',
            '수익 (₩)': '{:,.0f}',
            'ROAS': '{:.2f}',
            'CPC (₩)': '{:.0f}',
            'CPA (₩)': '{:,.0f}'
        }))
    
    with tab3:
        # 타겟팅별 성과
        col1, col2 = st.columns(2)
        
        with col1:
            # 연령별 성과
            age_data = filtered_df.groupby('target_age').agg({
                'spend': 'sum',
                'impressions': 'sum',
                'clicks': 'sum',
                'conversions': 'sum',
                'revenue': 'sum'
            }).reset_index()
            
            # 파생 지표 계산
            age_data['ctr'] = age_data['clicks'] / age_data['impressions'] * 100
            age_data['cvr'] = age_data['conversions'] / age_data['clicks'] * 100
            age_data['roas'] = age_data['revenue'] / age_data['spend']
            
            # 선택된 지표
            age_metric = st.selectbox(
                "지표 선택 (연령별)",
                options=['roas', 'ctr', 'cvr'],
                format_func=lambda x: {
                    'roas': 'ROAS',
                    'ctr': 'CTR (%)',
                    'cvr': 'CVR (%)'
                }[x]
            )
            
            # 차트
            fig = px.bar(
                age_data,
                x='target_age',
                y=age_metric,
                color='target_age',
                text_auto='.2f',
                labels={
                    'target_age': '타겟 연령',
                    age_metric: {
                        'roas': 'ROAS',
                        'ctr': 'CTR (%)',
                        'cvr': 'CVR (%)'
                    }[age_metric]
                },
                title='연령별 성과 비교'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 성별별 성과
            gender_data = filtered_df.groupby('target_gender').agg({
                'spend': 'sum',
                'impressions': 'sum',
                'clicks': 'sum',
                'conversions': 'sum',
                'revenue': 'sum'
            }).reset_index()
            
            # 파생 지표 계산
            gender_data['ctr'] = gender_data['clicks'] / gender_data['impressions'] * 100
            gender_data['cvr'] = gender_data['conversions'] / gender_data['clicks'] * 100
            gender_data['roas'] = gender_data['revenue'] / gender_data['spend']
            
            # 선택된 지표
            gender_metric = st.selectbox(
                "지표 선택 (성별)",
                options=['roas', 'ctr', 'cvr'],
                format_func=lambda x: {
                    'roas': 'ROAS',
                    'ctr': 'CTR (%)',
                    'cvr': 'CVR (%)'
                }[x]
            )
            
            # 차트
            fig = px.bar(
                gender_data,
                x='target_gender',
                y=gender_metric,
                color='target_gender',
                text_auto='.2f',
                labels={
                    'target_gender': '타겟 성별',
                    gender_metric: {
                        'roas': 'ROAS',
                        'ctr': 'CTR (%)',
                        'cvr': 'CVR (%)'
                    }[gender_metric]
                },
                title='성별 성과 비교'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # A/B 테스트 탭 내용 (탭 정의 이후에 추가)
    if 'test_group' in filtered_df.columns:
        with tab4:
            # A/B 테스트 분석 모듈 초기화
            ab_test_analyzer = ABTestAnalyzer()
            
            # A/B 테스트 분석 실행
            ab_test_analyzer.analyze_ab_test(filtered_df)
    
    
    # ROAS 예측 모델
    st.header("ROAS 예측 모델")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # ROAS 예측을 위한 입력값
        st.subheader("ROAS 예측")
        st.write("각 변수의 값을 입력하여 ROAS를 예측해보세요.")
        
        # 입력 슬라이더
        spend = st.slider("지출 (₩)", 
                        min_value=float(filtered_df['spend'].min()),
                        max_value=float(filtered_df['spend'].max()),
                        value=float(filtered_df['spend'].mean()))
        
        impressions = st.slider("노출 수", 
                               min_value=int(filtered_df['impressions'].min()),
                               max_value=int(filtered_df['impressions'].max()),
                               value=int(filtered_df['impressions'].mean()))
        
        clicks = st.slider("클릭 수", 
                          min_value=int(filtered_df['clicks'].min()),
                          max_value=int(filtered_df['clicks'].max()),
                          value=int(filtered_df['clicks'].mean()))
        
        ctr = clicks / impressions if impressions > 0 else 0
        
        conversions = st.slider("전환 수", 
                               min_value=int(filtered_df['conversions'].min()),
                               max_value=int(filtered_df['conversions'].max()),
                               value=int(filtered_df['conversions'].mean()))
        
        cvr = conversions / clicks if clicks > 0 else 0
        
        # 예측 모델 생성
        try:
            # 회귀 모델 학습
            formula = "roas ~ spend + impressions + clicks + ctr + conversions + cvr"
            model = ols(formula, data=filtered_df).fit()
            
            # 예측 데이터 구성
            prediction_data = pd.DataFrame({
                'spend': [spend],
                'impressions': [impressions],
                'clicks': [clicks],
                'ctr': [ctr],
                'conversions': [conversions],
                'cvr': [cvr]
            })
            
            # 예측
            prediction = model.predict(prediction_data)
            
            # 결과 표시
            st.metric(
                label="예측 ROAS",
                value=f"{prediction.iloc[0]:.2f}"
            )
            
            # 참고용 계산값 표시
            st.write(f"참고: CTR = {ctr:.2%}, CVR = {cvr:.2%}")
            
        except Exception as e:
            st.error(f"예측 모델 구성 중 오류가 발생했습니다: {e}")
    
    with col2:
        # ROAS 최적화 제안
        st.subheader("ROAS 최적화 제안")
        
        # 회귀 모델을 사용한 시뮬레이션 (이미 위에서 학습됨)
        try:
            # 모델 계수 표시
            st.write("**변수별 ROAS 영향력:**")
            
            # 계수를 데이터프레임으로 변환
            coef_df = pd.DataFrame({
                '변수': model.params.index,
                '계수': model.params.values,
                'p-value': model.pvalues.values
            })
            
            # 상수항 제외하고 절대값 기준으로 정렬
            coef_df = coef_df[coef_df['변수'] != 'Intercept'].sort_values(by='계수', key=abs, ascending=False)
            
            # 유의성 표시
            coef_df['유의성'] = coef_df['p-value'].apply(
                lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
            )
            
            # 영향력이 가장 큰 변수들 제안
            st.dataframe(coef_df.style.format({
                '계수': '{:.4f}',
                'p-value': '{:.4f}'
            }))
            
            # 최적화 제안
            st.write("**최적화 제안:**")
            
            # 긍정적 영향이 큰 변수 찾기
            positive_vars = coef_df[coef_df['계수'] > 0].sort_values(by='계수', ascending=False)
            negative_vars = coef_df[coef_df['계수'] < 0].sort_values(by='계수', ascending=True)
            
            # 제안 목록
            recommendations = []
            
            if not positive_vars.empty:
                for _, row in positive_vars.head(2).iterrows():
                    var_name = row['변수']
                    var_label = {
                        'spend': '지출',
                        'impressions': '노출 수',
                        'clicks': '클릭 수',
                        'ctr': '클릭률',
                        'conversions': '전환 수',
                        'cvr': '전환율'
                    }.get(var_name, var_name)
                    
                    recommendations.append(f"{var_label}을(를) 증가시키세요. (영향력: {row['계수']:.4f})")
            
            if not negative_vars.empty:
                for _, row in negative_vars.head(2).iterrows():
                    var_name = row['변수']
                    var_label = {
                        'spend': '지출',
                        'impressions': '노출 수',
                        'clicks': '클릭 수',
                        'ctr': '클릭률',
                        'conversions': '전환 수',
                        'cvr': '전환율'
                    }.get(var_name, var_name)
                    
                    recommendations.append(f"{var_label}을(를) 감소시키세요. (영향력: {row['계수']:.4f})")
            
            # 제안 표시
            for rec in recommendations:
                st.write(f"- {rec}")
            
            # 최적 설정 제안
            st.write("**최적 설정 제안:**")

            # 플랫폼별 ROAS
            best_platform = platform_data.sort_values('roas', ascending=False).iloc[0]
            st.write(f"- 최적 플랫폼: **{best_platform['platform']}** (ROAS: {best_platform['roas']:.2f})")

            # 크리에이티브별 ROAS
            best_creative = creative_data.sort_values('roas', ascending=False).iloc[0]
            st.write(f"- 최적 크리에이티브: **{best_creative['creative_type']}** (ROAS: {best_creative['roas']:.2f})")

            # 타겟 연령별 ROAS
            best_age = age_data.sort_values('roas', ascending=False).iloc[0]
            st.write(f"- 최적 타겟 연령: **{best_age['target_age']}** (ROAS: {best_age['roas']:.2f})")

            # 타겟 성별별 ROAS
            best_gender = gender_data.sort_values('roas', ascending=False).iloc[0]
            st.write(f"- 최적 타겟 성별: **{best_gender['target_gender']}** (ROAS: {best_gender['roas']:.2f})")

            # 캠페인별 ROAS (추가)
            campaign_roas = filtered_df.groupby('campaign_name').agg({
                'spend': 'sum',
                'revenue': 'sum'
            }).reset_index()
            campaign_roas['roas'] = campaign_roas['revenue'] / campaign_roas['spend']
            best_campaign = campaign_roas.sort_values('roas', ascending=False).iloc[0]
            st.write(f"- 최적 캠페인: **{best_campaign['campaign_name']}** (ROAS: {best_campaign['roas']:.2f})")
        except Exception as e:
            st.error(f"최적화 제안 생성 중 오류가 발생했습니다: {e}")
    
    # 피드백 섹션
    st.markdown("---")
    st.header("피드백")
    st.write("대시보드 개선을 위한 의견을 남겨주세요.")

    feedback_text = st.text_area("피드백 내용", height=100)
    user_email = st.text_input("이메일 (선택사항)")

    if st.button("피드백 제출"):
        if feedback_text:
            # 피드백 저장 로직 (파일이나 데이터베이스에 저장)
            feedback_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            feedback_data = {
                "timestamp": feedback_time,
                "feedback": feedback_text,
                "email": user_email if user_email else "익명"
            }
            
            # 피드백 디렉토리 확인 및 생성
            feedback_dir = "feedback"
            if not os.path.exists(feedback_dir):
                os.makedirs(feedback_dir)
                logger.info(f"피드백 디렉토리 생성됨: {feedback_dir}")
                
            import json
            try:
                feedback_file = os.path.join(feedback_dir, "user_feedback.json")
                
                # 기존 파일이 있는지 확인
                if os.path.exists(feedback_file):
                    with open(feedback_file, "r", encoding="utf-8") as f:
                        try:
                            feedbacks = json.load(f)
                        except json.JSONDecodeError:
                            feedbacks = []
                else:
                    # 파일이 없으면 새로 생성
                    feedbacks = []
                    
                # 피드백 추가
                feedbacks.append(feedback_data)
                
                # 파일에 저장
                with open(feedback_file, "w", encoding="utf-8") as f:
                    json.dump(feedbacks, f, ensure_ascii=False, indent=2)
                    
                st.success("피드백을 보내주셔서 감사합니다!")
            except Exception as e:
                st.error(f"피드백 저장 중 오류가 발생했습니다: {e}")
        else:
            st.warning("피드백 내용을 입력해주세요.")
    
    # 푸터
    st.markdown("---")
    st.markdown(f"데이터 마지막 업데이트: {max_date}")
    st.markdown("© 2025 광고 캠페인 분석 대시보드")

except FileNotFoundError:
    st.error("설정 파일 또는 데이터 파일을 찾을 수 없습니다. config.yaml 파일이 존재하는지 확인해주세요.")
except Exception as e:
    st.error(f"대시보드 실행 중 오류가 발생했습니다: {e}")
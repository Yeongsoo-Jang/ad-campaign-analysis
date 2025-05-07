import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import streamlit as st
import logging
import os

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/ab_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ab_test")

class ABTestAnalyzer:
    """
    A/B 테스트 데이터를 분석하고 시각화하는 클래스
    """
    
    def __init__(self):
        """
        ABTestAnalyzer 클래스 초기화
        """
        # 로그 디렉토리 생성
        if not os.path.exists("logs"):
            os.makedirs("logs")
            logger.info("로그 디렉토리 생성됨")
    
    def analyze_ab_test(self, df):
        """
        A/B 테스트 데이터를 분석하고 결과를 시각화합니다.
        
        Args:
            df (pandas.DataFrame): A/B 테스트 정보가 포함된 데이터
        """
        if df.empty:
            st.warning("데이터가 없습니다. 필터 조건을 변경해보세요.")
            return
        
        # A/B 테스트 그룹 확인
        if 'test_group' not in df.columns:
            st.error("데이터에 A/B 테스트 그룹 정보가 없습니다.")
            return
        
        # A/B 테스트 분석 섹션
        st.header("A/B 테스트 분석")
        
        # 그룹별 데이터 분리
        group_a = df[df['test_group'] == 'A']
        group_b = df[df['test_group'] == 'B']
        
        # 기본 통계 표시
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("A 그룹 (대조군)")
            st.write(f"데이터 수: {len(group_a)} 행")
        
        with col2:
            st.subheader("B 그룹 (실험군)")
            st.write(f"데이터 수: {len(group_b)} 행")
        
        # 분석할 지표 선택
        metrics = ['ctr', 'cvr', 'roas']
        selected_metric = st.selectbox(
            "분석할 지표 선택",
            options=metrics,
            format_func=lambda x: {
                'ctr': '클릭률 (CTR)',
                'cvr': '전환율 (CVR)',
                'roas': '광고 투자 수익률 (ROAS)'
            }.get(x, x)
        )
        
        # 캠페인별 분석 여부
        by_campaign = st.checkbox("캠페인별 분석", value=True)
        
        if by_campaign:
            self._analyze_by_campaign(df, group_a, group_b, selected_metric)
        else:
            self._analyze_overall(df, group_a, group_b, selected_metric)
    
    def _analyze_overall(self, df, group_a, group_b, metric):
        """
        전체 데이터에 대한 A/B 테스트 분석을 수행합니다.
        
        Args:
            df (pandas.DataFrame): 전체 데이터
            group_a (pandas.DataFrame): A 그룹 데이터
            group_b (pandas.DataFrame): B 그룹 데이터
            metric (str): 분석할 지표
        """
        # 지표 이름 변환
        metric_name = {
            'ctr': '클릭률 (CTR)',
            'cvr': '전환율 (CVR)',
            'roas': '광고 투자 수익률 (ROAS)'
        }.get(metric, metric)
        
        # 그룹별 평균값
        a_mean = group_a[metric].mean()
        b_mean = group_b[metric].mean()
        
        # 변화율
        change_pct = ((b_mean - a_mean) / a_mean) * 100
        
        # t-test
        t_stat, p_value = stats.ttest_ind(
            group_a[metric].dropna(),
            group_b[metric].dropna(),
            equal_var=False  # Welch's t-test
        )
        
        # 결과 표시
        st.subheader(f"전체 {metric_name} 분석")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label=f"A 그룹 평균 {metric_name}",
                value=f"{a_mean:.4f}" if metric != 'roas' else f"{a_mean:.2f}"
            )
        
        with col2:
            st.metric(
                label=f"B 그룹 평균 {metric_name}",
                value=f"{b_mean:.4f}" if metric != 'roas' else f"{b_mean:.2f}",
                delta=f"{change_pct:+.2f}%"
            )
        
        with col3:
            significant = p_value < 0.05
            st.metric(
                label="p-value (통계적 유의성)",
                value=f"{p_value:.4f}",
                delta="유의미함" if significant else "유의미하지 않음",
                delta_color="normal" if significant else "off"
            )
        
        # 시각화
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=['A 그룹', 'B 그룹'],
            y=[a_mean, b_mean],
            text=[f"{a_mean:.4f}", f"{b_mean:.4f}"],
            textposition='auto',
            name=metric_name
        ))
        
        fig.update_layout(
            title=f'A/B 테스트 결과: {metric_name} 비교',
            xaxis_title='테스트 그룹',
            yaxis_title=metric_name,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 결과 해석
        st.subheader("결과 해석")
        
        if p_value < 0.05:
            if b_mean > a_mean:
                st.success(f"B 그룹이 A 그룹보다 {metric_name}이(가) 통계적으로 유의미하게 높습니다. (p-value: {p_value:.4f})")
                st.write(f"B 그룹의 {metric_name}은(는) A 그룹보다 {change_pct:.2f}% 높습니다.")
            else:
                st.error(f"B 그룹이 A 그룹보다 {metric_name}이(가) 통계적으로 유의미하게 낮습니다. (p-value: {p_value:.4f})")
                st.write(f"B 그룹의 {metric_name}은(는) A 그룹보다 {change_pct:.2f}% 낮습니다.")
        else:
            st.info(f"두 그룹 간 {metric_name}의 차이는 통계적으로 유의미하지 않습니다. (p-value: {p_value:.4f})")
            st.write("더 많은 데이터를 수집하거나, 테스트 기간을 늘려보세요.")
        
        # 신뢰구간 계산
        from scipy import stats
        
        # A 그룹 신뢰구간
        a_std = group_a[metric].std()
        a_n = len(group_a[metric].dropna())
        a_se = a_std / np.sqrt(a_n)
        a_ci = stats.t.interval(0.95, a_n-1, loc=a_mean, scale=a_se)
        
        # B 그룹 신뢰구간
        b_std = group_b[metric].std()
        b_n = len(group_b[metric].dropna())
        b_se = b_std / np.sqrt(b_n)
        b_ci = stats.t.interval(0.95, b_n-1, loc=b_mean, scale=b_se)
        
        # 효과 크기 (Cohen's d)
        pooled_std = np.sqrt(((a_n - 1) * a_std**2 + (b_n - 1) * b_std**2) / (a_n + b_n - 2))
        cohens_d = (b_mean - a_mean) / pooled_std
        
        # 추가 통계 표시
        st.subheader("추가 통계 정보")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**A 그룹 95% 신뢰구간:** [{a_ci[0]:.4f}, {a_ci[1]:.4f}]")
            st.write(f"**표준편차:** {a_std:.4f}")
            st.write(f"**표본 크기:** {a_n}")
        
        with col2:
            st.write(f"**B 그룹 95% 신뢰구간:** [{b_ci[0]:.4f}, {b_ci[1]:.4f}]")
            st.write(f"**표준편차:** {b_std:.4f}")
            st.write(f"**표본 크기:** {b_n}")
        
        st.write(f"**효과 크기 (Cohen's d):** {cohens_d:.4f}")
        
        # 효과 크기 해석
        effect_size_interpretation = ""
        if abs(cohens_d) < 0.2:
            effect_size_interpretation = "작은 효과 (Negligible effect)"
        elif abs(cohens_d) < 0.5:
            effect_size_interpretation = "작은 효과 (Small effect)"
        elif abs(cohens_d) < 0.8:
            effect_size_interpretation = "중간 효과 (Medium effect)"
        else:
            effect_size_interpretation = "큰 효과 (Large effect)"
        
        st.write(f"**효과 크기 해석:** {effect_size_interpretation}")
        
        # 필요한 표본 크기 계산 (검정력 80%, 알파 0.05)
        from statsmodels.stats.power import TTestIndPower
        
        power_analysis = TTestIndPower()
        sample_size = power_analysis.solve_power(
            effect_size=abs(cohens_d),
            power=0.8,
            alpha=0.05,
            ratio=1.0
        )
        
        st.write(f"**검정력 80%를 위한 필요 표본 크기 (각 그룹):** {int(np.ceil(sample_size))}")
        
        if a_n < sample_size or b_n < sample_size:
            st.warning("현재 표본 크기가 충분하지 않습니다. 더 많은 데이터를 수집하여 검정력을 높이세요.")
    
    def _analyze_by_campaign(self, df, group_a, group_b, metric):
        """
        캠페인별 A/B 테스트 분석을 수행합니다.
        
        Args:
            df (pandas.DataFrame): 전체 데이터
            group_a (pandas.DataFrame): A 그룹 데이터
            group_b (pandas.DataFrame): B 그룹 데이터
            metric (str): 분석할 지표
        """
        # 지표 이름 변환
        metric_name = {
            'ctr': '클릭률 (CTR)',
            'cvr': '전환율 (CVR)',
            'roas': '광고 투자 수익률 (ROAS)'
        }.get(metric, metric)
        
        # 캠페인 목록
        campaigns = df['campaign_name'].unique()
        
        # 캠페인별 A/B 테스트 결과
        st.subheader(f"캠페인별 {metric_name} A/B 테스트 결과")
        
        # 결과 저장용 리스트
        results = []
        
        for campaign in campaigns:
            # 캠페인별 데이터
            camp_a = group_a[group_a['campaign_name'] == campaign]
            camp_b = group_b[group_b['campaign_name'] == campaign]
            
            # 충분한 데이터가 있는지 확인
            if len(camp_a) < 2 or len(camp_b) < 2:
                continue
            
            # 평균값
            a_mean = camp_a[metric].mean()
            b_mean = camp_b[metric].mean()
            
            # 변화율
            change_pct = ((b_mean - a_mean) / a_mean) * 100 if a_mean != 0 else 0
            
            # t-test
            try:
                t_stat, p_value = stats.ttest_ind(
                    camp_a[metric].dropna(),
                    camp_b[metric].dropna(),
                    equal_var=False  # Welch's t-test
                )
            except:
                t_stat, p_value = 0, 1
            
            # 효과 크기 (Cohen's d)
            a_std = camp_a[metric].std()
            b_std = camp_b[metric].std()
            a_n = len(camp_a[metric].dropna())
            b_n = len(camp_b[metric].dropna())
            
            try:
                pooled_std = np.sqrt(((a_n - 1) * a_std**2 + (b_n - 1) * b_std**2) / (a_n + b_n - 2))
                cohens_d = (b_mean - a_mean) / pooled_std if pooled_std != 0 else 0
            except:
                cohens_d = 0
            
            # 결과 추가
            results.append({
                'campaign_name': campaign,
                'a_mean': a_mean,
                'b_mean': b_mean,
                'change_pct': change_pct,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'cohens_d': cohens_d,
                'a_sample_size': a_n,
                'b_sample_size': b_n
            })
        
        # 결과가 없는 경우
        if not results:
            st.warning("캠페인별 분석을 위한 충분한 데이터가 없습니다.")
            return
        
        # 결과를 데이터프레임으로 변환
        results_df = pd.DataFrame(results)
        
        # 유의미한 캠페인 수
        significant_count = results_df['significant'].sum()
        total_count = len(results_df)
        
        st.write(f"**유의미한 차이가 있는 캠페인:** {significant_count}/{total_count} ({significant_count/total_count*100:.1f}%)")
        
        # 변화율 기준 정렬
        results_df = results_df.sort_values('change_pct', ascending=False)
        
        # 캠페인별 변화율 시각화
        fig = px.bar(
            results_df,
            x='campaign_name',
            y='change_pct',
            color='significant',
            color_discrete_map={True: '#2ecc71', False: '#95a5a6'},
            text=results_df['change_pct'].apply(lambda x: f"{x:+.1f}%"),
            title=f'캠페인별 {metric_name} 변화율 (%)'
        )
        
        fig.update_layout(
            xaxis_title='캠페인',
            yaxis_title='변화율 (%)',
            height=500,
            legend_title_text='통계적 유의성',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # A/B 그룹 비교 시각화
        fig2 = px.bar(
            results_df,
            x='campaign_name',
            y=['a_mean', 'b_mean'],
            barmode='group',
            text_auto='.3f',
            title=f'캠페인별 A/B 그룹 {metric_name} 비교',
            labels={'value': metric_name, 'variable': '테스트 그룹', 'campaign_name': '캠페인'},
            color_discrete_map={'a_mean': '#3498db', 'b_mean': '#e74c3c'}
        )
        
        fig2.update_layout(
            height=500,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        )
        
        # 범례 이름 변경
        fig2.data[0].name = 'A 그룹'
        fig2.data[1].name = 'B 그룹'
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # 캠페인별 상세 결과
        st.subheader("캠페인별 상세 결과")
        
        # 테이블 형태로 표시
        display_df = results_df[['campaign_name', 'a_mean', 'b_mean', 'change_pct', 'p_value', 'significant', 'cohens_d']]
        display_df.columns = ['캠페인', 'A 그룹 평균', 'B 그룹 평균', '변화율 (%)', 'p-value', '통계적 유의성', '효과 크기']
        
        st.dataframe(display_df.style.format({
            'A 그룹 평균': '{:.4f}',
            'B 그룹 평균': '{:.4f}',
            '변화율 (%)': '{:+.2f}%',
            'p-value': '{:.4f}',
            '효과 크기': '{:.2f}'
        }).background_gradient(cmap='Blues', subset=['효과 크기'])
        .apply(lambda x: ['background-color: #d5f5e3' if v else '' for v in x['통계적 유의성']], axis=1, subset=['통계적 유의성'])
        )
        
        # 추천 사항
        st.subheader("추천 사항")
        
        # 유의미하고 긍정적인 변화가 있는 캠페인
        positive_df = results_df[(results_df['significant'] == True) & (results_df['change_pct'] > 0)]
        
        if not positive_df.empty:
            st.success("다음 캠페인에서 B 그룹이 통계적으로 유의미한 개선을 보였습니다:")
            
            for _, row in positive_df.iterrows():
                st.write(f"- **{row['campaign_name']}**: {row['change_pct']:+.2f}% 개선 (p-value: {row['p_value']:.4f}, 효과 크기: {row['cohens_d']:.2f})")
            
            st.write("이러한 캠페인에 대해서는 B 그룹의 설정을 모든 트래픽에 적용하는 것을 고려해보세요.")
        
        # 유의미하고 부정적인 변화가 있는 캠페인
        negative_df = results_df[(results_df['significant'] == True) & (results_df['change_pct'] < 0)]
        
        if not negative_df.empty:
            st.error("다음 캠페인에서 B 그룹이 통계적으로 유의미한 성능 저하를 보였습니다:")
            
            for _, row in negative_df.iterrows():
                st.write(f"- **{row['campaign_name']}**: {row['change_pct']:+.2f}% 저하 (p-value: {row['p_value']:.4f}, 효과 크기: {row['cohens_d']:.2f})")
            
            st.write("이러한 캠페인에 대해서는 A 그룹의 설정을 유지하고, B 그룹의 설정을 재검토하세요.")
        
        # 표본 크기가 부족한 캠페인
        small_sample_df = results_df[(results_df['a_sample_size'] < 30) | (results_df['b_sample_size'] < 30)]
        
        if not small_sample_df.empty:
            st.warning("다음 캠페인은 충분한 표본 크기가 확보되지 않았습니다:")
            
            for _, row in small_sample_df.iterrows():
                st.write(f"- **{row['campaign_name']}**: A 그룹 {row['a_sample_size']} 샘플, B 그룹 {row['b_sample_size']} 샘플")
            
            st.write("더 정확한 분석을 위해 테스트 기간을 연장하거나 더 많은 데이터를 수집하세요.")
        
        # A/B 테스트 가이드라인
        st.markdown("""
        ### A/B 테스트 가이드라인
        
        1. **충분한 표본 크기 확보**: 각 그룹당 최소 30개 이상의 데이터 포인트가 필요합니다.
        2. **통계적 유의성 확인**: p-value가 0.05 미만일 때 결과를 신뢰할 수 있습니다.
        3. **효과 크기 고려**: 통계적으로 유의미하더라도 효과 크기가 작다면 실질적인 개선이 미미할 수 있습니다.
        4. **여러 지표 확인**: CTR, CVR, ROAS 등 다양한 지표를 함께 확인하세요.
        5. **검정력 계산**: 작은 효과를 감지하려면, 더 많은 데이터가 필요합니다.
        """)


# 모듈 테스트 코드
if __name__ == "__main__":
    # 가상의 A/B 테스트 데이터 생성 (테스트용)
    import pandas as pd
    import numpy as np
    
    # 기본 데이터 생성
    np.random.seed(42)
    
    # 캠페인별 성과 정의
    campaigns = [f"campaign_{i}" for i in range(1, 6)]
    
    # 데이터 프레임 생성
    test_data = []
    
    for campaign in campaigns:
        # 캠페인별 기본 성과
        base_ctr = np.random.uniform(0.01, 0.05)
        base_cvr = np.random.uniform(0.05, 0.15)
        base_roas = np.random.uniform(1.5, 4.0)
        
        # A 그룹 데이터
        for _ in range(50):
            test_data.append({
                'campaign_name': campaign,
                'test_group': 'A',
                'ctr': base_ctr + np.random.normal(0, 0.005),
                'cvr': base_cvr + np.random.normal(0, 0.01),
                'roas': base_roas + np.random.normal(0, 0.3)
            })
        
        # B 그룹 데이터 (일부 캠페인에서는 유의미한 개선, 일부는 저하)
        effect = np.random.choice([0.2, -0.15, 0.1, 0, -0.05])
        
        for _ in range(50):
            test_data.append({
                'campaign_name': campaign,
                'test_group': 'B',
                'ctr': base_ctr * (1 + effect * 0.5) + np.random.normal(0, 0.005),
                'cvr': base_cvr * (1 + effect) + np.random.normal(0, 0.01),
                'roas': base_roas * (1 + effect * 1.2) + np.random.normal(0, 0.3)
            })
    
    test_df = pd.DataFrame(test_data)
    
    print(f"테스트 데이터 생성 완료: {len(test_df)} 행")
    print(f"캠페인 수: {len(test_df['campaign_name'].unique())}")
    print(f"A 그룹: {(test_df['test_group'] == 'A').sum()} 행")
    print(f"B 그룹: {(test_df['test_group'] == 'B').sum()} 행")
    
    # 여기서는 streamlit이 없으므로 분석은 실행하지 않음
    print("실제 앱에서 ABTestAnalyzer를 사용해 분석을 수행하세요.")
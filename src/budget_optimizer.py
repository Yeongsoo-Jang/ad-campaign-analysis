import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
import logging
import os

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/budget_optimizer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("budget_optimizer")

class BudgetOptimizer:
    """
    광고 캠페인 데이터를 분석하고 통계적 근거를 바탕으로 예산 최적화 권장 사항을 제공하는 클래스
    """
    
    def __init__(self):
        """
        BudgetOptimizer 클래스 초기화
        """
        # 로그 디렉토리 생성
        if not os.path.exists("logs"):
            os.makedirs("logs")
            logger.info("로그 디렉토리 생성됨")
    
    def analyze_and_visualize(self, df):
        """
        캠페인 데이터를 분석하고 예산 최적화 시각화를 생성합니다.
        
        Args:
            df (pandas.DataFrame): 광고 캠페인 데이터
        """
        if df.empty:
            st.warning("데이터가 없습니다. 필터 조건을 변경해보세요.")
            return
        
        # 원본 데이터 가져오기 (있는 경우)
        if hasattr(st.session_state, 'original_df'):
            original_df = st.session_state.original_df
            use_original = True
        else:
            original_df = df
            use_original = False
        
        # 캠페인 이름을 키로 사용하여 원본 데이터와 분석 데이터 매핑
        campaign_names = df['campaign_name'].unique()
        
        # 캠페인별 성과 집계 (분석용 - 전처리된 데이터)
        campaign_perf = df.groupby('campaign_name').agg({
            'spend': 'sum',
            'revenue': 'sum',
            'conversions': 'sum',
            'date': 'count'  # 데이터 포인트 수 (신뢰도 계산에 사용)
        }).reset_index()
        
        # 원본 데이터로 캠페인별 성과 집계 (표시용)
        if use_original:
            original_campaign_perf = original_df.groupby('campaign_name').agg({
                'spend': 'sum',
                'revenue': 'sum',
                'conversions': 'sum',
                'date': 'count'  # 데이터 포인트 수 (신뢰도 계산에 사용)
            }).reset_index()
            
            # 두 데이터프레임 병합
            campaign_perf = pd.merge(
                campaign_perf, 
                original_campaign_perf, 
                on='campaign_name', 
                suffixes=('', '_original')
            )
        
        # ROAS 및 기타 지표 계산 (분석용)
        campaign_perf['roas'] = campaign_perf['revenue'] / campaign_perf['spend']
        campaign_perf['cpa'] = campaign_perf['spend'] / campaign_perf['conversions'].apply(lambda x: max(x, 1))  # 0으로 나누기 방지
        
        # 원본 데이터 지표 계산 (표시용)
        if use_original:
            campaign_perf['roas_original'] = campaign_perf['revenue_original'] / campaign_perf['spend_original']
            campaign_perf['cpa_original'] = campaign_perf['spend_original'] / campaign_perf['conversions_original'].apply(lambda x: max(x, 1))
        
        # 효율성 점수 계산 (ROAS 기반 - 분석용)
        mean_roas = campaign_perf['roas'].mean()
        campaign_perf['efficiency'] = campaign_perf['roas'] / mean_roas
        
        # 예산 조정 권장 비율 계산 (표시용 지표는 원본 사용)
        campaign_perf['budget_adjustment'] = (campaign_perf['efficiency'] - 1) * 100
        
        # 통계적 유의성 테스트를 위한 데이터 준비
        campaign_stats = self._calculate_campaign_stats(df, campaign_perf)
        
        # 통계적 유의성 계산 (t-test)
        campaign_stats = self._perform_significance_tests(campaign_stats, mean_roas)
        
        # 신뢰구간 계산
        campaign_stats = self._calculate_confidence_intervals(campaign_stats)
        
        # 통계 데이터를 campaign_perf에 추가
        campaign_perf = self._add_stats_to_performance_data(campaign_perf, campaign_stats)
        
        # 효과 크기 계산 (Cohen's d)
        campaign_perf['effect_size'] = campaign_perf.apply(
            lambda row: (row['mean_daily_roas'] - mean_roas) / row['std_roas'] 
            if row['std_roas'] > 0 else np.nan, 
            axis=1
        )
        
        # 신뢰도 점수 계산
        campaign_perf = self._calculate_confidence_scores(campaign_perf)
        
        # 결과 정렬
        campaign_perf = campaign_perf.sort_values('efficiency', ascending=False)
        
        # 시각화 및 분석 결과 표시 시 원본 수치 사용
        display_columns = ['roas', 'cpa']
        if use_original:
            for col in display_columns:
                if f'{col}_original' in campaign_perf.columns:
                    campaign_perf[f'{col}_display'] = campaign_perf[f'{col}_original']
                else:
                    campaign_perf[f'{col}_display'] = campaign_perf[col]
        else:
            for col in display_columns:
                campaign_perf[f'{col}_display'] = campaign_perf[col]
        
        # 결과 시각화 및 표시 (원본 수치 사용)
        self._display_optimization_results(campaign_perf, mean_roas)
        
        return campaign_perf
    
    def _calculate_campaign_stats(self, df, campaign_perf):
        """
        각 캠페인의 통계 데이터를 계산합니다.
        
        Args:
            df (pandas.DataFrame): 원본 광고 캠페인 데이터
            campaign_perf (pandas.DataFrame): 집계된 캠페인 성과 데이터
            
        Returns:
            dict: 캠페인별 통계 데이터
        """
        campaign_stats = {}
        
        for campaign in campaign_perf['campaign_name']:
            camp_data = df[df['campaign_name'] == campaign]
            
            # 날짜별 ROAS 계산
            daily_roas = []
            for date in camp_data['date'].unique():
                date_data = camp_data[camp_data['date'] == date]
                date_roas = date_data['revenue'].sum() / date_data['spend'].sum() if date_data['spend'].sum() > 0 else 0
                daily_roas.append(date_roas)
            
            # 통계 계산
            campaign_stats[campaign] = {
                'mean_roas': np.mean(daily_roas) if daily_roas else 0,
                'std_roas': np.std(daily_roas) if len(daily_roas) > 1 else 0,
                'n_days': len(daily_roas),
                'daily_roas': daily_roas
            }
        
        return campaign_stats
    
    def _perform_significance_tests(self, campaign_stats, mean_roas):
        """
        각 캠페인의 통계적 유의성 테스트를 수행합니다.
        
        Args:
            campaign_stats (dict): 캠페인별 통계 데이터
            mean_roas (float): 전체 평균 ROAS
            
        Returns:
            dict: 통계적 유의성 테스트 결과가 추가된 캠페인 통계 데이터
        """
        from scipy import stats  # 여기서 scipy.stats 모듈 import
        
        for campaign, stat in campaign_stats.items():
            if stat['n_days'] > 1:  # t-test에는 최소 2개의 데이터 포인트가 필요
                # 평균 ROAS와의 t-test
                t_stat, p_value = stats.ttest_1samp(
                    stat['daily_roas'], 
                    mean_roas
                )
                stat['t_stat'] = t_stat
                stat['p_value'] = p_value
                
                # 유의성 판단 (p < 0.05)
                stat['significant'] = p_value < 0.05
            else:
                stat['t_stat'] = None
                stat['p_value'] = None
                stat['significant'] = False
        
        return campaign_stats
    
    def _calculate_confidence_intervals(self, campaign_stats):
        """
        각 캠페인의 95% 신뢰구간을 계산합니다.
        
        Args:
            campaign_stats (dict): 캠페인별 통계 데이터
            
        Returns:
            dict: 신뢰구간이 추가된 캠페인 통계 데이터
        """
        from scipy import stats  # scipy.stats 모듈 import
        
        for campaign, stat in campaign_stats.items():
            if stat['n_days'] > 1:
                std = stat['std_roas']
                n = stat['n_days']
                mean = stat['mean_roas']
                
                # 95% 신뢰구간
                t_critical = stats.t.ppf(0.975, n-1)  # 95% 신뢰수준에서의 t 임계값
                margin_error = t_critical * (std / np.sqrt(n))
                
                stat['ci_lower'] = mean - margin_error
                stat['ci_upper'] = mean + margin_error
            else:
                stat['ci_lower'] = None
                stat['ci_upper'] = None
        
        return campaign_stats
    
    def _add_stats_to_performance_data(self, campaign_perf, campaign_stats):
        """
        통계 데이터를 캠페인 성과 데이터프레임에 추가합니다.
        
        Args:
            campaign_perf (pandas.DataFrame): 집계된 캠페인 성과 데이터
            campaign_stats (dict): 캠페인별 통계 데이터
            
        Returns:
            pandas.DataFrame: 통계 데이터가 추가된 캠페인 성과 데이터
        """
        for campaign in campaign_perf['campaign_name']:
            idx = campaign_perf[campaign_perf['campaign_name'] == campaign].index[0]
            campaign_perf.loc[idx, 'mean_daily_roas'] = campaign_stats[campaign]['mean_roas']
            campaign_perf.loc[idx, 'std_roas'] = campaign_stats[campaign]['std_roas']
            campaign_perf.loc[idx, 'p_value'] = campaign_stats[campaign].get('p_value', np.nan)
            campaign_perf.loc[idx, 'significant'] = campaign_stats[campaign].get('significant', False)
            campaign_perf.loc[idx, 'ci_lower'] = campaign_stats[campaign].get('ci_lower', np.nan)
            campaign_perf.loc[idx, 'ci_upper'] = campaign_stats[campaign].get('ci_upper', np.nan)
            campaign_perf.loc[idx, 'n_days'] = campaign_stats[campaign]['n_days']
        
        return campaign_perf
    
    def _calculate_confidence_scores(self, campaign_perf):
        """
        각 캠페인의 신뢰도 점수를 계산합니다.
        
        Args:
            campaign_perf (pandas.DataFrame): 통계 데이터가 추가된 캠페인 성과 데이터
            
        Returns:
            pandas.DataFrame: 신뢰도 점수가 추가된 캠페인 성과 데이터
        """
        # 1. 데이터 포인트 수 (더 많은 데이터 = 더 높은 신뢰도)
        max_data_points = campaign_perf['n_days'].max()
        campaign_perf['data_score'] = campaign_perf['n_days'] / max_data_points if max_data_points > 0 else 0
        
        # 2. 지출 규모 (더 많은 지출 = 더 높은 신뢰도)
        max_spend = campaign_perf['spend'].max()
        campaign_perf['spend_score'] = campaign_perf['spend'] / max_spend if max_spend > 0 else 0
        
        # 3. 통계적 유의성 (p-value가 낮을수록 = 더 높은 신뢰도)
        campaign_perf['stat_score'] = campaign_perf.apply(
            lambda row: 1 - min(row['p_value'], 0.99) if not pd.isna(row['p_value']) else 0,
            axis=1
        )
        
        # 종합 신뢰도 점수 계산 (0-100%)
        campaign_perf['confidence_score'] = (
            campaign_perf['data_score'] * 0.25 +  # 데이터 양
            campaign_perf['spend_score'] * 0.25 +  # 지출 규모
            campaign_perf['stat_score'] * 0.5   # 통계적 유의성
        ) * 100
        
        return campaign_perf
    
    def _display_optimization_results(self, campaign_perf, mean_roas):
        """
        분석 결과와 시각화를 화면에 표시합니다.
        
        Args:
            campaign_perf (pandas.DataFrame): 분석이 완료된 캠페인 성과 데이터
            mean_roas (float): 전체 평균 ROAS
        """
        
        # 원본 데이터로 시각화
        if 'original_roas' in campaign_perf.columns:
            campaign_perf['roas_display'] = campaign_perf['original_roas']
            mean_roas_display = campaign_perf['original_roas'].mean()
        else:
            campaign_perf['roas_display'] = campaign_perf['roas']
            mean_roas_display = mean_roas
            
        # 요약 통계 표시
        st.subheader("예산 조정 분석 요약")
        
        # 평균 ROAS 및 분포 요약
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("평균 ROAS", f"{mean_roas:.2f}")
        with col2:
            st.metric("최고 캠페인 ROAS", f"{campaign_perf['roas'].max():.2f}")
        with col3:
            st.metric("최저 캠페인 ROAS", f"{campaign_perf['roas'].min():.2f}")
        
        # 통계적 분석 결과 요약
        st.markdown("### 통계적 분석 결과")
        
        # 통계적으로 유의미한 캠페인 수
        significant_campaigns = campaign_perf[campaign_perf['significant'] == True]
        st.write(f"**통계적으로 유의미한 캠페인 수:** {len(significant_campaigns)}개 / 전체 {len(campaign_perf)}개")
        
        # 가장 효율적인 캠페인과 가장 비효율적인 캠페인
        if not campaign_perf.empty:
            best_campaign = campaign_perf.iloc[0]
            worst_campaign = campaign_perf.iloc[-1]
            
            st.write(f"**가장 효율적인 캠페인:** {best_campaign['campaign_name']} (ROAS: {best_campaign['roas']:.2f}, 권장 예산 조정: {best_campaign['budget_adjustment']:.1f}%)")
            
            if best_campaign['significant']:
                st.write(f"이 캠페인은 통계적으로 유의미하게 평균보다 좋은 성과를 보입니다 (p-value: {best_campaign['p_value']:.4f}).")
            
            st.write(f"**가장 비효율적인 캠페인:** {worst_campaign['campaign_name']} (ROAS: {worst_campaign['roas']:.2f}, 권장 예산 조정: {worst_campaign['budget_adjustment']:.1f}%)")
            
            if worst_campaign['significant']:
                st.write(f"이 캠페인은 통계적으로 유의미하게 평균보다 나쁜 성과를 보입니다 (p-value: {worst_campaign['p_value']:.4f}).")
        
        # 시각화 - 권장 예산 조정 비율 (통계적 유의성 표시)
        fig = px.bar(
            campaign_perf,
            x='campaign_name',
            y='budget_adjustment',
            error_y=campaign_perf.apply(
                lambda row: abs(row['budget_adjustment']) * 0.5 if pd.isna(row['ci_upper']) 
                else abs((row['ci_upper'] / row['mean_daily_roas'] - 1) * 100 - row['budget_adjustment']), 
                axis=1
            ),
            color='significant',
            color_discrete_map={True: '#2ecc71', False: '#95a5a6'},
            text=campaign_perf.apply(
                lambda row: f"{row['budget_adjustment']:.1f}%", 
                axis=1
            ),
            title='캠페인별 권장 예산 조정 비율 (%)',
            hover_data={
                'campaign_name': True,
                'roas': True,
                'significant': False,  # 직접 표시하지 않고
                'p_value': ':.4f',     # p-value를 표시
                'confidence_score': ':.1f%'  # 신뢰도 점수도 표시
            },
            hover_name='campaign_name'  # 캠페인 이름을 툴팁 제목으로
        )
        
        fig.update_layout(
            xaxis_title='캠페인',
            yaxis_title='예산 조정 권장 비율 (%)',
            height=600,
            xaxis={'categoryorder': 'total descending'},
            legend_title_text='통계적 유의성',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        )
        
        # 통계적 유의성에 대한 표기 추가
        fig.update_traces(
            hovertemplate='<b>%{hovertext}</b><br>' +
                        'ROAS: %{customdata[0]:.2f}<br>' +
                        '예산 조정: %{y:.1f}%<br>' +
                        'p-value: %{customdata[1]}<br>' +
                        '신뢰도: %{customdata[2]}<br>' +
                        '통계적 유의성: %{customdata[3]}'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 신뢰도 점수 시각화
        fig2 = px.bar(
            campaign_perf,
            x='campaign_name',
            y='confidence_score',
            text=campaign_perf['confidence_score'].apply(lambda x: f"{x:.1f}%"),
            color='confidence_score',
            color_continuous_scale='Blues',
            title='캠페인별 예산 조정 신뢰도 점수 (%)'
        )
        
        fig2.update_layout(
            xaxis_title='캠페인',
            yaxis_title='신뢰도 점수 (%)',
            height=500,
            xaxis={'categoryorder': 'array', 'categoryarray': campaign_perf['campaign_name'].tolist()}  # 첫 번째 그래프와 같은 순서로 정렬
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # ROAS와 95% 신뢰구간 시각화
        fig3 = go.Figure()
        
        # ROAS 값
        fig3.add_trace(go.Scatter(
            x=campaign_perf['campaign_name'],
            y=campaign_perf['roas'],
            mode='markers',
            name='ROAS',
            marker=dict(size=10, color='blue')
        ))
        
        # 신뢰구간
        for i, row in campaign_perf.iterrows():
            if not pd.isna(row['ci_lower']) and not pd.isna(row['ci_upper']):
                fig3.add_trace(go.Scatter(
                    x=[row['campaign_name'], row['campaign_name']],
                    y=[row['ci_lower'], row['ci_upper']],
                    mode='lines',
                    line=dict(width=2, color='rgba(0,0,255,0.3)'),
                    showlegend=False
                ))
        
        # 평균 ROAS 라인
        fig3.add_trace(go.Scatter(
            x=campaign_perf['campaign_name'],
            y=[mean_roas] * len(campaign_perf),
            mode='lines',
            name='평균 ROAS',
            line=dict(width=2, color='red', dash='dash')
        ))
        
        fig3.update_layout(
            title='캠페인별 ROAS와 95% 신뢰구간',
            xaxis_title='캠페인',
            yaxis_title='ROAS',
            height=500,
            xaxis={'categoryorder': 'array', 'categoryarray': campaign_perf['campaign_name'].tolist()}
        )
        
        st.plotly_chart(fig3, use_container_width=True)
        
        # 통계 분석 설명
        st.markdown("""
        ### 통계적 분석 방법 설명
        
        **1. t-검정 (t-test):** 각 캠페인의 ROAS가 전체 평균과 통계적으로 유의미하게 다른지 확인했습니다. p-value가 0.05 미만인 경우 통계적으로 유의미하다고 판단합니다.
        
        **2. 95% 신뢰구간:** 각 캠페인의 실제 ROAS 값이 95% 확률로 존재할 것으로 예상되는 범위를 보여줍니다. 신뢰구간이 전체 평균 ROAS와 겹치지 않는 경우, 해당 캠페인은 통계적으로 유의미하게 다르다고 볼 수 있습니다.
        
        **3. 효과 크기 (Effect Size):** Cohen's d 값으로 표현되며, 평균과의 차이가 얼마나 큰지를 표준화된 방식으로 보여줍니다. 일반적으로 0.2는 작은 효과, 0.5는 중간 효과, 0.8 이상은 큰 효과로 해석됩니다.
        
        **4. 신뢰도 점수:** 데이터 포인트 수(25%), 지출 규모(25%), 통계적 유의성(50%)을 가중 평균하여 각 캠페인 분석 결과의 신뢰도를 0-100% 범위로 계산했습니다.
        """)
        
        # 추천 사항
        st.markdown("### 최종 예산 조정 추천")
        
        # 조정 추천 문장 생성
        if not significant_campaigns.empty:
            # 신뢰도와 조정 비율이 모두 높은 캠페인 선택
            high_confidence = campaign_perf[
                (campaign_perf['confidence_score'] > 70) & 
                (campaign_perf['significant'] == True)
            ]
            
            if not high_confidence.empty:
                increase_campaigns = high_confidence[high_confidence['budget_adjustment'] > 10]
                decrease_campaigns = high_confidence[high_confidence['budget_adjustment'] < -10]
                
                recommendations = []
                
                if not increase_campaigns.empty:
                    top_increase = increase_campaigns.iloc[0]
                    recommendations.append(f"**증액 권장:** {top_increase['campaign_name']} 캠페인의 예산을 {top_increase['budget_adjustment']:.1f}% 증액하세요. (ROAS: {top_increase['roas']:.2f}, 신뢰도: {top_increase['confidence_score']:.1f}%)")
                
                if not decrease_campaigns.empty:
                    top_decrease = decrease_campaigns.iloc[-1]
                    recommendations.append(f"**감액 권장:** {top_decrease['campaign_name']} 캠페인의 예산을 {-top_decrease['budget_adjustment']:.1f}% 감액하세요. (ROAS: {top_decrease['roas']:.2f}, 신뢰도: {top_decrease['confidence_score']:.1f}%)")
                
                if recommendations:
                    for rec in recommendations:
                        st.write(rec)
                else:
                    st.write("현재 데이터에서는 통계적으로 유의미한 예산 조정 권장 사항이 없습니다.")
            else:
                st.write("신뢰도 높은 예산 조정 권장 사항이 없습니다. 더 많은 데이터를 수집하거나 분석 기간을 늘려보세요.")
        else:
            st.write("통계적으로 유의미한 예산 조정 권장 사항이 없습니다. 더 많은 데이터를 수집하거나 분석 기간을 늘려보세요.")
        
        # 설명 추가
        st.info("""
        **예산 최적화 방법:**
        - 양수 값: 해당 캠페인에 더 많은 예산 배정 권장
        - 음수 값: 해당 캠페인의 예산 감소 권장
        - 각 값은 ROAS 대비 효율성을 기준으로 계산되었습니다.
        - '*' 표시는 통계적으로 유의미한 캠페인을 나타냅니다. (p < 0.05)
        
        **신뢰도 점수:**
        - 데이터 포인트 수 (25%): 더 많은 데이터 = 더 높은 신뢰도
        - 지출 규모 (25%): 더 많은 지출 = 더 높은 신뢰도
        - 통계적 유의성 (50%): p-value가 낮을수록 = 더 높은 신뢰도
        - 신뢰도가 높고 통계적으로 유의미한 캠페인의 조정 권장 사항을 우선적으로 고려하세요.
        """)
        
        # 상세 데이터 표시
        st.subheader("캠페인별 성과 및 예산 조정 상세")
        # 'significant' 열을 파이썬 bool 타입으로 변환
        campaign_perf['significant'] = campaign_perf['significant'].astype(bool)

        styled_df = campaign_perf[
            ['campaign_name', 'spend', 'revenue', 'roas', 'efficiency', 'budget_adjustment', 'p_value', 'significant', 'confidence_score']
        ].style.format({
            'spend': '₩{:,.0f}',
            'revenue': '₩{:,.0f}',
            'roas': '{:.2f}',
            'efficiency': '{:.2f}',
            'budget_adjustment': '{:+.1f}%',
            'p_value': '{:.4f}',
            'confidence_score': '{:.1f}%'
        }).background_gradient(cmap='Blues', subset=['confidence_score'])

        # NumPy bool 문제를 해결하기 위해 수정된 방식으로 스타일 적용
        def highlight_significant(x):
            return ['background-color: #d5f5e3' if bool(v) else '' for v in x]

        styled_df = styled_df.apply(highlight_significant, axis=0, subset=['significant'])
        st.dataframe(styled_df)


# 모듈 테스트 코드
if __name__ == "__main__":
    import pandas as pd
    
    # 테스트 데이터 생성
    data = pd.read_csv("data/campaign_data.csv")
    
    # BudgetOptimizer 초기화
    optimizer = BudgetOptimizer()
    
    # Streamlit 앱이 아닌 환경에서 테스트
    print(f"데이터 로드 완료: {len(data)} 행")
    print("BudgetOptimizer 초기화 완료")
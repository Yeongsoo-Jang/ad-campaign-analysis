# src/insight_generator.py
import pandas as pd
import numpy as np
import logging
import os

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/insight_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("insight_generator")

class InsightGenerator:
    """
    데이터를 분석하여 자동으로 인사이트를 생성하는 클래스
    """
    
    def __init__(self):
        """
        InsightGenerator 클래스 초기화
        """
        # 로그 디렉토리 생성
        if not os.path.exists("logs"):
            os.makedirs("logs")
            logger.info("로그 디렉토리 생성됨")
    
    def generate_insights(self, df, target_col='roas', top_n=3):
        """
        데이터를 분석하여 주요 인사이트를 생성합니다.
        
        Args:
            df (pandas.DataFrame): 분석할 데이터
            target_col (str): 타겟 지표 컬럼 (예: 'roas')
            top_n (int): 반환할 상위 인사이트 수
            
        Returns:
            list: 인사이트 텍스트 목록
        """
        insights = []
        
        # 충분한 데이터가 있는지 확인
        if df.empty or len(df) < 5:
            insights.append("충분한 데이터가 없습니다. 더 많은 데이터를 수집하거나 필터 조건을 변경해보세요.")
            return insights
        
        try:
            # 1. 성과 상위/하위 캠페인 식별
            if 'campaign_name' in df.columns and target_col in df.columns:
                campaign_insights = self._get_campaign_insights(df, target_col)
                insights.extend(campaign_insights)
            
            # 2. 트렌드 인사이트 (시계열 데이터가 있는 경우)
            if 'date' in df.columns and target_col in df.columns:
                trend_insights = self._get_trend_insights(df, target_col)
                insights.extend(trend_insights)
            
            # 3. 그룹별 인사이트 (플랫폼, 크리에이티브 등)
            group_cols = ['platform', 'creative_type', 'target_age', 'target_gender']
            for col in group_cols:
                if col in df.columns:
                    group_insights = self._get_group_insights(df, col, target_col)
                    insights.extend(group_insights)
            
            # 4. 상관관계 인사이트
            corr_insights = self._get_correlation_insights(df, target_col)
            insights.extend(corr_insights)
            
            # 5. 이상치/특이점 인사이트
            anomaly_insights = self._get_anomaly_insights(df, target_col)
            insights.extend(anomaly_insights)
            
            # 중요도에 따라 인사이트 정렬 및 상위 N개 선택
            insights = sorted(insights, key=lambda x: x.get('importance', 0), reverse=True)
            top_insights = insights[:top_n]
            
            # 인사이트 텍스트만 추출
            return [insight['text'] for insight in top_insights]
            
        except Exception as e:
            logger.error(f"인사이트 생성 중 오류 발생: {e}")
            return ["데이터 분석 중 오류가 발생했습니다."]
    
    def _get_campaign_insights(self, df, target_col):
        """
        캠페인 성과에 관한 인사이트를 생성합니다.
        """
        insights = []
        
        try:
            # 캠페인별 타겟 지표 집계
            campaign_perf = df.groupby('campaign_name')[target_col].mean().reset_index()
            
            # 상위/하위 캠페인
            top_campaign = campaign_perf.nlargest(1, target_col).iloc[0]
            bottom_campaign = campaign_perf.nsmallest(1, target_col).iloc[0]
            
            # 전체 평균과 비교
            avg_value = campaign_perf[target_col].mean()
            top_vs_avg = (top_campaign[target_col] / avg_value - 1) * 100
            bottom_vs_avg = (bottom_campaign[target_col] / avg_value - 1) * 100
            
            # 인사이트 생성
            insights.append({
                'text': f"최고 성과 캠페인은 {top_campaign['campaign_name']}으로, {target_col.upper()}가 {top_campaign[target_col]:.2f}이며 평균보다 {top_vs_avg:.1f}% 높습니다. 이 캠페인의 전략을 다른 캠페인에 적용해 볼 가치가 있습니다.",
                'importance': abs(top_vs_avg) / 10,  # 중요도 지표: 평균과의 차이가 클수록 중요
                'type': 'top_performer'
            })
            
            insights.append({
                'text': f"가장 개선이 필요한 캠페인은 {bottom_campaign['campaign_name']}으로, {target_col.upper()}가 {bottom_campaign[target_col]:.2f}이며 평균보다 {abs(bottom_vs_avg):.1f}% 낮습니다. 이 캠페인의 타겟팅, 크리에이티브, 입찰 전략을 검토할 필요가 있습니다.",
                'importance': abs(bottom_vs_avg) / 15,  # 중요도: 하위 성과는 상위보다 약간 낮게 평가
                'type': 'bottom_performer'
            })
            
            # 변동성 분석
            if len(campaign_perf) >= 5:  # 충분한 캠페인이 있는 경우
                std_value = campaign_perf[target_col].std()
                cv = std_value / avg_value  # 변동 계수
                
                # 변동성이 큰 경우 인사이트 추가
                if cv > 0.3:  # 변동 계수가 30% 이상인 경우
                    insights.append({
                        'text': f"캠페인 간 {target_col.upper()} 변동성이 큽니다(CV: {cv:.2f}). 성과가 불안정한 캠페인을 식별하고 원인을 파악하는 것이 중요합니다.",
                        'importance': cv * 10,
                        'type': 'variability'
                    })
        except Exception as e:
            logger.error(f"캠페인 인사이트 생성 중 오류 발생: {e}")
        
        return insights
    
    def _get_trend_insights(self, df, target_col):
        """
        시간에 따른 추세에 관한 인사이트를 생성합니다.
        """
        insights = []
        
        try:
            # 날짜별 타겟 지표 집계
            if len(df['date'].unique()) >= 5:  # 충분한 날짜 데이터가 있는 경우
                daily_data = df.groupby('date')[target_col].mean().reset_index()
                daily_data = daily_data.sort_values('date')
                
                # 전반부와 후반부 비교
                mid_point = len(daily_data) // 2
                first_half = daily_data.iloc[:mid_point][target_col].mean()
                second_half = daily_data.iloc[mid_point:][target_col].mean()
                
                change_pct = (second_half / first_half - 1) * 100
                
                # 추세 인사이트 생성 (변화율이 5% 이상인 경우)
                if abs(change_pct) >= 5:
                    direction = "상승" if change_pct > 0 else "하락"
                    insights.append({
                        'text': f"기간 동안 {target_col.upper()}가 {abs(change_pct):.1f}% {direction}하는 추세를 보입니다. 이 변화의 원인을 파악하여 전략적으로 대응할 필요가 있습니다.",
                        'importance': abs(change_pct) / 5,
                        'type': 'trend'
                    })
                
                # 급격한 변화 감지
                daily_data['pct_change'] = daily_data[target_col].pct_change()
                large_changes = daily_data[abs(daily_data['pct_change']) > 0.2].dropna()  # 20% 이상 변화
                
                if not large_changes.empty:
                    largest_change = large_changes.iloc[large_changes['pct_change'].abs().argmax()]
                    direction = "증가" if largest_change['pct_change'] > 0 else "감소"
                    change_date = largest_change['date'].strftime('%Y-%m-%d')
                    
                    insights.append({
                        'text': f"{change_date}에 {target_col.upper()}가 {abs(largest_change['pct_change']) * 100:.1f}% 급격히 {direction}했습니다. 이 날짜 주변에 발생한 이벤트나 변경사항을 확인해 보세요.",
                        'importance': abs(largest_change['pct_change']) * 100,
                        'type': 'spike'
                    })
        except Exception as e:
            logger.error(f"트렌드 인사이트 생성 중 오류 발생: {e}")
        
        return insights
    
    def _get_group_insights(self, df, group_col, target_col):
        """
        그룹별(플랫폼, 크리에이티브 유형 등) 성과에 관한 인사이트를 생성합니다.
        """
        insights = []
        
        try:
            # 그룹별 타겟 지표 집계
            if len(df[group_col].unique()) >= 2:  # 최소 2개 이상의 그룹
                group_perf = df.groupby(group_col)[target_col].mean().reset_index()
                
                # 최고/최저 성과 그룹
                top_group = group_perf.nlargest(1, target_col).iloc[0]
                bottom_group = group_perf.nsmallest(1, target_col).iloc[0]
                
                # 차이 계산
                diff_pct = (top_group[target_col] / bottom_group[target_col] - 1) * 100
                
                # 차이가 클 경우 인사이트 생성 (20% 이상)
                if diff_pct >= 20:
                    # 그룹 유형에 따른 한글 레이블
                    group_type = {
                        'platform': '플랫폼',
                        'creative_type': '크리에이티브 유형',
                        'target_age': '타겟 연령대',
                        'target_gender': '타겟 성별'
                    }.get(group_col, group_col)
                    
                    insights.append({
                        'text': f"최고 성과 {group_type}({top_group[group_col]})과 최저 성과 {group_type}({bottom_group[group_col]}) 간의 {target_col.upper()} 차이가 {diff_pct:.1f}%로 상당합니다. 리소스를 최고 성과 {group_type}에 집중하고, 저성과 {group_type}의 전략을 재검토하세요.",
                        'importance': diff_pct / 20,
                        'type': f'{group_col}_comparison'
                    })
        except Exception as e:
            logger.error(f"그룹 인사이트 생성 중 오류 발생: {e}")
        
        return insights
    
    def _get_correlation_insights(self, df, target_col):
        """
        다양한 지표와 타겟 지표 간의 상관관계에 관한 인사이트를 생성합니다.
        """
        insights = []
        
        try:
            # 수치형 컬럼만 선택
            numeric_cols = df.select_dtypes(include=['number']).columns
            
            # 타겟 컬럼과 다른 수치형 컬럼 간의 상관관계 계산
            corr_cols = [col for col in numeric_cols if col != target_col]
            
            if corr_cols:
                correlations = {}
                for col in corr_cols:
                    if col in ['spend', 'impressions', 'clicks', 'ctr', 'conversions', 'cvr']:
                        corr = df[[target_col, col]].corr().iloc[0, 1]
                        correlations[col] = corr
                
                # 가장 강한 양/음의 상관관계
                if correlations:
                    abs_corr = {k: abs(v) for k, v in correlations.items()}
                    strongest_corr_col = max(abs_corr, key=abs_corr.get)
                    strongest_corr = correlations[strongest_corr_col]
                    
                    # 상관관계가 강한 경우만 인사이트 생성 (0.5 이상)
                    if abs(strongest_corr) >= 0.5:
                        direction = "양의" if strongest_corr > 0 else "음의"
                        
                        # 지표 한글 레이블
                        metric_label = {
                            'spend': '지출',
                            'impressions': '노출 수',
                            'clicks': '클릭 수',
                            'ctr': '클릭률',
                            'conversions': '전환 수',
                            'cvr': '전환율'
                        }.get(strongest_corr_col, strongest_corr_col)
                        
                        insights.append({
                            'text': f"{metric_label}와 {target_col.upper()} 간에 {abs(strongest_corr):.2f}의 {direction} 상관관계가 있습니다. 이는 {metric_label}{'이' if metric_label[-1] not in ['요', '수'] else '가'} {'증가할수록' if strongest_corr > 0 else '감소할수록'} {target_col.upper()}{'이' if target_col[-1] not in ['요', '수'] else '가'} {'증가하는' if strongest_corr > 0 else '감소하는'} 경향이 있음을 의미합니다.",
                            'importance': abs(strongest_corr) * 10,
                            'type': 'correlation'
                        })
        except Exception as e:
            logger.error(f"상관관계 인사이트 생성 중 오류 발생: {e}")
        
        return insights
    
    def _get_anomaly_insights(self, df, target_col):
        """
        이상치나 특이점에 관한 인사이트를 생성합니다.
        """
        insights = []
        
        try:
            if len(df) >= 10 and target_col in df.columns:
                # 타겟 지표의 사분위수 계산
                q1 = df[target_col].quantile(0.25)
                q3 = df[target_col].quantile(0.75)
                iqr = q3 - q1
                
                # 이상치 경계 설정
                upper_bound = q3 + 1.5 * iqr
                lower_bound = q1 - 1.5 * iqr
                
                # 이상치 식별
                upper_outliers = df[df[target_col] > upper_bound]
                lower_outliers = df[df[target_col] < lower_bound]
                
                # 상위 이상치 인사이트
                if not upper_outliers.empty:
                    num_outliers = len(upper_outliers)
                    
                    # 이상치가 특정 캠페인에 집중되어 있는지 확인
                    if 'campaign_name' in upper_outliers.columns:
                        campaign_counts = upper_outliers['campaign_name'].value_counts()
                        if not campaign_counts.empty:
                            top_campaign = campaign_counts.index[0]
                            top_count = campaign_counts.iloc[0]
                            
                            if top_count >= 2 and top_count / num_outliers >= 0.5:
                                insights.append({
                                    'text': f"{top_campaign} 캠페인이 {num_outliers}개의 상위 {target_col.upper()} 이상치 중 {top_count}개를 차지합니다. 이 캠페인의 성공 요인을 분석하여 다른 캠페인에 적용해 보세요.",
                                    'importance': top_count / num_outliers * 10,
                                    'type': 'outlier_campaign'
                                })
                
                # 하위 이상치 인사이트
                if not lower_outliers.empty:
                    num_outliers = len(lower_outliers)
                    
                    insights.append({
                        'text': f"{target_col.upper()} 하위 이상치가 {num_outliers}개 발견되었습니다. 이는 특정 조건에서 성과가 현저히 떨어지는 경우가 있음을 의미합니다. 이 데이터 포인트들의 공통점을 분석해 보세요.",
                        'importance': num_outliers / len(df) * 15,
                        'type': 'lower_outliers'
                    })
        except Exception as e:
            logger.error(f"이상치 인사이트 생성 중 오류 발생: {e}")
        
        return insights
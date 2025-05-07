import numpy as np
import os
import logging
import yaml
import jinja2
import smtplib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
# from weasyprint import HTML

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/reporter.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("reporter")

class ReportGenerator:
    """
    광고 캠페인 데이터를 기반으로 보고서를 생성하고 이메일로 전송하는 클래스
    """
    
    def __init__(self, config_path="config/config.yaml"):
        """
        설정 파일을 로드하고 초기화합니다.
        
        Args:
            config_path (str): 설정 파일 경로
        """
        self.config = self._load_config(config_path)
        self.report_config = self.config.get('reporting', {})
        self.email_config = self.config.get('email', {})
        self.reports_dir = "reports"
        self.templates_dir = "templates"
        self.ensure_dirs()
    
    def _load_config(self, config_path):
        """
        YAML 설정 파일을 로드합니다.
        
        Args:
            config_path (str): 설정 파일 경로
            
        Returns:
            dict: 설정 데이터
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"설정 파일 로드 중 오류 발생: {e}")
            return {}
    
    def ensure_dirs(self):
        """
        필요한 디렉토리가 존재하는지 확인하고, 없다면 생성합니다.
        """
        for dir_path in [self.reports_dir, self.templates_dir, "logs"]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                logger.info(f"디렉토리 생성됨: {dir_path}")
    
    def create_summary_report(self, df, output_format='pdf', email_recipients=None):
        """
        광고 캠페인 데이터 요약 보고서를 생성합니다.
        
        Args:
            df (pandas.DataFrame): 광고 캠페인 데이터
            output_format (str): 출력 형식 ('pdf', 'html', 'email')
            email_recipients (list, optional): 보고서를 전송할 이메일 주소 목록
            
        Returns:
            str: 생성된 보고서 파일 경로 또는 이메일 전송 성공 여부
        """
        try:
            logger.info("요약 보고서 생성 시작")
            
            # 날짜 범위 정보
            start_date = df['date'].min().strftime('%Y-%m-%d')
            end_date = df['date'].max().strftime('%Y-%m-%d')
            date_period = f"{start_date} ~ {end_date}"
            
            # 전체 요약 데이터
            total_spend = df['spend'].sum()
            total_impressions = df['impressions'].sum()
            total_clicks = df['clicks'].sum()
            total_conversions = df['conversions'].sum()
            total_revenue = df['revenue'].sum()
            
            # 평균값
            avg_ctr = (total_clicks / total_impressions) * 100 if total_impressions > 0 else 0
            avg_cvr = (total_conversions / total_clicks) * 100 if total_clicks > 0 else 0
            avg_roas = (total_revenue / total_spend) if total_spend > 0 else 0
            avg_cpc = (total_spend / total_clicks) if total_clicks > 0 else 0
            avg_cpa = (total_spend / total_conversions) if total_conversions > 0 else 0
            
            # 플랫폼별 요약
            platform_summary = df.groupby('platform').agg({
                'spend': 'sum',
                'impressions': 'sum',
                'clicks': 'sum',
                'conversions': 'sum',
                'revenue': 'sum'
            }).reset_index()
            
            # 파생 지표 계산
            platform_summary['ctr'] = (platform_summary['clicks'] / platform_summary['impressions']) * 100
            platform_summary['cvr'] = (platform_summary['conversions'] / platform_summary['clicks']) * 100
            platform_summary['roas'] = platform_summary['revenue'] / platform_summary['spend']
            platform_summary['cpc'] = platform_summary['spend'] / platform_summary['clicks']
            platform_summary['cpa'] = platform_summary['spend'] / platform_summary['conversions']
            
            # 일별 트렌드
            daily_trend = df.groupby('date').agg({
                'spend': 'sum',
                'impressions': 'sum',
                'clicks': 'sum',
                'conversions': 'sum',
                'revenue': 'sum'
            }).reset_index()
            
            daily_trend['ctr'] = (daily_trend['clicks'] / daily_trend['impressions']) * 100
            daily_trend['cvr'] = (daily_trend['conversions'] / daily_trend['clicks']) * 100
            daily_trend['roas'] = daily_trend['revenue'] / daily_trend['spend']
            
            # 크리에이티브 유형별 성과
            creative_summary = df.groupby('creative_type').agg({
                'spend': 'sum',
                'impressions': 'sum',
                'clicks': 'sum',
                'conversions': 'sum',
                'revenue': 'sum'
            }).reset_index()
            
            creative_summary['ctr'] = (creative_summary['clicks'] / creative_summary['impressions']) * 100
            creative_summary['cvr'] = (creative_summary['conversions'] / creative_summary['clicks']) * 100
            creative_summary['roas'] = creative_summary['revenue'] / creative_summary['spend']
            
            # 타겟 연령별 성과
            age_summary = df.groupby('target_age').agg({
                'spend': 'sum',
                'conversions': 'sum',
                'revenue': 'sum'
            }).reset_index()
            
            age_summary['roas'] = age_summary['revenue'] / age_summary['spend']
            age_summary['cpa'] = age_summary['spend'] / age_summary['conversions']
            
            # 타겟 성별별 성과
            gender_summary = df.groupby('target_gender').agg({
                'spend': 'sum',
                'conversions': 'sum',
                'revenue': 'sum'
            }).reset_index()
            
            gender_summary['roas'] = gender_summary['revenue'] / gender_summary['spend']
            gender_summary['cpa'] = gender_summary['spend'] / gender_summary['conversions']
            
            # 상위 캠페인
            campaign_summary = df.groupby(['campaign_name', 'platform']).agg({
                'spend': 'sum',
                'impressions': 'sum',
                'clicks': 'sum',
                'conversions': 'sum',
                'revenue': 'sum'
            }).reset_index()
            
            campaign_summary['roas'] = campaign_summary['revenue'] / campaign_summary['spend']
            campaign_summary = campaign_summary.sort_values('roas', ascending=False).head(5)
            
            # 그래프 생성
            # 1. 일별 트렌드 그래프
            plt.figure(figsize=(10, 6))
            plt.plot(daily_trend['date'], daily_trend['roas'], marker='o')
            plt.title('일별 ROAS 추이')
            plt.xlabel('날짜')
            plt.ylabel('ROAS')
            plt.grid(True)
            plt.tight_layout()
            daily_trend_chart_path = f"{self.reports_dir}/daily_trend_chart.png"
            plt.savefig(daily_trend_chart_path)
            plt.close()
            
            # 2. 플랫폼별 성과 그래프
            plt.figure(figsize=(10, 6))
            sns.barplot(x='platform', y='roas', data=platform_summary)
            plt.title('플랫폼별 ROAS')
            plt.xlabel('플랫폼')
            plt.ylabel('ROAS')
            plt.grid(True)
            plt.tight_layout()
            platform_chart_path = f"{self.reports_dir}/platform_chart.png"
            plt.savefig(platform_chart_path)
            plt.close()
            
            # 3. 크리에이티브 유형별 성과 그래프
            plt.figure(figsize=(10, 6))
            sns.barplot(x='creative_type', y='roas', data=creative_summary)
            plt.title('크리에이티브 유형별 ROAS')
            plt.xlabel('크리에이티브 유형')
            plt.ylabel('ROAS')
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.tight_layout()
            creative_chart_path = f"{self.reports_dir}/creative_chart.png"
            plt.savefig(creative_chart_path)
            plt.close()
            
            # 템플릿 엔진 설정
            template_loader = jinja2.FileSystemLoader(searchpath=self.templates_dir)
            template_env = jinja2.Environment(loader=template_loader)
            
            # 템플릿 파일이 없으면 기본 템플릿 생성
            template_path = f"{self.templates_dir}/summary_report_template.html"
            if not os.path.exists(template_path):
                self.create_default_template()
            
            # 템플릿 로드
            template = template_env.get_template("summary_report_template.html")
            
            # 템플릿 렌더링
            report_date = datetime.now().strftime('%Y-%m-%d')
            html_content = template.render(
                report_date=report_date,
                date_period=date_period,
                total_spend=total_spend,
                total_impressions=total_impressions,
                total_clicks=total_clicks,
                total_conversions=total_conversions,
                total_revenue=total_revenue,
                avg_ctr=avg_ctr,
                avg_cvr=avg_cvr,
                avg_roas=avg_roas,
                avg_cpc=avg_cpc,
                avg_cpa=avg_cpa,
                platform_summary=platform_summary.to_dict('records'),
                creative_summary=creative_summary.to_dict('records'),
                age_summary=age_summary.to_dict('records'),
                gender_summary=gender_summary.to_dict('records'),
                campaign_summary=campaign_summary.to_dict('records'),
                daily_trend_chart=daily_trend_chart_path,
                platform_chart=platform_chart_path,
                creative_chart=creative_chart_path
            )
            
            # 타임스탬프
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            
            # 출력 형식에 따른 처리
            if output_format.lower() == 'html':
                # HTML 파일로 저장
                output_path = f"{self.reports_dir}/summary_report_{timestamp}.html"
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                logger.info(f"HTML 보고서 생성 완료: {output_path}")
                
                if email_recipients:
                    # 이메일 전송
                    self.send_email(
                        recipients=email_recipients,
                        subject=f"광고 캠페인 요약 보고서 ({date_period})",
                        body=html_content,
                        attachments=[output_path]
                    )
                
                return output_path
                
            elif output_format.lower() == 'email':
                # 이메일로만 전송
                if not email_recipients:
                    logger.error("이메일 수신자가 지정되지 않았습니다.")
                    return None
                
                self.send_email(
                    recipients=email_recipients,
                    subject=f"광고 캠페인 요약 보고서 ({date_period})",
                    body=html_content,
                    is_html=True
                )
                
                logger.info(f"보고서 이메일 전송 완료: {', '.join(email_recipients)}")
                return "email_sent"
            
            else:
                logger.error(f"지원하지 않는 출력 형식: {output_format}")
                return None
                
        except Exception as e:
            logger.error(f"보고서 생성 중 오류 발생: {e}")
            return None
    
    def create_campaign_performance_report(self, df, campaign_name=None, platform=None, output_format='pdf', email_recipients=None):
        """
        특정 캠페인의 성과 보고서를 생성합니다.
        
        Args:
            df (pandas.DataFrame): 광고 캠페인 데이터
            campaign_name (str, optional): 캠페인 이름
            platform (str, optional): 플랫폼 이름
            output_format (str): 출력 형식 ('pdf', 'html', 'email')
            email_recipients (list, optional): 보고서를 전송할 이메일 주소 목록
            
        Returns:
            str: 생성된 보고서 파일 경로 또는 이메일 전송 성공 여부
        """
        try:
            logger.info(f"캠페인 성과 보고서 생성 시작: {campaign_name}, {platform}")
            
            # 데이터 필터링
            filtered_df = df.copy()
            
            if campaign_name:
                filtered_df = filtered_df[filtered_df['campaign_name'] == campaign_name]
            
            if platform:
                filtered_df = filtered_df[filtered_df['platform'] == platform]
            
            if filtered_df.empty:
                logger.error("필터링 결과가 비어 있습니다.")
                return None
            
            # 날짜 범위 정보
            start_date = filtered_df['date'].min().strftime('%Y-%m-%d')
            end_date = filtered_df['date'].max().strftime('%Y-%m-%d')
            date_period = f"{start_date} ~ {end_date}"
            
            # 캠페인 정보
            campaign_info = {
                'name': campaign_name or "모든 캠페인",
                'platform': platform or "모든 플랫폼",
                'date_period': date_period
            }
            
            # 일별 성과 지표
            daily_performance = filtered_df.groupby('date').agg({
                'spend': 'sum',
                'impressions': 'sum',
                'clicks': 'sum',
                'conversions': 'sum',
                'revenue': 'sum'
            }).reset_index()
            
            daily_performance['ctr'] = (daily_performance['clicks'] / daily_performance['impressions']) * 100
            daily_performance['cvr'] = (daily_performance['conversions'] / daily_performance['clicks']) * 100
            daily_performance['roas'] = daily_performance['revenue'] / daily_performance['spend']
            
            # 총 성과 지표
            total_performance = {
                'spend': filtered_df['spend'].sum(),
                'impressions': filtered_df['impressions'].sum(),
                'clicks': filtered_df['clicks'].sum(),
                'conversions': filtered_df['conversions'].sum(),
                'revenue': filtered_df['revenue'].sum(),
                'ctr': (filtered_df['clicks'].sum() / filtered_df['impressions'].sum()) * 100 if filtered_df['impressions'].sum() > 0 else 0,
                'cvr': (filtered_df['conversions'].sum() / filtered_df['clicks'].sum()) * 100 if filtered_df['clicks'].sum() > 0 else 0,
                'roas': filtered_df['revenue'].sum() / filtered_df['spend'].sum() if filtered_df['spend'].sum() > 0 else 0
            }
            
            # 타겟팅별 성과
            targeting_performance = {
                'age': filtered_df.groupby('target_age').agg({
                    'spend': 'sum',
                    'conversions': 'sum',
                    'revenue': 'sum'
                }).reset_index(),
                
                'gender': filtered_df.groupby('target_gender').agg({
                    'spend': 'sum',
                    'conversions': 'sum',
                    'revenue': 'sum'
                }).reset_index()
            }
            
            targeting_performance['age']['roas'] = targeting_performance['age']['revenue'] / targeting_performance['age']['spend']
            targeting_performance['gender']['roas'] = targeting_performance['gender']['revenue'] / targeting_performance['gender']['spend']
            
            # 크리에이티브 유형별 성과
            creative_performance = filtered_df.groupby('creative_type').agg({
                'spend': 'sum',
                'impressions': 'sum',
                'clicks': 'sum',
                'conversions': 'sum',
                'revenue': 'sum'
            }).reset_index()
            
            creative_performance['ctr'] = (creative_performance['clicks'] / creative_performance['impressions']) * 100
            creative_performance['cvr'] = (creative_performance['conversions'] / creative_performance['clicks']) * 100
            creative_performance['roas'] = creative_performance['revenue'] / creative_performance['spend']
            
            # 그래프 생성
            # 1. 일별 지출 및 ROAS 트렌드
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            ax1.set_xlabel('날짜')
            ax1.set_ylabel('지출 (₩)', color='tab:blue')
            ax1.plot(daily_performance['date'], daily_performance['spend'], color='tab:blue', marker='o')
            ax1.tick_params(axis='y', labelcolor='tab:blue')
            
            ax2 = ax1.twinx()
            ax2.set_ylabel('ROAS', color='tab:red')
            ax2.plot(daily_performance['date'], daily_performance['roas'], color='tab:red', marker='s')
            ax2.tick_params(axis='y', labelcolor='tab:red')
            
            plt.title('일별 지출 및 ROAS 추이')
            plt.grid(True)
            plt.tight_layout()
            daily_trend_chart_path = f"{self.reports_dir}/campaign_daily_trend_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
            plt.savefig(daily_trend_chart_path)
            plt.close()
            
            # 2. 타겟 연령별 ROAS
            plt.figure(figsize=(10, 6))
            sns.barplot(x='target_age', y='roas', data=targeting_performance['age'])
            plt.title('타겟 연령별 ROAS')
            plt.xlabel('타겟 연령')
            plt.ylabel('ROAS')
            plt.grid(True)
            plt.tight_layout()
            age_chart_path = f"{self.reports_dir}/campaign_age_chart_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
            plt.savefig(age_chart_path)
            plt.close()
            
            # 3. 크리에이티브 유형별 성과
            plt.figure(figsize=(10, 6))
            sns.barplot(x='creative_type', y='roas', data=creative_performance)
            plt.title('크리에이티브 유형별 ROAS')
            plt.xlabel('크리에이티브 유형')
            plt.ylabel('ROAS')
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.tight_layout()
            creative_chart_path = f"{self.reports_dir}/campaign_creative_chart_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
            plt.savefig(creative_chart_path)
            plt.close()
            
            # 템플릿 엔진 설정
            template_loader = jinja2.FileSystemLoader(searchpath=self.templates_dir)
            template_env = jinja2.Environment(loader=template_loader)
            
            # 템플릿 파일이 없으면 기본 템플릿 생성
            template_path = f"{self.templates_dir}/campaign_report_template.html"
            if not os.path.exists(template_path):
                self.create_default_campaign_template()
            
            # 템플릿 로드
            template = template_env.get_template("campaign_report_template.html")
            
            # 템플릿 렌더링
            report_date = datetime.now().strftime('%Y-%m-%d')
            html_content = template.render(
                report_date=report_date,
                campaign_info=campaign_info,
                total_performance=total_performance,
                daily_performance=daily_performance.to_dict('records'),
                targeting_age=targeting_performance['age'].to_dict('records'),
                targeting_gender=targeting_performance['gender'].to_dict('records'),
                creative_performance=creative_performance.to_dict('records'),
                daily_trend_chart=daily_trend_chart_path,
                age_chart=age_chart_path,
                creative_chart=creative_chart_path
            )
            
            # 타임스탬프
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            
            # 출력 형식에 따른 처리
            campaign_name_safe = campaign_name.replace(' ', '_') if campaign_name else 'all_campaigns'
            platform_safe = platform.replace(' ', '_') if platform else 'all_platforms'
            
            if output_format.lower() == 'html':
                # HTML 파일로 저장
                output_path = f"{self.reports_dir}/campaign_report_{campaign_name_safe}_{platform_safe}_{timestamp}.html"
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                logger.info(f"HTML 캠페인 보고서 생성 완료: {output_path}")
                
                if email_recipients:
                    # 이메일 전송
                    self.send_email(
                        recipients=email_recipients,
                        subject=f"광고 캠페인 성과 보고서: {campaign_info['name']} ({date_period})",
                        body=html_content,
                        attachments=[output_path]
                    )
                
                return output_path
                
            elif output_format.lower() == 'email':
                # 이메일로만 전송
                if not email_recipients:
                    logger.error("이메일 수신자가 지정되지 않았습니다.")
                    return None
                
                self.send_email(
                    recipients=email_recipients,
                    subject=f"광고 캠페인 성과 보고서: {campaign_info['name']} ({date_period})",
                    body=html_content,
                    is_html=True
                )
                
                logger.info(f"캠페인 보고서 이메일 전송 완료: {', '.join(email_recipients)}")
                return "email_sent"
            
            else:
                logger.error(f"지원하지 않는 출력 형식: {output_format}")
                return None
                
        except Exception as e:
            logger.error(f"캠페인 보고서 생성 중 오류 발생: {e}")
            return None
    
    def create_default_template(self):
        """
        기본 요약 보고서 템플릿을 생성합니다.
        """
        template_path = f"{self.templates_dir}/summary_report_template.html"
        
        template_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>광고 캠페인 요약 보고서</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 1000px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 2px solid #ddd;
            padding-bottom: 10px;
        }
        .section {
            margin-bottom: 30px;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }
        th, td {
            padding: 10px;
            border: 1px solid #ddd;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        .summary-cards {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            margin-bottom: 20px;
        }
        .card {
            background-color: #f8f9fa;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 15px;
            width: 30%;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .card-title {
            font-size: 14px;
            color: #6c757d;
            margin-bottom: 5px;
        }
        .card-value {
            font-size: 22px;
            font-weight: bold;
            color: #2c3e50;
        }
        .chart {
            margin: 20px 0;
            text-align: center;
        }
        .chart img {
            max-width: 100%;
            height: auto;
        }
        .footer {
            text-align: center;
            margin-top: 50px;
            font-size: 12px;
            color: #6c757d;
            border-top: 1px solid #ddd;
            padding-top: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>광고 캠페인 요약 보고서</h1>
            <p>기간: {{ date_period }}</p>
            <p>생성일: {{ report_date }}</p>
        </div>
        
        <div class="section">
            <h2>전체 성과 요약</h2>
            <div class="summary-cards">
                <div class="card">
                    <div class="card-title">총 지출</div>
                    <div class="card-value">₩{{ '{:,.0f}'.format(total_spend) }}</div>
                </div>
                <div class="card">
                    <div class="card-title">총 노출 수</div>
                    <div class="card-value">{{ '{:,.0f}'.format(total_impressions) }}</div>
                </div>
                <div class="card">
                    <div class="card-title">총 클릭 수</div>
                    <div class="card-value">{{ '{:,.0f}'.format(total_performance.clicks) }}</div>
                </div>
                <div class="card">
                    <div class="card-title">총 전환 수</div>
                    <div class="card-value">{{ '{:,.0f}'.format(total_performance.conversions) }}</div>
                </div>
                <div class="card">
                    <div class="card-title">총 수익</div>
                    <div class="card-value">₩{{ '{:,.0f}'.format(total_performance.revenue) }}</div>
                </div>
                <div class="card">
                    <div class="card-title">ROAS</div>
                    <div class="card-value">{{ '{:.2f}'.format(total_performance.roas) }}</div>
                </div>
            </div>
            
            <h3>주요 지표</h3>
            <table>
                <tr>
                    <th>지표</th>
                    <th>값</th>
                </tr>
                <tr>
                    <td>클릭률 (CTR)</td>
                    <td>{{ '{:.2f}%'.format(total_performance.ctr) }}</td>
                </tr>
                <tr>
                    <td>전환율 (CVR)</td>
                    <td>{{ '{:.2f}%'.format(total_performance.cvr) }}</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>일별 지출 및 ROAS 추이</h2>
            <div class="chart">
                <img src="{{ daily_trend_chart }}" alt="일별 지출 및 ROAS 추이">
            </div>
            
            <h3>일별 상세 성과</h3>
            <table>
                <tr>
                    <th>날짜</th>
                    <th>지출</th>
                    <th>노출 수</th>
                    <th>클릭 수</th>
                    <th>전환 수</th>
                    <th>ROAS</th>
                </tr>
                {% for row in daily_performance %}
                <tr>
                    <td>{{ row.date.strftime('%Y-%m-%d') }}</td>
                    <td>₩{{ '{:,.0f}'.format(row.spend) }}</td>
                    <td>{{ '{:,.0f}'.format(row.impressions) }}</td>
                    <td>{{ '{:,.0f}'.format(row.clicks) }}</td>
                    <td>{{ '{:,.0f}'.format(row.conversions) }}</td>
                    <td>{{ '{:.2f}'.format(row.roas) }}</td>
                </tr>
                {% endfor %}
            </table>
        </div>
        
        <div class="section">
            <h2>타겟팅별 성과</h2>
            
            <h3>연령별 ROAS</h3>
            <div class="chart">
                <img src="{{ age_chart }}" alt="타겟 연령별 ROAS">
            </div>
            
            <table>
                <tr>
                    <th>연령대</th>
                    <th>지출</th>
                    <th>전환 수</th>
                    <th>수익</th>
                    <th>ROAS</th>
                </tr>
                {% for row in targeting_age %}
                <tr>
                    <td>{{ row.target_age }}</td>
                    <td>₩{{ '{:,.0f}'.format(row.spend) }}</td>
                    <td>{{ '{:,.0f}'.format(row.conversions) }}</td>
                    <td>₩{{ '{:,.0f}'.format(row.revenue) }}</td>
                    <td>{{ '{:.2f}'.format(row.roas) }}</td>
                </tr>
                {% endfor %}
            </table>
            
            <h3>성별별 성과</h3>
            <table>
                <tr>
                    <th>성별</th>
                    <th>지출</th>
                    <th>전환 수</th>
                    <th>수익</th>
                    <th>ROAS</th>
                </tr>
                {% for row in targeting_gender %}
                <tr>
                    <td>{{ row.target_gender }}</td>
                    <td>₩{{ '{:,.0f}'.format(row.spend) }}</td>
                    <td>{{ '{:,.0f}'.format(row.conversions) }}</td>
                    <td>₩{{ '{:,.0f}'.format(row.revenue) }}</td>
                    <td>{{ '{:.2f}'.format(row.roas) }}</td>
                </tr>
                {% endfor %}
            </table>
        </div>
        
        <div class="section">
            <h2>크리에이티브 유형별 성과</h2>
            <div class="chart">
                <img src="{{ creative_chart }}" alt="크리에이티브 유형별 ROAS">
            </div>
            
            <table>
                <tr>
                    <th>크리에이티브 유형</th>
                    <th>지출</th>
                    <th>노출 수</th>
                    <th>클릭 수</th>
                    <th>전환 수</th>
                    <th>ROAS</th>
                </tr>
                {% for row in creative_performance %}
                <tr>
                    <td>{{ row.creative_type }}</td>
                    <td>₩{{ '{:,.0f}'.format(row.spend) }}</td>
                    <td>{{ '{:,.0f}'.format(row.impressions) }}</td>
                    <td>{{ '{:,.0f}'.format(row.clicks) }}</td>
                    <td>{{ '{:,.0f}'.format(row.conversions) }}</td>
                    <td>{{ '{:.2f}'.format(row.roas) }}</td>
                </tr>
                {% endfor %}
            </table>
        </div>
        
        <div class="footer">
            <p>광고 캠페인 분석 대시보드 | 생성일: {{ report_date }}</p>
        </div>
    </div>
</body>
</html>
"""
        
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(template_content)
        
        logger.info(f"기본 캠페인 보고서 템플릿 생성됨: {template_path}")
    
    def send_email(self, recipients, subject, body, attachments=None, is_html=False):
        """
        이메일을 전송합니다.
        
        Args:
            recipients (list): 수신자 이메일 주소 목록
            subject (str): 이메일 제목
            body (str): 이메일 본문
            attachments (list, optional): 첨부 파일 경로 목록
            is_html (bool): HTML 형식 여부
            
        Returns:
            bool: 이메일 전송 성공 여부
        """
        if not self.email_config:
            logger.error("이메일 설정이 없습니다.")
            return False
        
        try:
            # 이메일 설정
            smtp_server = self.email_config.get('smtp_server', 'smtp.gmail.com')
            smtp_port = self.email_config.get('smtp_port', 587)
            smtp_username = self.email_config.get('username')
            smtp_password = self.email_config.get('password')
            sender_email = self.email_config.get('sender_email', smtp_username)
            
            if not all([smtp_server, smtp_port, smtp_username, smtp_password, sender_email]):
                logger.error("필수 이메일 설정이 누락되었습니다.")
                return False
            
            # 이메일 메시지 생성
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = subject
            
            # 본문 추가
            if is_html:
                msg.attach(MIMEText(body, 'html'))
            else:
                msg.attach(MIMEText(body, 'plain'))
            
            # 첨부 파일 추가
            if attachments:
                for file_path in attachments:
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as file:
                            part = MIMEApplication(file.read(), Name=os.path.basename(file_path))
                            part['Content-Disposition'] = f'attachment; filename="{os.path.basename(file_path)}"'
                            msg.attach(part)
                    else:
                        logger.warning(f"첨부 파일을 찾을 수 없습니다: {file_path}")
            
            # SMTP 서버 연결 및 이메일 전송
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(smtp_username, smtp_password)
                server.send_message(msg)
            
            logger.info(f"이메일 전송 완료: {subject} ({len(recipients)} 명의 수신자)")
            return True
            
        except Exception as e:
            logger.error(f"이메일 전송 중 오류 발생: {e}")
            return False
    
    def schedule_reports(self, data_loader, schedule, report_type='summary', output_format='pdf', email_recipients=None):
        """
        보고서 생성을 스케줄링합니다.
        
        Args:
            data_loader (DataLoader): 데이터 로더 인스턴스
            schedule (str): 스케줄 ('daily', 'weekly', 'monthly')
            report_type (str): 보고서 유형 ('summary' 또는 'campaign')
            output_format (str): 출력 형식 ('pdf', 'html', 'email')
            email_recipients (list, optional): 이메일 수신자 목록
            
        Returns:
            bool: 스케줄링 성공 여부
        """
        # 이 함수는 스케줄러와 연동하여 사용해야 합니다.
        # 실제 구현은 외부 스케줄러(예: APScheduler, Celery, cron 등)에 따라 달라집니다.
        logger.info(f"{schedule} {report_type} 보고서 스케줄링 요청됨")
        return True

    # HTML 보고서 생성
    def create_html_report(self, df, output_path=None, include_sections=None):
        """
        HTML 형식의 보고서를 생성합니다.
        
        Args:
            df (pandas.DataFrame): 광고 캠페인 데이터
            output_path (str, optional): 출력 파일 경로
            include_sections (list, optional): 포함할 섹션 목록
            
        Returns:
            str: 생성된 보고서 파일 경로
        """
        try:
            logger.info("HTML 보고서 생성 시작")
            
            # 기본 포함 섹션 설정
            if include_sections is None:
                include_sections = ["주요 지표", "시계열 트렌드", "플랫폼별 성과"]
            
            # 날짜 범위 정보
            start_date = df['date'].min().strftime('%Y-%m-%d')
            end_date = df['date'].max().strftime('%Y-%m-%d')
            date_period = f"{start_date} ~ {end_date}"
            
            # 주요 지표 계산
            summary_metrics = {
                "총 지출": f"₩{df['spend'].sum():,.0f}",
                "총 수익": f"₩{df['revenue'].sum():,.0f}",
                "평균 ROAS": f"{df['revenue'].sum() / df['spend'].sum() if df['spend'].sum() > 0 else 0:.2f}",
                "총 노출": f"{df['impressions'].sum():,.0f}",
                "총 클릭": f"{df['clicks'].sum():,.0f}",
                "총 전환": f"{df['conversions'].sum():,.0f}",
                "클릭률 (CTR)": f"{df['clicks'].sum() / df['impressions'].sum() * 100 if df['impressions'].sum() > 0 else 0:.2f}%",
                "전환율 (CVR)": f"{df['conversions'].sum() / df['clicks'].sum() * 100 if df['clicks'].sum() > 0 else 0:.2f}%"
            }
            
            # 일별 데이터 집계
            daily_data = df.groupby('date').agg({
                'spend': 'sum',
                'impressions': 'sum',
                'clicks': 'sum',
                'conversions': 'sum',
                'revenue': 'sum'
            }).reset_index()
            
            # 파생 지표 계산
            daily_data['ctr'] = daily_data['clicks'] / daily_data['impressions'] * 100
            daily_data['cvr'] = daily_data['conversions'] / daily_data['clicks'] * 100
            daily_data['roas'] = daily_data['revenue'] / daily_data['spend']
            
            # 플랫폼별 데이터
            platform_data = df.groupby('platform').agg({
                'spend': 'sum',
                'impressions': 'sum',
                'clicks': 'sum',
                'conversions': 'sum',
                'revenue': 'sum'
            }).reset_index()
            
            platform_data['ctr'] = platform_data['clicks'] / platform_data['impressions'] * 100
            platform_data['cvr'] = platform_data['conversions'] / platform_data['clicks'] * 100
            platform_data['roas'] = platform_data['revenue'] / platform_data['spend']
            
            # 크리에이티브별 데이터
            creative_data = df.groupby('creative_type').agg({
                'spend': 'sum',
                'impressions': 'sum',
                'clicks': 'sum',
                'conversions': 'sum',
                'revenue': 'sum'
            }).reset_index()
            
            creative_data['ctr'] = creative_data['clicks'] / creative_data['impressions'] * 100
            creative_data['cvr'] = creative_data['conversions'] / creative_data['clicks'] * 100
            creative_data['roas'] = creative_data['revenue'] / creative_data['spend']
            
            # 타겟팅별 데이터
            age_data = df.groupby('target_age').agg({
                'spend': 'sum',
                'conversions': 'sum',
                'revenue': 'sum'
            }).reset_index()
            
            age_data['roas'] = age_data['revenue'] / age_data['spend']
            
            gender_data = df.groupby('target_gender').agg({
                'spend': 'sum',
                'conversions': 'sum',
                'revenue': 'sum'
            }).reset_index()
            
            gender_data['roas'] = gender_data['revenue'] / gender_data['spend']
            
            # 차트 생성
            import matplotlib.pyplot as plt
            import seaborn as sns
            import base64
            from io import BytesIO
            
            # 이미지를 base64로 인코딩하는 함수
            def get_image_base64(fig):
                buf = BytesIO()
                fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                buf.close()
                plt.close(fig)
                return img_str
            
            # 시계열 차트
            time_series_chart = None
            if "시계열 트렌드" in include_sections:
                fig, ax1 = plt.subplots(figsize=(10, 6))
                
                color = 'tab:blue'
                ax1.set_xlabel('날짜')
                ax1.set_ylabel('지출', color=color)
                ax1.plot(daily_data['date'], daily_data['spend'], color=color, marker='o')
                ax1.tick_params(axis='y', labelcolor=color)
                
                ax2 = ax1.twinx()
                color = 'tab:red'
                ax2.set_ylabel('ROAS', color=color)
                ax2.plot(daily_data['date'], daily_data['roas'], color=color, marker='s')
                ax2.tick_params(axis='y', labelcolor=color)
                
                fig.tight_layout()
                plt.title('일별 지출 및 ROAS 추이')
                
                time_series_chart = get_image_base64(fig)
            
            # 플랫폼별 차트
            platform_chart = None
            if "플랫폼별 성과" in include_sections:
                plt.figure(figsize=(10, 6))
                chart = sns.barplot(x='platform', y='roas', data=platform_data)
                plt.title('플랫폼별 ROAS')
                plt.xlabel('플랫폼')
                plt.ylabel('ROAS')
                plt.tight_layout()
                
                platform_chart = get_image_base64(plt.gcf())
            
            # 크리에이티브별 차트
            creative_chart = None
            if "크리에이티브별 성과" in include_sections:
                plt.figure(figsize=(10, 6))
                chart = sns.barplot(x='creative_type', y='roas', data=creative_data)
                plt.title('크리에이티브 유형별 ROAS')
                plt.xlabel('크리에이티브 유형')
                plt.ylabel('ROAS')
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                creative_chart = get_image_base64(plt.gcf())
            
            # 템플릿 엔진 설정
            import jinja2
            template_loader = jinja2.FileSystemLoader(searchpath=self.templates_dir)
            template_env = jinja2.Environment(loader=template_loader)
            
            # 템플릿 파일이 없으면 기본 템플릿 생성
            template_path = f"{self.templates_dir}/html_report_template.html"
            if not os.path.exists(template_path):
                self._create_html_template()
            
            # 템플릿 로드
            template = template_env.get_template("html_report_template.html")
            
            # 템플릿 렌더링
            report_date = datetime.now().strftime('%Y-%m-%d')
            html_content = template.render(
                report_date=report_date,
                date_period=date_period,
                summary_metrics=summary_metrics,
                daily_data=daily_data.to_dict('records'),
                platform_data=platform_data.to_dict('records'),
                creative_data=creative_data.to_dict('records'),
                age_data=age_data.to_dict('records'),
                gender_data=gender_data.to_dict('records'),
                time_series_chart=time_series_chart,
                platform_chart=platform_chart,
                creative_chart=creative_chart,
                include_sections=include_sections
            )
            
            # 타임스탬프
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            
            # 출력 경로 설정
            if output_path is None:
                output_path = f"{self.reports_dir}/campaign_report_{timestamp}.html"
            
            # HTML 파일로 저장
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"HTML 보고서 생성 완료: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"HTML 보고서 생성 중 오류 발생: {e}")
            return None

    def _create_html_template(self):
        """
        기본 HTML 보고서 템플릿을 생성합니다.
        """
        template_path = f"{self.templates_dir}/html_report_template.html"
        
        template_content = """<!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>광고 캠페인 분석 보고서</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                margin: 0;
                padding: 20px;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
                border-bottom: 2px solid #ddd;
                padding-bottom: 10px;
            }
            .section {
                margin-bottom: 30px;
            }
            h1, h2, h3 {
                color: #2c3e50;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }
            th, td {
                padding: 10px;
                border: 1px solid #ddd;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
            .summary-cards {
                display: flex;
                flex-wrap: wrap;
                justify-content: space-between;
                margin-bottom: 20px;
            }
            .card {
                background-color: #f8f9fa;
                border-radius: 5px;
                padding: 15px;
                margin-bottom: 15px;
                width: 22%;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .card-title {
                font-size: 14px;
                color: #6c757d;
                margin-bottom: 5px;
            }
            .card-value {
                font-size: 22px;
                font-weight: bold;
                color: #2c3e50;
            }
            .chart {
                margin: 20px 0;
                text-align: center;
            }
            .chart img {
                max-width: 100%;
                height: auto;
            }
            .footer {
                text-align: center;
                margin-top: 50px;
                font-size: 12px;
                color: #6c757d;
                border-top: 1px solid #ddd;
                padding-top: 10px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>광고 캠페인 분석 보고서</h1>
                <p>기간: {{ date_period }}</p>
                <p>생성일: {{ report_date }}</p>
            </div>
            
            {% if "주요 지표" in include_sections %}
            <div class="section">
                <h2>주요 지표</h2>
                <div class="summary-cards">
                    {% for title, value in summary_metrics.items() %}
                    <div class="card">
                        <div class="card-title">{{ title }}</div>
                        <div class="card-value">{{ value }}</div>
                    </div>
                    {% endfor %}
                </div>
            </div>
            {% endif %}
            
            {% if "시계열 트렌드" in include_sections %}
            <div class="section">
                <h2>시계열 트렌드</h2>
                {% if time_series_chart %}
                <div class="chart">
                    <img src="data:image/png;base64,{{ time_series_chart }}" alt="시계열 트렌드 차트">
                </div>
                {% endif %}
                
                <h3>일별 상세 데이터</h3>
                <table>
                    <tr>
                        <th>날짜</th>
                        <th>지출</th>
                        <th>노출</th>
                        <th>클릭</th>
                        <th>CTR (%)</th>
                        <th>전환</th>
                        <th>CVR (%)</th>
                        <th>수익</th>
                        <th>ROAS</th>
                    </tr>
                    {% for row in daily_data %}
                    <tr>
                        <td>{{ row.date.strftime('%Y-%m-%d') }}</td>
                        <td>₩{{ "{:,.0f}".format(row.spend) }}</td>
                        <td>{{ "{:,.0f}".format(row.impressions) }}</td>
                        <td>{{ "{:,.0f}".format(row.clicks) }}</td>
                        <td>{{ "{:.2f}".format(row.ctr) }}</td>
                        <td>{{ "{:,.0f}".format(row.conversions) }}</td>
                        <td>{{ "{:.2f}".format(row.cvr) }}</td>
                        <td>₩{{ "{:,.0f}".format(row.revenue) }}</td>
                        <td>{{ "{:.2f}".format(row.roas) }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
            {% endif %}
            
            {% if "플랫폼별 성과" in include_sections %}
            <div class="section">
                <h2>플랫폼별 성과</h2>
                {% if platform_chart %}
                <div class="chart">
                    <img src="data:image/png;base64,{{ platform_chart }}" alt="플랫폼별 성과 차트">
                </div>
                {% endif %}
                
                <h3>플랫폼별 상세 데이터</h3>
                <table>
                    <tr>
                        <th>플랫폼</th>
                        <th>지출</th>
                        <th>노출</th>
                        <th>클릭</th>
                        <th>CTR (%)</th>
                        <th>전환</th>
                        <th>CVR (%)</th>
                        <th>수익</th>
                        <th>ROAS</th>
                    </tr>
                    {% for row in platform_data %}
                    <tr>
                        <td>{{ row.platform }}</td>
                        <td>₩{{ "{:,.0f}".format(row.spend) }}</td>
                        <td>{{ "{:,.0f}".format(row.impressions) }}</td>
                        <td>{{ "{:,.0f}".format(row.clicks) }}</td>
                        <td>{{ "{:.2f}".format(row.ctr) }}</td>
                        <td>{{ "{:,.0f}".format(row.conversions) }}</td>
                        <td>{{ "{:.2f}".format(row.cvr) }}</td>
                        <td>₩{{ "{:,.0f}".format(row.revenue) }}</td>
                        <td>{{ "{:.2f}".format(row.roas) }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
            {% endif %}
            
            {% if "크리에이티브별 성과" in include_sections %}
            <div class="section">
                <h2>크리에이티브별 성과</h2>
                {% if creative_chart %}
                <div class="chart">
                    <img src="data:image/png;base64,{{ creative_chart }}" alt="크리에이티브별 성과 차트">
                </div>
                {% endif %}
                
                <h3>크리에이티브별 상세 데이터</h3>
                <table>
                    <tr>
                        <th>크리에이티브 유형</th>
                        <th>지출</th>
                        <th>노출</th>
                        <th>클릭</th>
                        <th>CTR (%)</th>
                        <th>전환</th>
                        <th>CVR (%)</th>
                        <th>수익</th>
                        <th>ROAS</th>
                    </tr>
                    {% for row in creative_data %}
                    <tr>
                        <td>{{ row.creative_type }}</td>
                        <td>₩{{ "{:,.0f}".format(row.spend) }}</td>
                        <td>{{ "{:,.0f}".format(row.impressions) }}</td>
                        <td>{{ "{:,.0f}".format(row.clicks) }}</td>
                        <td>{{ "{:.2f}".format(row.ctr) }}</td>
                        <td>{{ "{:,.0f}".format(row.conversions) }}</td>
                        <td>{{ "{:.2f}".format(row.cvr) }}</td>
                        <td>₩{{ "{:,.0f}".format(row.revenue) }}</td>
                        <td>{{ "{:.2f}".format(row.roas) }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
            {% endif %}
            
            {% if "타겟팅별 성과" in include_sections %}
            <div class="section">
                <h2>타겟팅별 성과</h2>
                
                <h3>연령별 성과</h3>
                <table>
                    <tr>
                        <th>연령대</th>
                        <th>지출</th>
                        <th>전환</th>
                        <th>수익</th>
                        <th>ROAS</th>
                    </tr>
                    {% for row in age_data %}
                    <tr>
                        <td>{{ row.target_age }}</td>
                        <td>₩{{ "{:,.0f}".format(row.spend) }}</td>
                        <td>{{ "{:,.0f}".format(row.conversions) }}</td>
                        <td>₩{{ "{:,.0f}".format(row.revenue) }}</td>
                        <td>{{ "{:.2f}".format(row.roas) }}</td>
                    </tr>
                    {% endfor %}
                </table>
                
                <h3>성별 성과</h3>
                <table>
                    <tr>
                        <th>성별</th>
                        <th>지출</th>
                        <th>전환</th>
                        <th>수익</th>
                        <th>ROAS</th>
                    </tr>
                    {% for row in gender_data %}
                    <tr>
                        <td>{{ row.target_gender }}</td>
                        <td>₩{{ "{:,.0f}".format(row.spend) }}</td>
                        <td>{{ "{:,.0f}".format(row.conversions) }}</td>
                        <td>₩{{ "{:,.0f}".format(row.revenue) }}</td>
                        <td>{{ "{:.2f}".format(row.roas) }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
            {% endif %}
            
            <div class="footer">
                <p>광고 캠페인 분석 대시보드 | 생성일: {{ report_date }}</p>
            </div>
        </div>
    </body>
    </html>
    """
        
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(template_content)
        
        logger.info(f"기본 HTML 보고서 템플릿 생성됨: {template_path}")


# 모듈 테스트 코드
if __name__ == "__main__":
    import pandas as pd
    from data_loader import DataLoader
    
    # 로그 디렉토리 생성
    if not os.path.exists("logs"):
        os.makedirs("logs")
    
    # 데이터 로드
    data_loader = DataLoader()
    data = data_loader.load_data()
    
    # 보고서 생성기 초기화
    generator = ReportGenerator()
    
    # 요약 보고서 생성
    report_path = generator.create_summary_report(data, output_format='html')
    print(f"요약 보고서 생성됨: {report_path}")
    
    # 특정 캠페인 보고서 생성
    campaign_name = data['campaign_name'].unique()[0]
    platform = data['platform'].unique()[0]
    campaign_report_path = generator.create_campaign_performance_report(
        data, 
        campaign_name=campaign_name, 
        platform=platform, 
        output_format='html'
    )
    print(f"캠페인 보고서 생성됨: {campaign_report_path}")
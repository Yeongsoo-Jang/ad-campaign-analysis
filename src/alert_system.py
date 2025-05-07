import pandas as pd
import numpy as np
import os
import logging
import yaml
import smtplib
import json
import requests
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/alert_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("alert_system")

class AlertSystem:
    """
    광고 캠페인 지표에 대한 알림을 생성하고 전송하는 클래스
    """
    
    def __init__(self, config_path="config/config.yaml"):
        """
        설정 파일을 로드하고 초기화합니다.
        
        Args:
            config_path (str): 설정 파일 경로
        """
        self.config = self._load_config(config_path)
        self.alert_config = self.config.get('alerts', {})
        self.email_config = self.config.get('email', {})
        self.slack_config = self.config.get('slack', {})
        self.thresholds = self.alert_config.get('thresholds', {})
        self.alert_history = []
        self.alerts_dir = "alerts"
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
        for dir_path in [self.alerts_dir, "logs"]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                logger.info(f"디렉토리 생성됨: {dir_path}")
    
    def check_alerts(self, df, period='daily'):
        """
        데이터에서 알림 조건을 확인합니다.
        
        Args:
            df (pandas.DataFrame): 광고 캠페인 데이터
            period (str): 알림 확인 주기 ('daily', 'weekly', 'monthly')
            
        Returns:
            list: 발생한 알림 목록
        """
        try:
            logger.info(f"{period} 알림 조건 확인 시작")
            alerts = []
            
            # 가장 최근 날짜의 데이터만 사용
            if period == 'daily':
                latest_date = df['date'].max()
                current_df = df[df['date'] == latest_date]
                
                # 하루 전 데이터와 비교
                previous_date = latest_date - timedelta(days=1)
                previous_df = df[df['date'] == previous_date]
                
            elif period == 'weekly':
                # 최근 7일 데이터
                latest_date = df['date'].max()
                date_7days_ago = latest_date - timedelta(days=7)
                current_df = df[(df['date'] > date_7days_ago) & (df['date'] <= latest_date)]
                
                # 이전 7일 데이터
                date_14days_ago = date_7days_ago - timedelta(days=7)
                previous_df = df[(df['date'] > date_14days_ago) & (df['date'] <= date_7days_ago)]
                
            elif period == 'monthly':
                # 최근 30일 데이터
                latest_date = df['date'].max()
                date_30days_ago = latest_date - timedelta(days=30)
                current_df = df[(df['date'] > date_30days_ago) & (df['date'] <= latest_date)]
                
                # 이전 30일 데이터
                date_60days_ago = date_30days_ago - timedelta(days=30)
                previous_df = df[(df['date'] > date_60days_ago) & (df['date'] <= date_30days_ago)]
            
            else:
                logger.error(f"지원하지 않는 주기: {period}")
                return []
            
            # 현재 및 이전 기간 데이터 집계
            if not current_df.empty:
                current_metrics = self._aggregate_metrics(current_df)
            else:
                logger.warning(f"현재 기간 데이터가 없습니다: {period}")
                return []
            
            if not previous_df.empty:
                previous_metrics = self._aggregate_metrics(previous_df)
                
                # 변화율 계산
                changes = self._calculate_changes(current_metrics, previous_metrics)
                
                # 알림 조건 확인
                for metric, value in changes.items():
                    threshold_key = f"{metric}_change_threshold"
                    if threshold_key in self.thresholds:
                        threshold = self.thresholds[threshold_key]
                        
                        # 절대값 비교 (양수/음수 구분)
                        if value <= -threshold:
                            alert = {
                                'type': 'metric_decrease',
                                'metric': metric,
                                'value': value,
                                'threshold': -threshold,
                                'current': current_metrics.get(metric),
                                'previous': previous_metrics.get(metric),
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'period': period
                            }
                            alerts.append(alert)
                            logger.info(f"알림 발생: {metric} {value:.2%} 감소 (임계값: {-threshold:.2%})")
                            
                        elif value >= threshold:
                            alert = {
                                'type': 'metric_increase',
                                'metric': metric,
                                'value': value,
                                'threshold': threshold,
                                'current': current_metrics.get(metric),
                                'previous': previous_metrics.get(metric),
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'period': period
                            }
                            alerts.append(alert)
                            logger.info(f"알림 발생: {metric} {value:.2%} 증가 (임계값: {threshold:.2%})")
            
            # 절대 임계값 확인
            for metric, value in current_metrics.items():
                min_threshold_key = f"{metric}_min_threshold"
                max_threshold_key = f"{metric}_max_threshold"
                
                if min_threshold_key in self.thresholds and value < self.thresholds[min_threshold_key]:
                    alert = {
                        'type': 'metric_below_min',
                        'metric': metric,
                        'value': value,
                        'threshold': self.thresholds[min_threshold_key],
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'period': period
                    }
                    alerts.append(alert)
                    logger.info(f"알림 발생: {metric} {value:.4f}가 최소 임계값 {self.thresholds[min_threshold_key]:.4f} 미만")
                
                if max_threshold_key in self.thresholds and value > self.thresholds[max_threshold_key]:
                    alert = {
                        'type': 'metric_above_max',
                        'metric': metric,
                        'value': value,
                        'threshold': self.thresholds[max_threshold_key],
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'period': period
                    }
                    alerts.append(alert)
                    logger.info(f"알림 발생: {metric} {value:.4f}가 최대 임계값 {self.thresholds[max_threshold_key]:.4f} 초과")
            
            # 이상치 감지 (간단한 Z-점수 기반)
            if len(df) > 10:  # 충분한 데이터가 있을 때만 수행
                for metric in current_metrics.keys():
                    if metric in df.columns and df[metric].dtype in [np.int64, np.float64]:
                        z_score_threshold = self.thresholds.get(f"{metric}_z_score_threshold", 3.0)
                        metric_mean = df[metric].mean()
                        metric_std = df[metric].std()
                        
                        if metric_std > 0:
                            current_value = current_metrics[metric]
                            z_score = abs((current_value - metric_mean) / metric_std)
                            
                            if z_score > z_score_threshold:
                                alert = {
                                    'type': 'metric_anomaly',
                                    'metric': metric,
                                    'value': current_value,
                                    'z_score': z_score,
                                    'threshold': z_score_threshold,
                                    'mean': metric_mean,
                                    'std': metric_std,
                                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    'period': period
                                }
                                alerts.append(alert)
                                logger.info(f"알림 발생: {metric} 이상치 감지 (Z-점수: {z_score:.2f})")
            
            # 알림 기록 저장
            self.alert_history.extend(alerts)
            self._save_alert_history()
            
            logger.info(f"{period} 알림 조건 확인 완료: {len(alerts)}개 알림 발생")
            return alerts
            
        except Exception as e:
            logger.error(f"알림 조건 확인 중 오류 발생: {e}")
            return []
    
    def _aggregate_metrics(self, df):
        """
        데이터프레임에서 지표를 집계합니다.
        
        Args:
            df (pandas.DataFrame): 집계할 데이터
            
        Returns:
            dict: 집계된 지표
        """
        metrics = {}
        
        # 수치형 컬럼 합계
        num_columns = ['spend', 'impressions', 'clicks', 'conversions', 'revenue']
        for col in num_columns:
            if col in df.columns:
                metrics[col] = df[col].sum()
        
        # 비율 지표 계산
        if 'clicks' in metrics and 'impressions' in metrics and metrics['impressions'] > 0:
            metrics['ctr'] = metrics['clicks'] / metrics['impressions']
            
        if 'conversions' in metrics and 'clicks' in metrics and metrics['clicks'] > 0:
            metrics['cvr'] = metrics['conversions'] / metrics['clicks']
            
        if 'revenue' in metrics and 'spend' in metrics and metrics['spend'] > 0:
            metrics['roas'] = metrics['revenue'] / metrics['spend']
            
        if 'clicks' in metrics and 'spend' in metrics and metrics['clicks'] > 0:
            metrics['cpc'] = metrics['spend'] / metrics['clicks']
            
        if 'conversions' in metrics and 'spend' in metrics and metrics['conversions'] > 0:
            metrics['cpa'] = metrics['spend'] / metrics['conversions']
        
        return metrics
    
    def _calculate_changes(self, current, previous):
        """
        두 기간의 지표 간 변화율을 계산합니다.
        
        Args:
            current (dict): 현재 기간 지표
            previous (dict): 이전 기간 지표
            
        Returns:
            dict: 변화율
        """
        changes = {}
        
        for metric, current_value in current.items():
            if metric in previous and previous[metric] != 0:
                changes[metric] = (current_value - previous[metric]) / previous[metric]
        
        return changes
    
    def _save_alert_history(self):
        """
        알림 기록을 파일로 저장합니다.
        """
        try:
            history_path = f"{self.alerts_dir}/alert_history.json"
            
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(self.alert_history, f, ensure_ascii=False, indent=2)
                
            logger.info(f"알림 기록 저장 완료: {history_path}")
            
        except Exception as e:
            logger.error(f"알림 기록 저장 중 오류 발생: {e}")
    
    def send_alerts(self, alerts, channels=None):
        """
        발생한 알림을 지정된 채널로 전송합니다.
        
        Args:
            alerts (list): 발생한 알림 목록
            channels (list, optional): 알림을 전송할 채널 목록 ('email', 'slack')
            
        Returns:
            dict: 채널별 전송 결과
        """
        if not alerts:
            logger.info("전송할 알림이 없습니다.")
            return {}
        
        if channels is None:
            channels = self.alert_config.get('channels', ['email'])
        
        results = {}
        
        for channel in channels:
            if channel == 'email':
                result = self._send_email_alerts(alerts)
                results['email'] = result
                
            elif channel == 'slack':
                result = self._send_slack_alerts(alerts)
                results['slack'] = result
                
            else:
                logger.warning(f"지원하지 않는 알림 채널: {channel}")
        
        return results
    
    def _send_email_alerts(self, alerts):
        """
        이메일로 알림을 전송합니다.
        
        Args:
            alerts (list): 발생한 알림 목록
            
        Returns:
            bool: 전송 성공 여부
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
            recipients = self.alert_config.get('email_recipients', [])
            
            if not all([smtp_server, smtp_port, smtp_username, smtp_password, sender_email, recipients]):
                logger.error("필수 이메일 설정이 누락되었습니다.")
                return False
            
            # 알림 내용 생성
            alert_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            subject = f"[광고 캠페인 알림] {len(alerts)}개의 알림이 발생했습니다 - {alert_date}"
            
            # HTML 형식 이메일 본문
            body = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                    .container {{ max-width: 800px; margin: 0 auto; padding: 20px; }}
                    h1 {{ color: #2c3e50; border-bottom: 2px solid #ddd; padding-bottom: 10px; }}
                    h2 {{ color: #3498db; }}
                    table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
                    th, td {{ padding: 10px; border: 1px solid #ddd; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .alert-high {{ background-color: #f8d7da; }}
                    .alert-medium {{ background-color: #fff3cd; }}
                    .alert-low {{ background-color: #d1ecf1; }}
                    .footer {{ text-align: center; margin-top: 30px; font-size: 12px; color: #6c757d; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>광고 캠페인 알림</h1>
                    <p>다음과 같은 알림이 발생했습니다:</p>
                    
                    <table>
                        <tr>
                            <th>유형</th>
                            <th>지표</th>
                            <th>값</th>
                            <th>임계값</th>
                            <th>시간</th>
                        </tr>
            """
            
            # 알림 목록 추가
            for alert in alerts:
                alert_type = alert['type']
                metric = alert['metric']
                value = alert['value']
                threshold = alert.get('threshold', '')
                timestamp = alert.get('timestamp', '')
                
                # 알림 심각도에 따른 스타일 클래스
                severity_class = "alert-medium"
                if alert_type in ['metric_decrease', 'metric_below_min']:
                    if abs(value) > 0.2 or metric in ['roas', 'cvr']:
                        severity_class = "alert-high"
                elif alert_type in ['metric_above_max', 'metric_anomaly']:
                    severity_class = "alert-high"
                
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
                
                # 값 형식 지정
                if metric in ['ctr', 'cvr'] or alert_type in ['metric_decrease', 'metric_increase']:
                    value_text = f"{value:.2%}"
                    threshold_text = f"{threshold:.2%}" if threshold else ""
                else:
                    value_text = f"{value:.2f}" if isinstance(value, (int, float)) else str(value)
                    threshold_text = f"{threshold:.2f}" if isinstance(threshold, (int, float)) else str(threshold)
                
                body += f"""
                        <tr class="{severity_class}">
                            <td>{type_text}</td>
                            <td>{metric_text}</td>
                            <td>{value_text}</td>
                            <td>{threshold_text}</td>
                            <td>{timestamp}</td>
                        </tr>
                """
            
            # 이메일 푸터
            body += f"""
                    </table>
                    
                    <p>자세한 내용은 광고 캠페인 분석 대시보드를 확인해주세요.</p>
                    
                    <div class="footer">
                        <p>광고 캠페인 분석 대시보드 | 생성일: {alert_date}</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # 이메일 메시지 생성
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = subject
            
            # HTML 본문 추가
            msg.attach(MIMEText(body, 'html'))
            
            # SMTP 서버 연결 및 이메일 전송
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(smtp_username, smtp_password)
                server.send_message(msg)
            
            logger.info(f"알림 이메일 전송 완료: {len(alerts)}개 알림, {len(recipients)} 명의 수신자")
            return True
            
        except Exception as e:
            logger.error(f"알림 이메일 전송 중 오류 발생: {e}")
            return False
    
    def _send_slack_alerts(self, alerts):
        """
        Slack으로 알림을 전송합니다.
        
        Args:
            alerts (list): 발생한 알림 목록
            
        Returns:
            bool: 전송 성공 여부
        """
        if not self.slack_config:
            logger.error("Slack 설정이 없습니다.")
            return False
        
        try:
            # Slack Webhook URL
            webhook_url = self.slack_config.get('webhook_url')
            
            if not webhook_url:
                logger.error("Slack Webhook URL이 없습니다.")
                return False
            
            # 알림 내용 생성
            alert_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"🚨 광고 캠페인 알림: {len(alerts)}개 발생"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*알림 시간:* {alert_date}"
                    }
                },
                {
                    "type": "divider"
                }
            ]
            
            # 알림 목록 추가
            for index, alert in enumerate(alerts):
                alert_type = alert['type']
                metric = alert['metric']
                value = alert['value']
                threshold = alert.get('threshold', '')
                
                # 알림 유형에 따른 이모지 및 설명 텍스트
                emoji = "⚠️"
                if alert_type == 'metric_decrease':
                    emoji = "📉"
                    type_text = "지표 감소"
                elif alert_type == 'metric_increase':
                    emoji = "📈"
                    type_text = "지표 증가"
                elif alert_type == 'metric_below_min':
                    emoji = "🔻"
                    type_text = "최소 임계값 미만"
                elif alert_type == 'metric_above_max':
                    emoji = "🔺"
                    type_text = "최대 임계값 초과"
                elif alert_type == 'metric_anomaly':
                    emoji = "🔍"
                    type_text = "이상치 감지"
                else:
                    type_text = alert_type
                
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
                
                # 값 형식 지정
                if metric in ['ctr', 'cvr'] or alert_type in ['metric_decrease', 'metric_increase']:
                    value_text = f"{value:.2%}"
                    threshold_text = f"{threshold:.2%}" if threshold else ""
                else:
                    value_text = f"{value:.2f}" if isinstance(value, (int, float)) else str(value)
                    threshold_text = f"{threshold:.2f}" if isinstance(threshold, (int, float)) else str(threshold)
                
                # 알림 텍스트
                text = f"{emoji} *{type_text}*\n*지표:* {metric_text}\n*값:* {value_text}\n*임계값:* {threshold_text}"
                
                if alert_type == 'metric_anomaly' and 'z_score' in alert:
                    text += f"\n*Z-점수:* {alert['z_score']:.2f}"
                
                # 현재/이전 값 추가 (있는 경우)
                if 'current' in alert and 'previous' in alert:
                    current = alert['current']
                    previous = alert['previous']
                    
                    current_text = f"{current:.2f}" if isinstance(current, (int, float)) else str(current)
                    previous_text = f"{previous:.2f}" if isinstance(previous, (int, float)) else str(previous)
                    
                    text += f"\n*현재값:* {current_text}\n*이전값:* {previous_text}"
                
                # 블록 추가
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": text
                    }
                })
                
                # 알림 항목 사이에 구분선 추가 (마지막 항목 제외)
                if index < len(alerts) - 1:
                    blocks.append({
                        "type": "divider"
                    })
            
            # 메시지 구성
            message = {
                "blocks": blocks,
                "text": f"🚨 광고 캠페인 알림: {len(alerts)}개 발생"  # 알림이 비활성화된, 또는 미지원 클라이언트를 위한 대체 텍스트
            }
            
            # Slack Webhook 호출
            response = requests.post(webhook_url, json=message)
            
            if response.status_code == 200:
                logger.info(f"Slack 알림 전송 완료: {len(alerts)}개 알림")
                return True
            else:
                logger.error(f"Slack 알림 전송 실패: {response.status_code}, {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Slack 알림 전송 중 오류 발생: {e}")
            return False
    
    def load_alert_history(self):
        """
        저장된 알림 기록을 로드합니다.
        
        Returns:
            list: 알림 기록
        """
        try:
            history_path = f"{self.alerts_dir}/alert_history.json"
            
            if os.path.exists(history_path):
                with open(history_path, 'r', encoding='utf-8') as f:
                    self.alert_history = json.load(f)
                logger.info(f"알림 기록 로드 완료: {len(self.alert_history)}개 알림")
            else:
                logger.info("알림 기록 파일이 존재하지 않습니다.")
            
            return self.alert_history
            
        except Exception as e:
            logger.error(f"알림 기록 로드 중 오류 발생: {e}")
            return []
    
    def set_thresholds(self, thresholds):
        """
        알림 임계값을 설정합니다.
        
        Args:
            thresholds (dict): 임계값 딕셔너리
            
        Returns:
            bool: 설정 성공 여부
        """
        try:
            # 기존 임계값과 병합
            self.thresholds.update(thresholds)
            
            # 설정 파일에 저장
            self.config['alerts']['thresholds'] = self.thresholds
            
            with open("config/config.yaml", 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"알림 임계값 업데이트 완료: {thresholds}")
            return True
            
        except Exception as e:
            logger.error(f"알림 임계값 설정 중 오류 발생: {e}")
            return False
    
    def get_default_thresholds(self):
        """
        기본 알림 임계값을 반환합니다.
        
        Returns:
            dict: 기본 임계값 딕셔너리
        """
        return {
            # 변화율 임계값 (예: 0.2 = 20%)
            'spend_change_threshold': 0.2,
            'impressions_change_threshold': 0.3,
            'clicks_change_threshold': 0.25,
            'conversions_change_threshold': 0.2,
            'revenue_change_threshold': 0.2,
            'ctr_change_threshold': 0.15,
            'cvr_change_threshold': 0.2,
            'roas_change_threshold': 0.15,
            'cpc_change_threshold': 0.2,
            'cpa_change_threshold': 0.2,
            
            # 최소 임계값
            'roas_min_threshold': 1.0,
            'ctr_min_threshold': 0.01,
            'cvr_min_threshold': 0.01,
            
            # 최대 임계값
            'cpc_max_threshold': 10000,
            'cpa_max_threshold': 100000,
            
            # 이상치 감지 임계값 (Z-점수)
            'roas_z_score_threshold': 2.5,
            'ctr_z_score_threshold': 2.5,
            'cvr_z_score_threshold': 2.5,
            'spend_z_score_threshold': 3.0,
            'revenue_z_score_threshold': 3.0
        }


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
    
    # 알림 시스템 초기화
    alert_system = AlertSystem()
    
    # 기본 임계값 설정 (실제 운영 시에는 config.yaml에 저장)
    default_thresholds = alert_system.get_default_thresholds()
    alert_system.set_thresholds(default_thresholds)
    
    # 알림 조건 확인
    alerts = alert_system.check_alerts(data, period='daily')
    print(f"발생한 알림: {len(alerts)}개")
    
    # 알림 전송 (테스트용 코드)
    if alerts and False:  # 실제 전송을 방지하기 위해 False로 설정
        # 이메일로 전송
        email_result = alert_system._send_email_alerts(alerts)
        print(f"이메일 전송 결과: {email_result}")
        
        # Slack으로 전송
        slack_result = alert_system._send_slack_alerts(alerts)
        print(f"Slack 전송 결과: {slack_result}")
import pandas as pd
import yaml
import os
import logging
from datetime import datetime, timedelta
import requests
import json
from io import StringIO

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/data_loader.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("data_loader")

class DataLoader:
    """
    광고 캠페인 데이터를 로드하고 전처리하는 클래스
    """
    
    def __init__(self, config_path="config/config.yaml"):
        """
        설정 파일을 로드하고 초기화합니다.
        
        Args:
            config_path (str): 설정 파일 경로
        """
        self.config = self._load_config(config_path)
        self.data_path = self.config.get('data_path', 'data/campaign_data.csv')
        self.api_config = self.config.get('api', {})
        self.processed_data = None
    
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
    
    def load_data(self, file_path=None):
        """
        CSV 파일에서 데이터를 로드합니다.
        
        Args:
            file_path (str, optional): 데이터 파일 경로. 기본값은 설정 파일의 경로입니다.
            
        Returns:
            pandas.DataFrame: 로드된 데이터
        """
        try:
            if file_path is None:
                file_path = self.data_path
            
            logger.info(f"파일에서 데이터 로드 중: {file_path}")
            df = pd.read_csv(file_path)
            
            # 날짜 형식 변환
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            self.processed_data = df
            logger.info(f"데이터 로드 완료: {len(df)} 행")
            return df
        
        except Exception as e:
            logger.error(f"데이터 로드 중 오류 발생: {e}")
            return pd.DataFrame()
    
    def filter_data(self, df=None, filters=None):
        """
        데이터를 필터링합니다.
        
        Args:
            df (pandas.DataFrame, optional): 필터링할 데이터프레임.
            filters (dict, optional): 필터 조건을 담은 딕셔너리.
            
        Returns:
            pandas.DataFrame: 필터링된 데이터
        """
        if df is None:
            df = self.processed_data
        
        if df is None or df.empty:
            logger.warning("필터링할 데이터가 없습니다.")
            return pd.DataFrame()
        
        if not filters:
            return df
        
        filtered_df = df.copy()
        logger.info(f"데이터 필터링 시작: {len(filtered_df)} 행")
        
        try:
            # 날짜 범위 필터링
            if 'date_range' in filters and len(filters['date_range']) == 2:
                start_date, end_date = filters['date_range']
                filtered_df = filtered_df[
                    (filtered_df['date'] >= pd.Timestamp(start_date)) & 
                    (filtered_df['date'] <= pd.Timestamp(end_date))
                ]
            
            # 개별 컬럼 필터링
            for col, value in filters.items():
                if col != 'date_range' and col in filtered_df.columns and value != '전체':
                    filtered_df = filtered_df[filtered_df[col] == value]
            
            logger.info(f"필터링 완료: {len(filtered_df)} 행")
            return filtered_df
        
        except Exception as e:
            logger.error(f"데이터 필터링 중 오류 발생: {e}")
            return df
    
    def get_api_data(self, platform, start_date=None, end_date=None):
        """
        광고 플랫폼 API에서 데이터를 가져옵니다.
        
        Args:
            platform (str): 플랫폼 이름 ('google_ads', 'facebook_ads', 'naver_ads' 등)
            start_date (str, optional): 시작 날짜 (YYYY-MM-DD 형식)
            end_date (str, optional): 종료 날짜 (YYYY-MM-DD 형식)
            
        Returns:
            pandas.DataFrame: API에서 가져온 데이터
        """
        if platform not in self.api_config:
            logger.error(f"플랫폼 '{platform}'에 대한 API 설정이 없습니다.")
            return pd.DataFrame()
        
        # 날짜 설정
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        platform_config = self.api_config[platform]
        
        try:
            # Google Ads API 연동
            if platform == 'google_ads':
                return self._get_google_ads_data(platform_config, start_date, end_date)
            
            # Facebook Ads API 연동
            elif platform == 'facebook_ads':
                return self._get_facebook_ads_data(platform_config, start_date, end_date)
            
            # Naver 광고 API 연동
            elif platform == 'naver_ads':
                return self._get_naver_ads_data(platform_config, start_date, end_date)
            
            else:
                logger.error(f"지원하지 않는 플랫폼입니다: {platform}")
                return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"{platform} API 데이터 가져오기 중 오류 발생: {e}")
            return pd.DataFrame()
    
    def _get_google_ads_data(self, config, start_date, end_date):
        """
        Google Ads API에서 데이터를 가져옵니다.
        
        Args:
            config (dict): API 설정
            start_date (str): 시작 날짜
            end_date (str): 종료 날짜
            
        Returns:
            pandas.DataFrame: API에서 가져온 데이터
        """
        logger.info(f"Google Ads API 데이터 요청 중: {start_date} ~ {end_date}")
        
        # API 요청을 위한 헤더와 파라미터
        headers = {
            'Authorization': f"Bearer {config['api_token']}",
            'Content-Type': 'application/json'
        }
        
        # API URL 및 파라미터 설정
        url = f"{config['api_url']}/v12/customers/{config['customer_id']}/googleAds:search"
        
        # GAQL 쿼리
        query = f"""
        SELECT 
            campaign.id, 
            campaign.name, 
            metrics.impressions, 
            metrics.clicks, 
            metrics.cost_micros, 
            metrics.conversions, 
            metrics.all_conversions_value
        FROM campaign
        WHERE 
            segments.date >= '{start_date}' AND 
            segments.date <= '{end_date}'
        """
        
        payload = {
            'query': query
        }
        
        # API 요청
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            
            # 응답 데이터를 데이터프레임으로 변환
            rows = []
            for row in data.get('results', []):
                campaign = row['campaign']
                metrics = row['metrics']
                
                rows.append({
                    'campaign_id': campaign['id'],
                    'campaign_name': campaign['name'],
                    'platform': 'Google',
                    'date': row['segments']['date'],
                    'impressions': int(metrics['impressions']),
                    'clicks': int(metrics['clicks']),
                    'spend': float(metrics['costMicros']) / 1000000,  # 마이크로 단위를 달러로 변환
                    'conversions': float(metrics['conversions']),
                    'revenue': float(metrics['allConversionsValue']),
                    'ctr': float(metrics['clicks']) / float(metrics['impressions']) if float(metrics['impressions']) > 0 else 0,
                    'cvr': float(metrics['conversions']) / float(metrics['clicks']) if float(metrics['clicks']) > 0 else 0,
                    'roas': float(metrics['allConversionsValue']) / (float(metrics['costMicros']) / 1000000) if float(metrics['costMicros']) > 0 else 0
                })
            
            df = pd.DataFrame(rows)
            logger.info(f"Google Ads 데이터 가져오기 성공: {len(df)} 행")
            return df
        else:
            logger.error(f"Google Ads API 요청 실패: {response.status_code}, {response.text}")
            return pd.DataFrame()
    
    def _get_facebook_ads_data(self, config, start_date, end_date):
        """
        Facebook Ads API에서 데이터를 가져옵니다.
        
        Args:
            config (dict): API 설정
            start_date (str): 시작 날짜
            end_date (str): 종료 날짜
            
        Returns:
            pandas.DataFrame: API에서 가져온 데이터
        """
        logger.info(f"Facebook Ads API 데이터 요청 중: {start_date} ~ {end_date}")
        
        # API 요청을 위한 파라미터
        params = {
            'access_token': config['api_token'],
            'level': 'campaign',
            'fields': 'campaign_id,campaign_name,impressions,clicks,spend,conversions,conversion_values,ctr,cpc',
            'time_range': json.dumps({
                'since': start_date,
                'until': end_date
            }),
            'limit': 1000
        }
        
        # API URL 설정
        url = f"https://graph.facebook.com/v17.0/{config['account_id']}/insights"
        
        # API 요청
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            # 응답 데이터를 데이터프레임으로 변환
            rows = []
            for row in data.get('data', []):
                rows.append({
                    'campaign_id': row['campaign_id'],
                    'campaign_name': row['campaign_name'],
                    'platform': 'Meta',
                    'date': row.get('date_start', start_date),
                    'impressions': int(row.get('impressions', 0)),
                    'clicks': int(row.get('clicks', 0)),
                    'spend': float(row.get('spend', 0)),
                    'conversions': float(row.get('conversions', 0)),
                    'revenue': float(row.get('conversion_values', 0)),
                    'ctr': float(row.get('ctr', 0)),
                    'cvr': float(row.get('conversions', 0)) / float(row.get('clicks', 1)) if float(row.get('clicks', 0)) > 0 else 0,
                    'roas': float(row.get('conversion_values', 0)) / float(row.get('spend', 1)) if float(row.get('spend', 0)) > 0 else 0
                })
            
            df = pd.DataFrame(rows)
            logger.info(f"Facebook Ads 데이터 가져오기 성공: {len(df)} 행")
            return df
        else:
            logger.error(f"Facebook Ads API 요청 실패: {response.status_code}, {response.text}")
            return pd.DataFrame()
    
    def _get_naver_ads_data(self, config, start_date, end_date):
        """
        Naver 광고 API에서 데이터를 가져옵니다.
        
        Args:
            config (dict): API 설정
            start_date (str): 시작 날짜
            end_date (str): 종료 날짜
            
        Returns:
            pandas.DataFrame: API에서 가져온 데이터
        """
        logger.info(f"Naver 광고 API 데이터 요청 중: {start_date} ~ {end_date}")
        
        # API 요청을 위한 헤더
        headers = {
            'X-API-KEY': config['api_key'],
            'X-Customer-ID': config['customer_id'],
            'Content-Type': 'application/json'
        }
        
        # API URL 설정
        url = f"{config['api_url']}/stats"
        
        # 요청 데이터
        payload = {
            'statsType': 'CAMPAIGN',
            'datePreset': 'CUSTOM',
            'startDate': start_date.replace('-', ''),
            'endDate': end_date.replace('-', ''),
            'timeUnit': 'DAY',
            'campaignTypes': ['WEB_SITE', 'SHOPPING']
        }
        
        # API 요청
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            
            # 응답 데이터를 데이터프레임으로 변환
            rows = []
            for row in data.get('data', []):
                stats = row.get('stats', {})
                
                for daily_stat in stats:
                    rows.append({
                        'campaign_id': row.get('campaignId', ''),
                        'campaign_name': row.get('campaignName', ''),
                        'platform': 'Naver',
                        'date': daily_stat.get('dateStart', ''),
                        'impressions': int(daily_stat.get('impressions', 0)),
                        'clicks': int(daily_stat.get('clicks', 0)),
                        'spend': float(daily_stat.get('cost', 0)),
                        'conversions': int(daily_stat.get('conversions', 0)),
                        'revenue': float(daily_stat.get('sales', 0)),
                        'ctr': float(daily_stat.get('ctr', 0)),
                        'cvr': float(daily_stat.get('cvr', 0)),
                        'roas': float(daily_stat.get('roas', 0))
                    })
            
            df = pd.DataFrame(rows)
            logger.info(f"Naver 광고 데이터 가져오기 성공: {len(df)} 행")
            return df
        else:
            logger.error(f"Naver 광고 API 요청 실패: {response.status_code}, {response.text}")
            return pd.DataFrame()
    
    def combine_api_data(self, platforms=None, start_date=None, end_date=None):
        """
        여러 플랫폼의 API 데이터를 결합합니다.
        
        Args:
            platforms (list, optional): 데이터를 가져올 플랫폼 목록
            start_date (str, optional): 시작 날짜
            end_date (str, optional): 종료 날짜
            
        Returns:
            pandas.DataFrame: 결합된 데이터
        """
        if platforms is None:
            platforms = list(self.api_config.keys())
        
        all_data = []
        
        for platform in platforms:
            platform_data = self.get_api_data(platform, start_date, end_date)
            if not platform_data.empty:
                all_data.append(platform_data)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"API 데이터 결합 완료: {len(combined_df)} 행")
            
            # 결합된 데이터 저장
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            output_path = f"data/api_data_{timestamp}.csv"
            combined_df.to_csv(output_path, index=False)
            logger.info(f"API 데이터 저장 완료: {output_path}")
            
            return combined_df
        else:
            logger.warning("결합할 API 데이터가 없습니다.")
            return pd.DataFrame()
    
    def save_processed_data(self, df=None, output_path=None):
        """
        처리된 데이터를 CSV 파일로 저장합니다.
        
        Args:
            df (pandas.DataFrame, optional): 저장할 데이터프레임
            output_path (str, optional): 출력 파일 경로
            
        Returns:
            bool: 저장 성공 여부
        """
        if df is None:
            df = self.processed_data
        
        if df is None or df.empty:
            logger.warning("저장할 데이터가 없습니다.")
            return False
        
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            output_path = f"data/processed_data_{timestamp}.csv"
        
        try:
            # 출력 디렉토리 확인 및 생성
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 데이터 저장
            df.to_csv(output_path, index=False)
            logger.info(f"데이터 저장 완료: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"데이터 저장 중 오류 발생: {e}")
            return False


# 모듈 테스트 코드
if __name__ == "__main__":
    # 로그 디렉토리 생성
    if not os.path.exists("logs"):
        os.makedirs("logs")
    
    loader = DataLoader()
    data = loader.load_data()
    
    # 데이터 필터링 테스트
    filters = {
        'date_range': ['2025-04-01', '2025-04-30'],
        'platform': 'Naver'
    }
    filtered_data = loader.filter_data(data, filters)
    print(f"필터링된 데이터: {len(filtered_data)} 행")
    
    # API 데이터 가져오기 테스트 (설정이 있는 경우만)
    if loader.api_config:
        for platform in loader.api_config:
            platform_data = loader.get_api_data(platform)
            print(f"{platform} 데이터: {len(platform_data)} 행")
        
        # 결합 테스트
        combined_data = loader.combine_api_data()
        print(f"결합된 데이터: {len(combined_data)} 행")
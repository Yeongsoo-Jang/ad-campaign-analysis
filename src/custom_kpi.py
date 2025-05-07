import pandas as pd
import numpy as np
import os
import logging
import yaml
import json
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/custom_kpi.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("custom_kpi")

class CustomKPI:
    """
    광고 캠페인 데이터에 대한 사용자 정의 KPI(핵심 성과 지표)를 관리하는 클래스
    """
    
    def __init__(self, config_path="config/config.yaml"):
        """
        설정 파일을 로드하고 초기화합니다.
        
        Args:
            config_path (str): 설정 파일 경로
        """
        self.config = self._load_config(config_path)
        self.kpi_config = self.config.get('custom_kpi', {})
        self.custom_kpis = self.kpi_config.get('definitions', {})
        self.kpi_targets = self.kpi_config.get('targets', {})
        self.kpi_history = []
        self.kpis_dir = "kpis"
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
        for dir_path in [self.kpis_dir, "logs"]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                logger.info(f"디렉토리 생성됨: {dir_path}")
    
    def add_kpi(self, name, formula, description=None, unit=None, aggregation='sum'):
        """
        새로운 사용자 정의 KPI를 추가합니다.
        
        Args:
            name (str): KPI 이름
            formula (str): KPI 계산 공식
            description (str, optional): KPI 설명
            unit (str, optional): KPI 단위 (예: %, ₩, 회)
            aggregation (str, optional): 집계 방식 ('sum', 'avg', 'min', 'max', 'last', 'first')
            
        Returns:
            bool: 추가 성공 여부
        """
        try:
            # KPI 정의
            kpi_def = {
                'formula': formula,
                'description': description or f"사용자 정의 KPI: {name}",
                'unit': unit,
                'aggregation': aggregation,
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # KPI 저장
            self.custom_kpis[name] = kpi_def
            
            # 설정 업데이트
            self.config['custom_kpi'] = self.config.get('custom_kpi', {})
            self.config['custom_kpi']['definitions'] = self.custom_kpis
            
            # 설정 파일에 저장
            with open("config/config.yaml", 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"KPI 추가됨: {name}, 공식: {formula}")
            return True
            
        except Exception as e:
            logger.error(f"KPI 추가 중 오류 발생: {e}")
            return False
    
    def remove_kpi(self, name):
        """
        사용자 정의 KPI를 제거합니다.
        
        Args:
            name (str): 제거할 KPI 이름
            
        Returns:
            bool: 제거 성공 여부
        """
        try:
            if name in self.custom_kpis:
                # KPI 제거
                del self.custom_kpis[name]
                
                # 관련 타겟도 제거
                if name in self.kpi_targets:
                    del self.kpi_targets[name]
                
                # 설정 업데이트
                self.config['custom_kpi'] = self.config.get('custom_kpi', {})
                self.config['custom_kpi']['definitions'] = self.custom_kpis
                self.config['custom_kpi']['targets'] = self.kpi_targets
                
                # 설정 파일에 저장
                with open("config/config.yaml", 'w', encoding='utf-8') as f:
                    yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
                
                logger.info(f"KPI 제거됨: {name}")
                return True
            else:
                logger.warning(f"KPI를 찾을 수 없음: {name}")
                return False
                
        except Exception as e:
            logger.error(f"KPI 제거 중 오류 발생: {e}")
            return False
    
    def set_kpi_target(self, name, target_value, target_type='min'):
        """
        KPI의 목표값을 설정합니다.
        
        Args:
            name (str): KPI 이름
            target_value (float): 목표값
            target_type (str): 목표 유형 ('min': 최소, 'max': 최대, 'exact': 정확한 값)
            
        Returns:
            bool: 설정 성공 여부
        """
        try:
            if name not in self.custom_kpis and name not in self.get_standard_kpis():
                logger.warning(f"KPI를 찾을 수 없음: {name}")
                return False
            
            # 타겟 설정
            self.kpi_targets[name] = {
                'value': target_value,
                'type': target_type,
                'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # 설정 업데이트
            self.config['custom_kpi'] = self.config.get('custom_kpi', {})
            self.config['custom_kpi']['targets'] = self.kpi_targets
            
            # 설정 파일에 저장
            with open("config/config.yaml", 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"KPI 타겟 설정됨: {name}, 값: {target_value}, 유형: {target_type}")
            return True
            
        except Exception as e:
            logger.error(f"KPI 타겟 설정 중 오류 발생: {e}")
            return False
    
    def calculate_kpis(self, df):
        """
        데이터프레임에 대해 모든 KPI를 계산합니다.
        
        Args:
            df (pandas.DataFrame): 광고 캠페인 데이터
            
        Returns:
            pandas.DataFrame: KPI가 추가된 데이터프레임
        """
        try:
            result_df = df.copy()
            
            # 각 사용자 정의 KPI 계산
            for name, kpi_def in self.custom_kpis.items():
                formula = kpi_def['formula']
                
                try:
                    # 수식 실행 (안전한 방식으로)
                    # 여기서는 문자열 수식을 파이썬 코드로 변환하여 실행하는 대신
                    # 미리 정의된 연산을 사용하여 안전하게 계산합니다.
                    result_df[name] = self._evaluate_formula(result_df, formula)
                    logger.info(f"KPI 계산됨: {name}")
                except Exception as e:
                    logger.error(f"KPI '{name}' 계산 중 오류 발생: {e}")
                    # 에러 발생 시 KPI 컬럼에 NaN 값 설정
                    result_df[name] = np.nan
            
            # KPI 기록 저장
            self._save_kpi_history(result_df)
            
            return result_df
            
        except Exception as e:
            logger.error(f"KPI 계산 중 오류 발생: {e}")
            return df
    
    def _evaluate_formula(self, df, formula):
        """
        안전한 방식으로 KPI 수식을 평가합니다.
        
        Args:
            df (pandas.DataFrame): 데이터프레임
            formula (str): KPI 수식
            
        Returns:
            pandas.Series: 계산 결과
        """
        # 지원되는 연산자 및 함수 목록
        # +, -, *, /, %, >, <, >=, <=, ==, !=, sum(), avg(), min(), max()
        
        # 기본 컬럼 및 연산자 검증
        for column in df.columns:
            if column in formula:
                # 컬럼 이름에 점(.)이 포함된 경우 (예: 'df.column')
                formula = formula.replace(column, f"df['{column}']")
        
        # 수식에 허용되지 않는 파이썬 내장 함수/예약어 사용 방지
        forbidden_keywords = ['import', 'exec', 'eval', 'compile', 'globals', 'locals',
                            'getattr', 'setattr', 'delattr', 'open', 'file',
                            '__import__', 'os', 'sys', 'subprocess']
        
        for keyword in forbidden_keywords:
            if keyword in formula:
                raise ValueError(f"수식에 허용되지 않는 키워드가 포함되어 있습니다: {keyword}")
        
        # 안전한 함수만 허용
        safe_dict = {
            'df': df,
            'sum': np.sum,
            'avg': np.mean,
            'mean': np.mean,
            'min': np.min,
            'max': np.max,
            'abs': np.abs,
            'round': np.round,
            'floor': np.floor,
            'ceil': np.ceil
        }
        
        # 수식 실행 (안전한 범위에서만)
        try:
            result = eval(formula, {"__builtins__": {}}, safe_dict)
            return result
        except Exception as e:
            logger.error(f"수식 평가 중 오류 발생: {formula}, {e}")
            raise
    
    def aggregate_kpis(self, df, group_by=None):
        """
        KPI를 집계합니다.
        
        Args:
            df (pandas.DataFrame): KPI가 포함된 데이터프레임
            group_by (str, optional): 집계 기준 컬럼
            
        Returns:
            pandas.DataFrame: 집계된 KPI 데이터프레임
        """
        try:
            # KPI 컬럼 및 집계 방식 목록
            kpi_aggs = {}
            
            # 사용자 정의 KPI 집계 방식
            for name, kpi_def in self.custom_kpis.items():
                if name in df.columns:
                    agg_method = kpi_def.get('aggregation', 'sum')
                    kpi_aggs[name] = agg_method
            
            # 표준 KPI 집계 방식
            standard_kpis = self.get_standard_kpis()
            for name in standard_kpis:
                if name in df.columns:
                    agg_method = standard_kpis[name].get('aggregation', 'sum')
                    kpi_aggs[name] = agg_method
            
            # 집계 수행
            if group_by and group_by in df.columns:
                result = df.groupby(group_by).agg(kpi_aggs).reset_index()
            else:
                # 집계 함수를 딕셔너리로 변환
                agg_dict = {}
                for col, method in kpi_aggs.items():
                    if method == 'sum':
                        agg_dict[col] = np.sum
                    elif method in ['avg', 'mean']:
                        agg_dict[col] = np.mean
                    elif method == 'min':
                        agg_dict[col] = np.min
                    elif method == 'max':
                        agg_dict[col] = np.max
                    elif method == 'last':
                        agg_dict[col] = lambda x: x.iloc[-1] if len(x) > 0 else np.nan
                    elif method == 'first':
                        agg_dict[col] = lambda x: x.iloc[0] if len(x) > 0 else np.nan
                    else:
                        agg_dict[col] = np.sum  # 기본값
                
                # 전체 데이터에 대해 집계
                result = pd.DataFrame({col: [func(df[col])] for col, func in agg_dict.items()})
            
            logger.info(f"KPI 집계 완료: {len(result)} 행")
            return result
            
        except Exception as e:
            logger.error(f"KPI 집계 중 오류 발생: {e}")
            return pd.DataFrame()
    
    def evaluate_kpi_targets(self, df):
        """
        KPI 값을 목표값과 비교하여 평가합니다.
        
        Args:
            df (pandas.DataFrame): KPI가 포함된 데이터프레임
            
        Returns:
            dict: KPI 평가 결과
        """
        try:
            results = {}
            
            # 각 KPI 목표에 대해 평가
            for name, target in self.kpi_targets.items():
                if name in df.columns:
                    target_value = target['value']
                    target_type = target['type']
                    
                    # 집계된 값 또는 마지막 값 사용
                    if len(df) == 1:
                        actual_value = df[name].iloc[0]
                    else:
                        # 가장 최근 값 사용
                        if 'date' in df.columns:
                            latest_date = df['date'].max()
                            latest_df = df[df['date'] == latest_date]
                            if not latest_df.empty:
                                actual_value = latest_df[name].mean()
                            else:
                                actual_value = df[name].mean()
                        else:
                            actual_value = df[name].mean()
                    
                    # 목표 달성 여부 평가
                    achieved = False
                    
                    if target_type == 'min':
                        achieved = actual_value >= target_value
                    elif target_type == 'max':
                        achieved = actual_value <= target_value
                    elif target_type == 'exact':
                        achieved = abs(actual_value - target_value) < 0.001  # 부동소수점 오차 허용
                    
                    # 목표 달성률 계산
                    if target_value != 0:
                        if target_type == 'min':
                            achievement_rate = actual_value / target_value
                        elif target_type == 'max':
                            achievement_rate = target_value / actual_value if actual_value != 0 else float('inf')
                        else:  # exact
                            achievement_rate = 1 - abs(actual_value - target_value) / abs(target_value)
                    else:
                        achievement_rate = float('inf') if actual_value > 0 else 0
                    
                    # 결과 저장
                    results[name] = {
                        'target_value': target_value,
                        'target_type': target_type,
                        'actual_value': actual_value,
                        'achieved': achieved,
                        'achievement_rate': achievement_rate
                    }
            
            logger.info(f"KPI 목표 평가 완료: {len(results)} 개의 KPI")
            return results
            
        except Exception as e:
            logger.error(f"KPI 목표 평가 중 오류 발생: {e}")
            return {}
    
    def _save_kpi_history(self, df):
        """
        KPI 계산 결과를 기록으로 저장합니다.
        
        Args:
            df (pandas.DataFrame): KPI가 포함된 데이터프레임
        """
        try:
            # 날짜별로 KPI 집계
            if 'date' in df.columns:
                # KPI 컬럼 목록
                kpi_columns = list(self.custom_kpis.keys())
                standard_kpis = self.get_standard_kpis().keys()
                kpi_columns.extend([k for k in standard_kpis if k in df.columns])
                
                # 존재하는 KPI 컬럼만 선택
                available_kpis = [k for k in kpi_columns if k in df.columns]
                
                if available_kpis:
                    # 날짜별 KPI 집계
                    kpi_history = df.groupby('date')[available_kpis].mean().reset_index()
                    
                    # 타임스탬프
                    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                    
                    # 저장 경로
                    history_path = f"{self.kpis_dir}/kpi_history_{timestamp}.csv"
                    
                    # CSV로 저장
                    kpi_history.to_csv(history_path, index=False)
                    logger.info(f"KPI 기록 저장 완료: {history_path}")
                    
                    # 가장 최근 기록은 최신 상태로 유지
                    latest_path = f"{self.kpis_dir}/kpi_history_latest.csv"
                    kpi_history.to_csv(latest_path, index=False)
                    
                    # 메모리에도 저장
                    self.kpi_history = kpi_history.to_dict('records')
            
        except Exception as e:
            logger.error(f"KPI 기록 저장 중 오류 발생: {e}")
    
    def load_kpi_history(self):
        """
        저장된 KPI 기록을 로드합니다.
        
        Returns:
            pandas.DataFrame: KPI 기록
        """
        try:
            latest_path = f"{self.kpis_dir}/kpi_history_latest.csv"
            
            if os.path.exists(latest_path):
                kpi_history = pd.read_csv(latest_path)
                if 'date' in kpi_history.columns:
                    kpi_history['date'] = pd.to_datetime(kpi_history['date'])
                
                logger.info(f"KPI 기록 로드 완료: {len(kpi_history)} 행")
                return kpi_history
            else:
                logger.info("KPI 기록 파일이 존재하지 않습니다.")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"KPI 기록 로드 중 오류 발생: {e}")
            return pd.DataFrame()
    
    def get_standard_kpis(self):
        """
        표준 KPI 정의를 반환합니다.
        
        Returns:
            dict: 표준 KPI 정의
        """
        return {
            'ctr': {
                'description': '클릭률 (Click-Through Rate)',
                'formula': 'clicks / impressions',
                'unit': '%',
                'aggregation': 'avg'
            },
            'cvr': {
                'description': '전환율 (Conversion Rate)',
                'formula': 'conversions / clicks',
                'unit': '%',
                'aggregation': 'avg'
            },
            'roas': {
                'description': '광고 투자 수익률 (Return on Ad Spend)',
                'formula': 'revenue / spend',
                'unit': '',
                'aggregation': 'avg'
            },
            'cpc': {
                'description': '클릭당 비용 (Cost Per Click)',
                'formula': 'spend / clicks',
                'unit': '₩',
                'aggregation': 'avg'
            },
            'cpa': {
                'description': '전환당 비용 (Cost Per Acquisition)',
                'formula': 'spend / conversions',
                'unit': '₩',
                'aggregation': 'avg'
            },
            'roi': {
                'description': '투자 수익률 (Return on Investment)',
                'formula': '(revenue - spend) / spend',
                'unit': '%',
                'aggregation': 'avg'
            }
        }
    
    def get_kpi_definitions(self):
        """
        모든 KPI 정의(표준 및 사용자 정의)를 반환합니다.
        
        Returns:
            dict: KPI 정의
        """
        # 표준 KPI와 사용자 정의 KPI 병합
        all_kpis = self.get_standard_kpis().copy()
        all_kpis.update(self.custom_kpis)
        return all_kpis
    
    def export_kpi_definitions(self, output_path=None):
        """
        KPI 정의를 JSON 파일로 내보냅니다.
        
        Args:
            output_path (str, optional): 출력 파일 경로
            
        Returns:
            str: 저장된 파일 경로
        """
        try:
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                output_path = f"{self.kpis_dir}/kpi_definitions_{timestamp}.json"
            
            # KPI 정의 가져오기
            kpi_definitions = self.get_kpi_definitions()
            
            # JSON으로 저장
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(kpi_definitions, f, ensure_ascii=False, indent=2)
            
            logger.info(f"KPI 정의 내보내기 완료: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"KPI 정의 내보내기 중 오류 발생: {e}")
            return None
    
    def import_kpi_definitions(self, input_path):
        """
        KPI 정의를 JSON 파일에서 가져옵니다.
        
        Args:
            input_path (str): 입력 파일 경로
            
        Returns:
            bool: 가져오기 성공 여부
        """
        try:
            if not os.path.exists(input_path):
                logger.error(f"파일을 찾을 수 없음: {input_path}")
                return False
            
            # JSON에서 로드
            with open(input_path, 'r', encoding='utf-8') as f:
                kpi_definitions = json.load(f)
            
            # 표준 KPI 목록
            standard_kpis = self.get_standard_kpis().keys()
            
            # 사용자 정의 KPI만 가져오기
            custom_kpis = {k: v for k, v in kpi_definitions.items() if k not in standard_kpis}
            
            # KPI 설정 업데이트
            self.custom_kpis.update(custom_kpis)
            
            # 설정 파일에 저장
            self.config['custom_kpi'] = self.config.get('custom_kpi', {})
            self.config['custom_kpi']['definitions'] = self.custom_kpis
            
            with open("config/config.yaml", 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"KPI 정의 가져오기 완료: {len(custom_kpis)} 개의 KPI")
            return True
            
        except Exception as e:
            logger.error(f"KPI 정의 가져오기 중 오류 발생: {e}")
            return False
    
    def get_kpi_suggestions(self, df):
        """
        데이터 분석을 통해 새로운 KPI를 제안합니다.
        
        Args:
            df (pandas.DataFrame): 광고 캠페인 데이터
            
        Returns:
            list: KPI 제안 목록
        """
        suggestions = []
        
        try:
            # 데이터에 있는 컬럼 확인
            available_columns = df.columns.tolist()
            
            # 1. 기본 KPI가 없는 경우 제안
            standard_kpis = self.get_standard_kpis()
            for kpi_name, kpi_def in standard_kpis.items():
                if kpi_name not in self.custom_kpis:
                    formula = kpi_def['formula']
                    description = kpi_def['description']
                    
                    # 포뮬러에 포함된 컬럼이 모두 있는지 확인
                    formula_columns = [col.strip() for col in formula.replace('+', ' ').replace('-', ' ').replace('*', ' ').replace('/', ' ').replace('(', ' ').replace(')', ' ').split()]
                    formula_cols_available = all(col in available_columns for col in formula_columns if col not in ['sum', 'avg', 'min', 'max'])
                    
                    if formula_cols_available:
                        suggestions.append({
                            'name': kpi_name,
                            'formula': formula,
                            'description': description,
                            'type': 'standard'
                        })
            
            # 2. 특정 패턴에 따른 사용자 정의 KPI 제안
            
            # 캠페인별 성과 비교
            if 'campaign_name' in available_columns and 'roas' in available_columns:
                suggestions.append({
                    'name': 'campaign_roas_variance',
                    'formula': 'df.groupby("campaign_name")["roas"].std() / df.groupby("campaign_name")["roas"].mean()',
                    'description': '캠페인별 ROAS 변동성 (낮을수록 안정적)',
                    'type': 'custom'
                })
            
            # 플랫폼별 효율성
            if 'platform' in available_columns and 'roas' in available_columns:
                suggestions.append({
                    'name': 'platform_efficiency',
                    'formula': 'df.groupby("platform")["roas"].mean() / df.groupby("platform")["spend"].sum()',
                    'description': '플랫폼별 투자 효율성 (높을수록 효율적)',
                    'type': 'custom'
                })
            
            # 크리에이티브 효과
            if 'creative_type' in available_columns and 'cvr' in available_columns:
                suggestions.append({
                    'name': 'creative_effectiveness',
                    'formula': 'df.groupby("creative_type")["cvr"].mean() * df.groupby("creative_type")["ctr"].mean()',
                    'description': '크리에이티브 효과 (클릭률과 전환율의 조합)',
                    'type': 'custom'
                })
            
            # 타겟팅 효율성
            if 'target_age' in available_columns and 'target_gender' in available_columns and 'roas' in available_columns:
                suggestions.append({
                    'name': 'targeting_efficiency',
                    'formula': 'df.groupby(["target_age", "target_gender"])["roas"].mean() / df.groupby(["target_age", "target_gender"])["spend"].sum()',
                    'description': '타겟팅 효율성 (높을수록 효율적)',
                    'type': 'custom'
                })
            
            # 광고 위치 효과
            if 'ad_position' in available_columns and 'ctr' in available_columns:
                suggestions.append({
                    'name': 'position_effect',
                    'formula': 'df.groupby("ad_position")["ctr"].mean() / df.groupby("ad_position")["cpc"].mean()',
                    'description': '광고 위치 효과 (클릭률 대비 비용)',
                    'type': 'custom'
                })
            
            # 예산 효율성
            if 'daily_budget' in available_columns and 'spend' in available_columns and 'roas' in available_columns:
                suggestions.append({
                    'name': 'budget_efficiency',
                    'formula': 'df["roas"] * (df["spend"] / df["daily_budget"])',
                    'description': '예산 효율성 (ROAS와 예산 활용률의 조합)',
                    'type': 'custom'
                })
            
            logger.info(f"KPI 제안 생성 완료: {len(suggestions)} 개의 제안")
            return suggestions
            
        except Exception as e:
            logger.error(f"KPI 제안 생성 중 오류 발생: {e}")
            return suggestions


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
    
    # 사용자 정의 KPI 초기화
    custom_kpi = CustomKPI()
    
    # 사용자 정의 KPI 추가
    custom_kpi.add_kpi(
        name='effective_cvr',
        formula='conversions / impressions',
        description='노출당 전환율 (CTR와 CVR의 곱)',
        unit='%',
        aggregation='avg'
    )
    
    custom_kpi.add_kpi(
        name='revenue_per_impression',
        formula='revenue / impressions',
        description='노출당 수익',
        unit='₩',
        aggregation='avg'
    )
    
    # KPI 계산
    data_with_kpis = custom_kpi.calculate_kpis(data)
    
    # KPI 집계
    aggregated_kpis = custom_kpi.aggregate_kpis(data_with_kpis)
    print("집계된 KPI:")
    print(aggregated_kpis)
    
    # KPI 제안
    suggestions = custom_kpi.get_kpi_suggestions(data)
    print(f"\nKPI 제안 ({len(suggestions)}개):")
    for suggestion in suggestions:
        print(f"- {suggestion['name']}: {suggestion['description']}")
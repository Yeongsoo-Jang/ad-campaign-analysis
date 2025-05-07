import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from scipy import stats
import logging
import os

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/preprocessing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("preprocessing")

class DataPreprocessor:
    """
    광고 캠페인 데이터의 전처리를 수행하는 클래스
    """
    
    def __init__(self):
        """
        DataPreprocessor 클래스 초기화
        """
        # 로그 디렉토리 생성
        if not os.path.exists("logs"):
            os.makedirs("logs")
            logger.info("로그 디렉토리 생성됨")
            
        # 스케일러 초기화
        self.scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler()
        }
    
    def preprocess_data(self, df, remove_outliers=True, outlier_method='iqr', 
                        normalize=True, normalization_method='robust',
                        create_features=True):
        """
        데이터 전처리를 수행합니다.
        
        Args:
            df (pandas.DataFrame): 원본 광고 캠페인 데이터
            remove_outliers (bool): 이상치 제거 여부
            outlier_method (str): 이상치 탐지 방법 ('iqr', 'zscore')
            normalize (bool): 정규화 여부
            normalization_method (str): 정규화 방법 ('standard', 'robust', 'minmax')
            create_features (bool): 파생변수 생성 여부
            
        Returns:
            pandas.DataFrame: 전처리된 데이터
            dict: 적용된 전처리 단계 정보
        """
        if df.empty:
            logger.warning("데이터가 비어 있습니다.")
            return df, {}
        
        # 데이터 복사
        processed_df = df.copy()
        
        # 전처리 단계 정보
        preprocessing_info = {
            'original_shape': processed_df.shape,
            'steps_applied': []
        }
        
        # 1. 결측치 처리
        processed_df, missings_info = self._handle_missing_values(processed_df)
        preprocessing_info['missing_values'] = missings_info
        preprocessing_info['steps_applied'].append('missing_values_handling')
        
        # 2. 이상치 제거
        if remove_outliers:
            processed_df, outliers_info = self._remove_outliers(processed_df, method=outlier_method)
            preprocessing_info['outliers'] = outliers_info
            preprocessing_info['steps_applied'].append(f'outlier_removal_{outlier_method}')
        
        # 3. 정규화
        if normalize:
            processed_df, normalization_info = self._normalize_data(processed_df, method=normalization_method)
            preprocessing_info['normalization'] = normalization_info
            preprocessing_info['steps_applied'].append(f'normalization_{normalization_method}')
        
        # 4. 파생변수 생성
        if create_features:
            processed_df, features_info = self._create_features(processed_df)
            preprocessing_info['features'] = features_info
            preprocessing_info['steps_applied'].append('feature_creation')
        
        # 최종 정보 추가
        preprocessing_info['final_shape'] = processed_df.shape
        
        logger.info(f"데이터 전처리 완료: {len(processed_df)} 행, {len(processed_df.columns)} 열")
        
        return processed_df, preprocessing_info
    
    def _handle_missing_values(self, df):
        """
        결측치를 처리합니다.
        
        Args:
            df (pandas.DataFrame): 처리할 데이터프레임
            
        Returns:
            pandas.DataFrame: 결측치가 처리된 데이터프레임
            dict: 결측치 처리 정보
        """
        before_rows = len(df)
        
        # 결측치 정보 수집
        missing_info = {
            'before': {
                'total_missing': df.isnull().sum().sum(),
                'missing_by_column': df.isnull().sum().to_dict()
            }
        }
        
        # 수치형 열의 결측치는 중앙값으로 대체
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        # 범주형 열의 결측치는 최빈값으로 대체
        categorical_cols = df.select_dtypes(exclude=['number']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0])
        
        # 결측치 처리 후 정보 수집
        missing_info['after'] = {
            'total_missing': df.isnull().sum().sum(),
            'missing_by_column': df.isnull().sum().to_dict()
        }
        
        # 행 수 변화 확인
        after_rows = len(df)
        missing_info['rows_affected'] = before_rows - after_rows
        
        return df, missing_info
    
    def _remove_outliers(self, df, method='iqr'):
        """
        이상치를 탐지하고 제거합니다.
        
        Args:
            df (pandas.DataFrame): 처리할 데이터프레임
            method (str): 이상치 탐지 방법 ('iqr', 'zscore')
            
        Returns:
            pandas.DataFrame: 이상치가 제거된 데이터프레임
            dict: 이상치 제거 정보
        """
        before_rows = len(df)
        outliers_info = {'method': method, 'outliers_by_column': {}}
        
        # 수치형 열만 선택
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # 지표와 직접 관련된 중요 열만 선택 (비율/ID/날짜 등 제외)
        outlier_cols = ['spend', 'impressions', 'clicks', 'conversions', 'revenue']
        outlier_cols = [col for col in outlier_cols if col in numeric_cols]
        
        if method == 'iqr':
            # IQR(사분위 범위) 방법
            for col in outlier_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # 이상치 식별
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outliers_info['outliers_by_column'][col] = len(outliers)
                
                # 극단적인 이상치만 제거 (상한값/하한값으로 대체)
                df.loc[df[col] < lower_bound, col] = lower_bound
                df.loc[df[col] > upper_bound, col] = upper_bound
        
        elif method == 'zscore':
            # Z-점수 방법
            for col in outlier_cols:
                z_scores = np.abs(stats.zscore(df[col]))
                outliers = df[z_scores > 3]  # 표준편차 3 이상을 이상치로 간주
                outliers_info['outliers_by_column'][col] = len(outliers)
                
                # 극단적인 이상치만 제거 (Z-점수가 3 이상인 데이터)
                df.loc[z_scores > 3, col] = df[col].median()
        
        # 행 수 변화 확인
        after_rows = len(df)
        outliers_info['rows_before'] = before_rows
        outliers_info['rows_after'] = after_rows
        outliers_info['total_outliers_handled'] = sum(outliers_info['outliers_by_column'].values())
        
        return df, outliers_info
    
    def _normalize_data(self, df, method='robust'):
        """
        데이터를 정규화합니다.
        
        Args:
            df (pandas.DataFrame): 처리할 데이터프레임
            method (str): 정규화 방법 ('standard', 'robust', 'minmax')
            
        Returns:
            pandas.DataFrame: 정규화된 데이터프레임
            dict: 정규화 정보
        """
        normalization_info = {'method': method, 'columns_normalized': []}
        
        # 수치형 열만 선택
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # 정규화할 열 선택 (비율 제외)
        normalize_cols = ['spend', 'impressions', 'clicks', 'conversions', 'revenue']
        normalize_cols = [col for col in normalize_cols if col in numeric_cols]
        
        # 선택된 스케일러
        scaler = self.scalers.get(method, self.scalers['robust'])
        
        # 정규화 적용
        if normalize_cols:
            # 스케일러 학습 및 변환
            df[normalize_cols] = scaler.fit_transform(df[normalize_cols])
            normalization_info['columns_normalized'] = normalize_cols
        
        return df, normalization_info
    
    def _create_features(self, df):
        """
        유용한 파생변수를 생성합니다.
        
        Args:
            df (pandas.DataFrame): 처리할 데이터프레임
            
        Returns:
            pandas.DataFrame: 파생변수가 추가된 데이터프레임
            dict: 파생변수 생성 정보
        """
        features_info = {'created_features': []}
        
        # 1. 효율성 지표
        if all(col in df.columns for col in ['revenue', 'spend']):
            # ROI (Return on Investment)
            df['roi'] = (df['revenue'] - df['spend']) / df['spend']
            features_info['created_features'].append('roi')
        
        # 2. 복합 지표
        if all(col in df.columns for col in ['ctr', 'cvr']):
            # 효과적 전환율 (CTR * CVR): 노출 대비 전환율
            df['effective_cvr'] = df['ctr'] * df['cvr']
            features_info['created_features'].append('effective_cvr')
        
        # 3. 비용 효율성 지표
        if all(col in df.columns for col in ['spend', 'impressions']):
            # CPM (Cost Per Mille): 천 회 노출당 비용
            df['cpm'] = (df['spend'] / df['impressions']) * 1000
            features_info['created_features'].append('cpm')
        
        # 4. 수익성 지표
        if all(col in df.columns for col in ['revenue', 'conversions']):
            # 전환당 수익
            df['revenue_per_conversion'] = df['revenue'] / df['conversions'].replace(0, 1)
            features_info['created_features'].append('revenue_per_conversion')
        
        # 5. 날짜 기반 특성
        if 'date' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
            
            # 요일
            df['day_of_week'] = df['date'].dt.dayofweek
            features_info['created_features'].append('day_of_week')
            
            # 월
            df['month'] = df['date'].dt.month
            features_info['created_features'].append('month')
            
            # 주말 여부
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            features_info['created_features'].append('is_weekend')
        
        # 6. 효율성 순위 (캠페인별)
        if 'roas' in df.columns and 'campaign_name' in df.columns:
            # 각 캠페인의 평균 ROAS 계산
            campaign_avg_roas = df.groupby('campaign_name')['roas'].mean()
            
            # 전체 평균 ROAS
            overall_avg_roas = df['roas'].mean()
            
            # 각 캠페인의 ROAS 효율성 (전체 평균 대비)
            campaign_efficiency = campaign_avg_roas / overall_avg_roas
            
            # 효율성 매핑
            df['campaign_efficiency'] = df['campaign_name'].map(campaign_efficiency)
            features_info['created_features'].append('campaign_efficiency')
            
            # 효율성 순위
            efficiency_rank = campaign_avg_roas.rank(ascending=False)
            df['campaign_rank'] = df['campaign_name'].map(efficiency_rank)
            features_info['created_features'].append('campaign_rank')
        
        return df, features_info
    
    def evaluate_preprocessing(self, original_df, processed_df, target='roas'):
        """
        전처리 전후의 데이터 품질을 평가합니다.
        
        Args:
            original_df (pandas.DataFrame): 원본 데이터
            processed_df (pandas.DataFrame): 전처리된 데이터
            target (str): 타겟 변수
            
        Returns:
            dict: 평가 결과
        """
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score, mean_squared_error
        
        evaluation = {
            'original': {},
            'processed': {},
            'comparison': {}
        }
        
        # 타겟 변수가 존재하는지 확인
        if target not in original_df.columns or target not in processed_df.columns:
            logger.warning(f"타겟 변수 '{target}'가 데이터에 없습니다.")
            return evaluation
        
        # 원본 데이터 통계
        evaluation['original']['shape'] = original_df.shape
        evaluation['original']['missing_values'] = original_df.isnull().sum().sum()
        
        # 전처리된 데이터 통계
        evaluation['processed']['shape'] = processed_df.shape
        evaluation['processed']['missing_values'] = processed_df.isnull().sum().sum()
        
        # 비교
        evaluation['comparison']['rows_diff'] = processed_df.shape[0] - original_df.shape[0]
        evaluation['comparison']['columns_diff'] = processed_df.shape[1] - original_df.shape[1]
        
        # 간단한 예측 모델로 성능 비교 (타겟 변수가 수치형인 경우)
        try:
            # 원본 데이터 성능
            original_features = original_df.select_dtypes(include=['number']).columns.tolist()
            original_features = [f for f in original_features if f != target and not pd.isna(original_df[f]).any()]
            
            if len(original_features) > 0:
                # 원본 데이터로 모델 학습
                X_orig = original_df[original_features]
                y_orig = original_df[target]
                
                X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
                    X_orig, y_orig, test_size=0.2, random_state=42
                )
                
                model_orig = LinearRegression()
                model_orig.fit(X_train_orig, y_train_orig)
                
                y_pred_orig = model_orig.predict(X_test_orig)
                r2_orig = r2_score(y_test_orig, y_pred_orig)
                rmse_orig = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
                
                evaluation['original']['r2_score'] = r2_orig
                evaluation['original']['rmse'] = rmse_orig
            
            # 전처리된 데이터 성능
            processed_features = processed_df.select_dtypes(include=['number']).columns.tolist()
            processed_features = [f for f in processed_features if f != target and not pd.isna(processed_df[f]).any()]
            
            if len(processed_features) > 0:
                # 전처리된 데이터로 모델 학습
                X_proc = processed_df[processed_features]
                y_proc = processed_df[target]
                
                X_train_proc, X_test_proc, y_train_proc, y_test_proc = train_test_split(
                    X_proc, y_proc, test_size=0.2, random_state=42
                )
                
                model_proc = LinearRegression()
                model_proc.fit(X_train_proc, y_train_proc)
                
                y_pred_proc = model_proc.predict(X_test_proc)
                r2_proc = r2_score(y_test_proc, y_pred_proc)
                rmse_proc = np.sqrt(mean_squared_error(y_test_proc, y_pred_proc))
                
                evaluation['processed']['r2_score'] = r2_proc
                evaluation['processed']['rmse'] = rmse_proc
                
                # 성능 향상도
                if 'r2_score' in evaluation['original']:
                    evaluation['comparison']['r2_improvement'] = r2_proc - r2_orig
                    evaluation['comparison']['rmse_improvement'] = rmse_orig - rmse_proc
                    evaluation['comparison']['r2_improvement_percent'] = (r2_proc - r2_orig) / max(abs(r2_orig), 1e-10) * 100
                    evaluation['comparison']['rmse_improvement_percent'] = (rmse_orig - rmse_proc) / max(rmse_orig, 1e-10) * 100
        
        except Exception as e:
            logger.error(f"성능 비교 중 오류 발생: {e}")
        
        return evaluation


# 모듈 테스트 코드
if __name__ == "__main__":
    # 테스트 데이터 로드
    data_path = "data/campaign_data.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"데이터 로드 완료: {len(df)} 행")
        
        # 데이터 전처리기 초기화
        preprocessor = DataPreprocessor()
        
        # 이상치 제거 및 정규화 적용
        processed_df, preprocessing_info = preprocessor.preprocess_data(
            df, 
            remove_outliers=True, 
            outlier_method='iqr',
            normalize=True,
            normalization_method='robust',
            create_features=True
        )
        
        print(f"전처리 완료: {len(processed_df)} 행, {len(processed_df.columns)} 열")
        print(f"적용된 전처리 단계: {preprocessing_info['steps_applied']}")
        
        if 'features' in preprocessing_info:
            print(f"생성된 파생변수: {preprocessing_info['features']['created_features']}")
        
        # 전처리 효과 평가
        evaluation = preprocessor.evaluate_preprocessing(df, processed_df, target='roas')
        
        if 'comparison' in evaluation and 'r2_improvement' in evaluation['comparison']:
            print(f"R² 향상도: {evaluation['comparison']['r2_improvement']:.4f}")
            print(f"RMSE 향상도: {evaluation['comparison']['rmse_improvement']:.4f}")
    else:
        print(f"데이터 파일을 찾을 수 없습니다: {data_path}")
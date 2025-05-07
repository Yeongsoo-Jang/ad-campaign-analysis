import pandas as pd
import numpy as np
import logging
import pickle
import os
import yaml
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import statsmodels.api as sm
from statsmodels.formula.api import ols

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/modeling.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("modeling")

class ModelTrainer:
    """
    광고 캠페인 데이터에 대한 예측 모델을 훈련시키는 클래스
    """
    
    def __init__(self, config_path="config/config.yaml"):
        """
        설정 파일을 로드하고 초기화합니다.
        
        Args:
            config_path (str): 설정 파일 경로
        """
        self.config = self._load_config(config_path)
        self.model_config = self.config.get('modeling', {})
        self.models = {}
        self.best_model = None
        self.feature_importance = None
        self.trained_models_dir = "models"
        self.ensure_model_dir()
    
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
    
    def ensure_model_dir(self):
        """
        모델 저장 디렉토리가 존재하는지 확인하고, 없다면 생성합니다.
        """
        if not os.path.exists(self.trained_models_dir):
            os.makedirs(self.trained_models_dir)
            logger.info(f"모델 디렉토리 생성됨: {self.trained_models_dir}")
    
    def preprocess_data(self, df, target='roas', test_size=0.2, random_state=42):
        """
        모델 훈련을 위해 데이터를 전처리합니다.
        
        Args:
            df (pandas.DataFrame): 원본 데이터
            target (str): 타겟 변수 이름
            test_size (float): 테스트 세트 비율
            random_state (int): 랜덤 시드
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, preprocessor)
        """
        logger.info("데이터 전처리 시작")
        
        try:
            # 가능한 독립변수 목록
            numeric_features = self.model_config.get('numeric_features', [
                'daily_budget', 'spend', 'impressions', 'clicks', 'ctr', 
                'conversions', 'cvr'
            ])
            
            categorical_features = self.model_config.get('categorical_features', [
                'platform', 'target_age', 'target_gender', 'creative_type', 'ad_position'
            ])
            
            # 실제 데이터에 있는 컬럼만 선택
            numeric_features = [col for col in numeric_features if col in df.columns]
            categorical_features = [col for col in categorical_features if col in df.columns]
            
            # 타겟 변수가 없는 경우 오류
            if target not in df.columns:
                logger.error(f"타겟 변수 '{target}'가 데이터에 없습니다.")
                return None
            
            # 불필요한 컬럼 제거
            exclude_columns = self.model_config.get('exclude_columns', [
                'date', 'campaign_id', 'campaign_name', 'revenue'
            ])
            exclude_columns = [col for col in exclude_columns if col in df.columns and col != target]
            
            # 데이터 분할
            X = df.drop(columns=[target] + exclude_columns)
            y = df[target]
            
            # 전처리기 설정
            numeric_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])
            
            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            
            available_num_features = [f for f in numeric_features if f in X.columns]
            available_cat_features = [f for f in categorical_features if f in X.columns]
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, available_num_features),
                    ('cat', categorical_transformer, available_cat_features)
                ]
            )
            
            # 훈련/테스트 데이터 분할
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            logger.info(f"데이터 전처리 완료: {X_train.shape[0]} 훈련 샘플, {X_test.shape[0]} 테스트 샘플")
            
            return X_train, X_test, y_train, y_test, preprocessor, X.columns.tolist()
        
        except Exception as e:
            logger.error(f"데이터 전처리 중 오류 발생: {e}")
            return None
    
    def train_models(self, X_train, y_train, preprocessor, feature_names=None):
        """
        여러 회귀 모델을 훈련시킵니다.
        
        Args:
            X_train (pandas.DataFrame): 훈련 독립변수
            y_train (pandas.Series): 훈련 종속변수
            preprocessor (ColumnTransformer): 데이터 전처리기
            feature_names (list): 특성 이름 목록
            
        Returns:
            dict: 훈련된 모델들의 딕셔너리
        """
        logger.info("모델 훈련 시작")
        
        try:
            # 사용할 모델 정의
            models = {
                'linear_regression': Pipeline([
                    ('preprocessor', preprocessor),
                    ('regressor', LinearRegression())
                ]),
                
                'ridge': Pipeline([
                    ('preprocessor', preprocessor),
                    ('regressor', Ridge(alpha=1.0))
                ]),
                
                'lasso': Pipeline([
                    ('preprocessor', preprocessor),
                    ('regressor', Lasso(alpha=0.1))
                ]),
                
                'elastic_net': Pipeline([
                    ('preprocessor', preprocessor),
                    ('regressor', ElasticNet(alpha=0.1, l1_ratio=0.5))
                ]),
                
                'random_forest': Pipeline([
                    ('preprocessor', preprocessor),
                    ('regressor', RandomForestRegressor(
                        n_estimators=100, 
                        max_depth=None, 
                        min_samples_split=2, 
                        random_state=42
                    ))
                ]),
                
                'gradient_boosting': Pipeline([
                    ('preprocessor', preprocessor),
                    ('regressor', GradientBoostingRegressor(
                        n_estimators=100, 
                        learning_rate=0.1, 
                        max_depth=3, 
                        random_state=42
                    ))
                ]),
                
                'xgboost': Pipeline([
                    ('preprocessor', preprocessor),
                    ('regressor', XGBRegressor(
                        n_estimators=100, 
                        learning_rate=0.1, 
                        max_depth=3, 
                        random_state=42
                    ))
                ])
            }
            
            # 각 모델 훈련
            trained_models = {}
            model_scores = {}
            
            for name, model in models.items():
                logger.info(f"{name} 모델 훈련 중...")
                model.fit(X_train, y_train)
                
                # 교차 검증 점수
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                avg_cv_score = np.mean(cv_scores)
                
                trained_models[name] = model
                model_scores[name] = avg_cv_score
                
                logger.info(f"{name} 훈련 완료, 평균 CV R^2: {avg_cv_score:.4f}")
            
            # 최고 성능 모델 선택
            best_model_name = max(model_scores, key=model_scores.get)
            self.best_model = trained_models[best_model_name]
            
            logger.info(f"최고 성능 모델: {best_model_name}, R^2: {model_scores[best_model_name]:.4f}")
            
            # 특성 중요도 계산 (가능한 경우)
            self.calculate_feature_importance(trained_models, feature_names)
            
            self.models = trained_models
            return trained_models
        
        except Exception as e:
            logger.error(f"모델 훈련 중 오류 발생: {e}")
            return {}
    
    def calculate_feature_importance(self, trained_models, feature_names=None):
        """
        각 모델의 특성 중요도를 계산합니다.
        
        Args:
            trained_models (dict): 훈련된 모델들의 딕셔너리
            feature_names (list): 특성 이름 목록
            
        Returns:
            pandas.DataFrame: 특성 중요도 데이터프레임
        """
        try:
            importance_df = pd.DataFrame()
            
            for name, model in trained_models.items():
                # 트리 기반 모델에 대한 특성 중요도 계산
                if name in ['random_forest', 'gradient_boosting', 'xgboost']:
                    # 전처리된 특성 이름 가져오기
                    preprocessor = model.named_steps['preprocessor']
                    regressor = model.named_steps['regressor']
                    
                    # 특성 중요도 가져오기
                    importances = regressor.feature_importances_
                    
                    # 변환된 특성 이름 가져오기
                    transformed_features = []
                    
                    if hasattr(preprocessor, 'transformers_'):
                        # 수치형 특성
                        if 'num' in dict(preprocessor.transformers_):
                            num_features = preprocessor.transformers_[0][2]
                            transformed_features.extend(num_features)
                        
                        # 범주형 특성 (원핫 인코딩)
                        if 'cat' in dict(preprocessor.transformers_):
                            cat_features = preprocessor.transformers_[1][2]
                            cat_transformer = preprocessor.transformers_[1][1]
                            
                            if hasattr(cat_transformer, 'named_steps') and 'onehot' in cat_transformer.named_steps:
                                onehot = cat_transformer.named_steps['onehot']
                                if hasattr(onehot, 'get_feature_names_out'):
                                    cat_transformed = onehot.get_feature_names_out(cat_features)
                                    transformed_features.extend(cat_transformed)
                    
                    # 특성 중요도 딕셔너리 생성
                    if len(transformed_features) == len(importances):
                        importance_dict = dict(zip(transformed_features, importances))
                        importance_series = pd.Series(importance_dict, name=name)
                        importance_df = pd.concat([importance_df, importance_series], axis=1)
            
            if not importance_df.empty:
                # 특성 중요도 정렬
                importance_df['average'] = importance_df.mean(axis=1)
                importance_df = importance_df.sort_values('average', ascending=False)
                
                logger.info(f"특성 중요도 계산 완료: {len(importance_df)} 특성")
                self.feature_importance = importance_df
                return importance_df
            else:
                logger.warning("특성 중요도를 계산할 수 없습니다.")
                return None
                
        except Exception as e:
            logger.error(f"특성 중요도 계산 중 오류 발생: {e}")
            return None
    
    def evaluate_models(self, X_test, y_test):
        """
        훈련된 모델들을 평가합니다.
        
        Args:
            X_test (pandas.DataFrame): 테스트 독립변수
            y_test (pandas.Series): 테스트 종속변수
            
        Returns:
            pandas.DataFrame: 모델 평가 결과
        """
        if not self.models:
            logger.error("훈련된 모델이 없습니다.")
            return None
        
        try:
            results = []
            
            for name, model in self.models.items():
                # 예측
                y_pred = model.predict(X_test)
                
                # 평가 지표 계산
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results.append({
                    'model': name,
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2
                })
            
            # 결과를 데이터프레임으로 변환
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('r2', ascending=False)
            
            logger.info("모델 평가 완료")
            return results_df
            
        except Exception as e:
            logger.error(f"모델 평가 중 오류 발생: {e}")
            return None
    
    def fine_tune_model(self, X_train, y_train, X_test, y_test, model_name='random_forest'):
        """
        선택한 모델을 세부 튜닝합니다.
        
        Args:
            X_train (pandas.DataFrame): 훈련 독립변수
            y_train (pandas.Series): 훈련 종속변수
            X_test (pandas.DataFrame): 테스트 독립변수
            y_test (pandas.Series): 테스트 종속변수
            model_name (str): 튜닝할 모델 이름
            
        Returns:
            Pipeline: 튜닝된 모델
        """
        if model_name not in self.models:
            logger.error(f"모델 '{model_name}'이(가) 존재하지 않습니다.")
            return None
        
        try:
            model = self.models[model_name]
            preprocessor = model.named_steps['preprocessor']
            
            # 모델별 하이퍼파라미터 그리드 설정
            param_grids = {
                'random_forest': {
                    'regressor__n_estimators': [50, 100, 200],
                    'regressor__max_depth': [None, 10, 20, 30],
                    'regressor__min_samples_split': [2, 5, 10],
                    'regressor__min_samples_leaf': [1, 2, 4]
                },
                'gradient_boosting': {
                    'regressor__n_estimators': [50, 100, 200],
                    'regressor__learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'regressor__max_depth': [3, 5, 7],
                    'regressor__min_samples_split': [2, 5, 10]
                },
                'xgboost': {
                    'regressor__n_estimators': [50, 100, 200],
                    'regressor__learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'regressor__max_depth': [3, 5, 7],
                    'regressor__subsample': [0.7, 0.8, 0.9, 1.0]
                },
                'linear_regression': {},  # 튜닝할 파라미터 없음
                'ridge': {
                    'regressor__alpha': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
                },
                'lasso': {
                    'regressor__alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0]
                },
                'elastic_net': {
                    'regressor__alpha': [0.001, 0.01, 0.1, 0.5, 1.0],
                    'regressor__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
                }
            }
            
            # 해당 모델에 대한 파라미터 그리드 선택
            param_grid = param_grids.get(model_name, {})
            
            if not param_grid:
                logger.info(f"모델 '{model_name}'에 대한 그리드 서치를 수행하지 않습니다.")
                return model
            
            # 그리드 서치 수행
            logger.info(f"모델 '{model_name}' 그리드 서치 시작")
            
            grid_search = GridSearchCV(
                model, 
                param_grid, 
                cv=5, 
                scoring='r2', 
                n_jobs=-1,  # 모든 코어 사용
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # 최적 모델
            best_model = grid_search.best_estimator_
            
            # 성능 평가
            best_score = grid_search.best_score_
            best_params = grid_search.best_params_
            
            logger.info(f"최적 파라미터: {best_params}")
            logger.info(f"최적 CV 점수 (R^2): {best_score:.4f}")
            
            # 테스트 데이터 성능 평가
            y_pred = best_model.predict(X_test)
            test_r2 = r2_score(y_test, y_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            logger.info(f"최적 모델 테스트 성능 - R^2: {test_r2:.4f}, RMSE: {test_rmse:.4f}")
            
            # 최적 모델 저장
            self.models[model_name] = best_model
            
            # 전체 모델 중 최고 성능 모델 업데이트
            if self.best_model is None or test_r2 > self.best_model_score:
                self.best_model = best_model
                self.best_model_score = test_r2
                self.best_model_name = model_name
                logger.info(f"새로운 최고 성능 모델: {model_name}, R^2: {test_r2:.4f}")
            
            return best_model
            
        except Exception as e:
            logger.error(f"모델 세부 튜닝 중 오류 발생: {e}")
            return None
    
    def predict(self, X, model_name=None):
        """
        주어진 데이터에 대한 예측을 수행합니다.
        
        Args:
            X (pandas.DataFrame): 예측할 독립변수
            model_name (str, optional): 사용할 모델 이름. None이면 최고 성능 모델 사용
            
        Returns:
            numpy.ndarray: 예측값
        """
        if model_name and model_name in self.models:
            model = self.models[model_name]
        elif self.best_model:
            model = self.best_model
        else:
            logger.error("사용 가능한 모델이 없습니다.")
            return None
        
        try:
            predictions = model.predict(X)
            return predictions
            
        except Exception as e:
            logger.error(f"예측 중 오류 발생: {e}")
            return None
    
    def get_statsmodels_regression(self, df, target='roas', features=None):
        """
        statsmodels를 사용한 회귀 분석을 수행합니다.
        
        Args:
            df (pandas.DataFrame): 데이터프레임
            target (str): 타겟 변수 이름
            features (list, optional): 사용할 특성 목록
            
        Returns:
            statsmodels.regression.linear_model.RegressionResultsWrapper: 회귀 분석 결과
        """
        try:
            if features is None:
                # 수치 특성만 선택
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                features = [col for col in numeric_cols if col != target]
            
            # 공식 구성
            formula = f"{target} ~ " + " + ".join(features)
            
            # 모델 구성 및 학습
            model = ols(formula, data=df).fit()
            
            logger.info(f"statsmodels 회귀 분석 완료, R^2: {model.rsquared:.4f}")
            return model
            
        except Exception as e:
            logger.error(f"statsmodels 회귀 분석 중 오류 발생: {e}")
            return None
    
    def save_model(self, model=None, model_name=None, output_path=None):
        """
        훈련된 모델을 파일로 저장합니다.
        
        Args:
            model (object, optional): 저장할 모델. None이면 최고 성능 모델 사용
            model_name (str, optional): 모델 이름. None이면 'best_model' 사용
            output_path (str, optional): 출력 파일 경로
            
        Returns:
            str: 저장된 모델 파일 경로
        """
        if model is None:
            if self.best_model:
                model = self.best_model
                if model_name is None:
                    model_name = "best_model"
            else:
                logger.error("저장할 모델이 없습니다.")
                return None
        
        if model_name is None:
            model_name = "model"
        
        try:
            # 출력 경로 설정
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                output_path = f"{self.trained_models_dir}/{model_name}_{timestamp}.pkl"
            
            # 모델 저장
            with open(output_path, 'wb') as f:
                pickle.dump(model, f)
            
            logger.info(f"모델 저장 완료: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"모델 저장 중 오류 발생: {e}")
            return None
    
    def load_model(self, input_path):
        """
        저장된 모델을 로드합니다.
        
        Args:
            input_path (str): 모델 파일 경로
            
        Returns:
            object: 로드된 모델
        """
        try:
            with open(input_path, 'rb') as f:
                model = pickle.load(f)
            
            logger.info(f"모델 로드 완료: {input_path}")
            return model
            
        except Exception as e:
            logger.error(f"모델 로드 중 오류 발생: {e}")
            return None


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
    
    # 모델 훈련
    trainer = ModelTrainer()
    
    # 데이터 전처리
    X_train, X_test, y_train, y_test, preprocessor, feature_names = trainer.preprocess_data(data)
    
    # 모델 훈련
    models = trainer.train_models(X_train, y_train, preprocessor, feature_names)
    
    # 모델 평가
    evaluation = trainer.evaluate_models(X_test, y_test)
    print(evaluation)
    
    # 최고 성능 모델 세부 튜닝
    if evaluation is not None and not evaluation.empty:
        best_model_name = evaluation.iloc[0]['model']
        tuned_model = trainer.fine_tune_model(X_train, y_train, X_test, y_test, best_model_name)
    
    # 모델 저장
    if trainer.best_model:
        model_path = trainer.save_model()
        print(f"최고 성능 모델이 저장됨: {model_path}")
    
    # statsmodels 회귀 분석
    stats_model = trainer.get_statsmodels_regression(data)
    if stats_model:
        print(stats_model.summary())
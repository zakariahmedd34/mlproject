import sys
import os
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import evaluation_models, save_object

@dataclass
class ModelTrainerConfig:
    trained_model_path: str = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    

    def initiate_model_trainer(self,train_arr,test_arr):
        logging.info("Model Training is Starting")
        try:

            logging.info("Split the data")

            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(),
            'Lasso Regression': Lasso(),
            'KNeighbors Regressor': KNeighborsRegressor(),
            'Decision Tree Regressor': DecisionTreeRegressor(),
            'Random Forest Regressor': RandomForestRegressor(),
            'AdaBoost Regressor': AdaBoostRegressor(),
            'Support Vector Regressor': SVR(),
            'XGBoost Regressor': XGBRegressor()
            }

            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest":{
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{

                },
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            evaluation_report:dict = evaluation_models(X_train,y_train,X_test,y_test,models,params)

            best_model_name = max(evaluation_report,
                key=lambda m: (evaluation_report[m]["test_score"], -evaluation_report[m]["gap"])
            )

            best_model =  models[best_model_name]

            if evaluation_report[best_model_name]['test_score'] < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing data")

            save_object(
                file_path= self.model_trainer_config.trained_model_path,
                obj = best_model
            )


            pred = best_model.predict(X_test)

            R2 = r2_score(y_test,pred)
            return R2

        except Exception as e:
            raise CustomException(e,sys)
            
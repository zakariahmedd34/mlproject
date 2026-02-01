'''
Utility functions for the ML project.

reusable helper functions
reduces code duplication
shared across module

'''
import sys
import os 
import dill
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluation_models(X_train,y_train,X_test,y_test,models):
    report = {}
    try:
        for i in models:
            model = models[i]

            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)
            
            y_test_pred = model.predict(X_test)

            train_score = r2_score(y_train,y_train_pred)

            test_score = r2_score(y_test,y_test_pred)

            report[i] = {'test_score' : test_score,'gap':abs(train_score-test_score)}

            logging.info(f"model evalutaion report {report}")

            return report
    except Exception as e:
        raise CustomException(e,sys)
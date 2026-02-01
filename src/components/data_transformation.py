'''
file_name: data_transformation.py
created_at : 2/1/2026
'''
import os
import sys
import pandas as pd
import numpy as np
from src.utils import save_object
from src.logger import logging
from src.exception import CustomException
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from dataclasses import dataclass
@dataclass
class dataTransformationConfig:
    """
    (1..4) will be inputs to method (initiate_data_transformation) only 
    the main target of this file is the preprocessing so we store 
    only the preprossor obj and the way to store object or models
    and can use it later is by store is in pkl file and that is
    what we going to do...

    1 trans_train_data_path:  str= os.path.join('artifacts',"trans_train.csv")
    2 trans_test_data_path:   str = os.path.join('artifacts',"trans_test.csv")
    3 train_target_path:      str= os.path.join('artifacts',"train_target.csv")
    4 test_target_path:       str = os.path.join('artifacts',"test_target.csv")
    """

    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")

class dataTransformation:
    def __init__(self):
       self.data_transfor_config = dataTransformationConfig()


    @property
    def get_data_transformation_object(self):
        try:
             
            numerical_features = [
                                'reading score',
                                 'writing score'
                                 ]
            categorical_features=['gender', 
                                'race/ethnicity', 
                                'parental level of education', 
                                'lunch', 
                                'test preparation course']
            #num_pipeline
            num_pipeline = Pipeline(
                steps=[ 
                ("impute",SimpleImputer(strategy="median")),
                ("scale",StandardScaler())
                ]
            ) 
            #cat pipeline
            cat_pipeline = Pipeline(
                steps = [
                ("impute",SimpleImputer(strategy='most_frequent')),
                ("one_hot_encoder",OneHotEncoder(handle_unknown='ignore',sparse_output=False)),
                ("scale",StandardScaler())
                ]
            )
            logging.info(f"numerical features {numerical_features}, categorical features {categorical_features}")
            #columntransformer_preprossor(num_pipe,cat,pipe)
            preprocessor = ColumnTransformer([
                ('num',num_pipeline,numerical_features),
                ('cat',cat_pipeline,categorical_features)
            ])

            return preprocessor
        except Exception as e:
             raise CustomException(e,sys)
             

    def initate_data_transformation(self,train_data_path,test_data_path):
        logging.info("Entered the data transformation method or component")
        try:

            logging.info("read train dataset")
            train_set = pd.read_csv(train_data_path)
            logging.info("read test dataset")
            test_set = pd.read_csv(test_data_path)

            logging.info("Obtaining preprocessing object") 
            
            preprocessing_obj = self.get_data_transformation_object

            logging.info(f'prepare data and target train col : {train_set.columns} test col {test_set.columns}')

            y_train = train_set['math score']
            logging.info(f'y_train {y_train.shape}')

            trans_X_train = train_set.drop('math score',axis = 1)
            logging.info(f'trans_X_test {trans_X_train.columns}')

            y_test = test_set['math score']
            logging.info(f'y_test {y_test.shape}')

            trans_X_test = test_set.drop('math score',axis = 1)
            logging.info(f'trans_X_test {trans_X_test.columns}')

           
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(trans_X_train)
            input_feature_test_arr = preprocessing_obj.transform(trans_X_test)
            
            train_arr = np.c_[input_feature_train_arr,np.array(y_train)]
            test_arr  = np.c_[input_feature_test_arr,np.array(y_test)]
            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path = self.data_transfor_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transfor_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.error("error in transforming data")
            raise CustomException(e,sys)
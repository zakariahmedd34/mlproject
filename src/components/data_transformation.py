'''
file_name: data_transformation.py
created_at : 2/1/2026
'''
import os
import sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from src.components.data_ingestion import DataIngestion
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
@dataclass
class dataTransformationConfig:
    trans_train_data_path:  str= os.path.join('artifacts',"trans_train.csv")
    trans_test_data_path:   str = os.path.join('artifacts',"trans_test.csv")
    train_target_path:      str= os.path.join('artifacts',"train_target.csv")
    test_target_path:       str = os.path.join('artifacts',"test_target.csv")


def get_features_col(df):
    '''
    Args:
        df: pandas dataframe

    Return:

        numeric_features_col: list of non categorical features
        categorical_feature_col : list of categorical features
    '''
    numeric_features_col = df.select_dtypes(include=['int64', 'float64']).columns

    categorical_feature_col = df.select_dtypes(include=['object', 'category', 'string']).columns

    return (
        list(numeric_features_col),
        list(categorical_feature_col)
    )

def encoding_scale(trans_X_train,trans_X_test):
            
            ohe = OneHotEncoder(handle_unknown = 'ignore',sparse_output=False).set_output(transform= 'pandas')

            logging.info('OneHotEncoding for Categorical Features')
            cat_feat = get_features_col(trans_X_train)[1]

            ohe_train = ohe.fit_transform(trans_X_train[cat_feat])
            ohe_test =  ohe.transform(trans_X_test[cat_feat])

            trans_X_train = pd.concat([trans_X_train,ohe_train],axis = 1).drop(cat_feat,axis = 1)
            trans_X_test = pd.concat([trans_X_test,ohe_test],axis= 1).drop(cat_feat,axis = 1)

            # Scaler for numerical features
            logging.info('Scaler for Numerical Features')

            scaler = StandardScaler().set_output(transform='pandas')

            trans_X_train = scaler.fit_transform(trans_X_train)
            trans_X_test = scaler.transform(trans_X_test)

            return(trans_X_train,trans_X_test)


class dataTransformation:
    def __init__(self):
       self.data_transfor_config = dataTransformationConfig()
        
    def data_preparation(self):
        logging.info("Entered the data transformation method or component")
        try:
            res =  DataIngestion().initial_data_ingestion()
            logging.info("read train dataset")
            train_set = pd.read_csv(res[0])
            logging.info("read test dataset")
            test_set = pd.read_csv(res[1])

            logging.info('prepare data and target')
            y_train = train_set['math score']
            trans_X_train = train_set.drop('math score',axis = 1)
            y_test = test_set['math score']
            trans_X_test = test_set.drop('math score',axis = 1)

            logging.info('save target train set')
            y_train.to_csv(self.data_transfor_config.train_target_path,index = False,header= True)
            logging.info('save target test set')
            y_test.to_csv(self.data_transfor_config.test_target_path,index= False,header= True)


            logging.info('Transform data')

            trans_df = encoding_scale(trans_X_train,trans_X_test)
            trans_X_train = trans_df[0]
            trans_X_test = trans_df [1]

            logging.info('save X train set')
            trans_X_train.to_csv(self.data_transfor_config.trans_train_data_path,index = False,header= True)
            logging.info('save X test set')
            trans_X_test.to_csv(self.data_transfor_config.trans_test_data_path,index= False,header= True)

            return(trans_X_train,trans_X_test,y_train,y_test)
        except Exception as e:
            logging.error("error in transforming data")
            raise CustomException(e,sys)

if __name__ == "__main__":
    obj = dataTransformation()
    obj.data_preparation()
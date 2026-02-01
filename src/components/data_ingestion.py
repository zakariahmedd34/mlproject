"""
Data Ingestion module.

This module is responsible for collecting data from the source location,
loading it into the project, and preparing the initial dataset artifacts.
It handles reading raw data files, performing the train/test split, and
saving the resulting datasets for downstream processing and model training.

This stage represents the entry point of the ML pipeline.
"""
import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import dataTransformation,dataTransformationConfig
from src.components.model_trainer import ModelTrainer
@dataclass
class DataIngestionConfig:
    train_data_path:  str= os.path.join('artifacts',"train.csv")
    test_data_path:   str = os.path.join('artifacts',"test.csv")
    raw_data_path:    str = os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config =DataIngestionConfig()

    def initial_data_ingestion(self)->tuple[str,str]:

        """
        Raw Data
            ↓
        Read CSV
            ↓
        Save raw artifact
            ↓
        Train/Test Split
            ↓
        Save artifacts
            ↓
        Return paths
        
        """
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv(r'notebook\data\StudentsPerformance.csv')

            logging.info('Read the dataset as data frame')
            
            # make Artifact directory
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            # save raw data before split
            df.to_csv(self.ingestion_config.raw_data_path,index = False,header = True)

            logging.info("Train test splite initated")
            train_set,test_set = train_test_split(df,test_size = 0.3,random_state=42)

            # save train set
            train_set.to_csv(self.ingestion_config.train_data_path,index = False,header = True)

            # save test set
            test_set.to_csv(self.ingestion_config.test_data_path,index = False,header = True)

            logging.info("Ingestion of data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:

            raise CustomException(e,sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_set,test_set = obj.initial_data_ingestion()
    trans_obj = dataTransformation()
    train_arr,test_arr,_ = trans_obj.initate_data_transformation(train_set,test_set)

    modeltrainer = ModelTrainer()
    r2 = modeltrainer.initiate_model_trainer(train_arr,test_arr)
    print(f"r2 score of the best model is {r2}")

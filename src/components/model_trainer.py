import sys
import os
import pandas as pd
import numpy as np

from dataclasses import dataclass

from sklearn.linear_model import LinearRegression ,Lasso,Ridge
from sklearn.ensemble import (AdaBoostRegressor ,GradientBoostingRegressor, RandomForestRegressor)
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

# Metrice
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_models

## Now as usual for every component we have to create config class in which we give path
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        # so inside this variable we get a path

    # Now we know we have get train and test array from data_transformation
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('Spitting training and test input data')
            
            ## Now we know we have train array and test array in which last columns is target
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
            'LinearRegression':LinearRegression(),
            'Lasso':Lasso(),
            'Ridge':Ridge(),
            'DecisionTree':DecisionTreeRegressor(),
            'Random_forest':RandomForestRegressor(),
            'Gradient_Boosting':GradientBoostingRegressor(),
            'K Neighbours Regressor':KNeighborsRegressor(),
            'XGB Regressor':XGBRegressor(),
            'Adaboost Regressor':AdaBoostRegressor()
            }
            ## Evaluate_model written in utils
            model_report:dict = evaluate_models(X_train,y_train,X_test,y_test,models)


            # so to get best model score from dictionary
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dictionay
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            # best model
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException('No best model found')
            
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_score_value = r2_score(y_test,predicted)
            return r2_score_value

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)
        
## Now to check wheteher it is running properly

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

if __name__ == '__main__':
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_array,test_array,preprocessor_path = data_transformation.initiate_data_transformation(train_data,test_data)
    
    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_array,test_array))

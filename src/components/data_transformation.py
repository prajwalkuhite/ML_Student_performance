## Here we do data transformation -- data cleaning , feature scaling, feature engineering


import sys
import os
import numpy as np
import pandas as pd

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer   ## to create pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder , StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

## similarly like data_ingestion we have to create config to store the output gain from 
## this component

@dataclass
class DataTransformationConfig :
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self) :
        self.data_transformation_config = DataTransformationConfig()
    # we initialise this DataTransformationConfig (preprocessor_path) in a variable

    # Now we have to create a function to converting our categorical to numerical , standardscaler
    # pipeline ,onehotencoding etc.

    def get_data_transformer_object(self):
        try:
            numerical_columns = ['writing_score','reading_score']
            categorical_columns = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
                ]
            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ("scaler",StandardScaler())
                ])

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ])
            
            logging.info(f"Categorical columns :{categorical_columns}")
            logging.info(f"Numerical columns :{numerical_columns}")

            preprocessor=ColumnTransformer(
                [
            ('num_pipeline',num_pipeline,numerical_columns),
            ('cat_pipeline',cat_pipeline,categorical_columns)
            ]
            )

            return preprocessor
            
            
        
        except Exception as e:
            logging.info('Error in data transformation')
            raise CustomException(e,sys)
        
        ## Now we successfully done our data transformation

## Now we are going to initiate our data_transformation for that we know we have return
## test and train data or data path from data ingestion component

    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f"Train Dataframe Head : /n{train_df.head().to_string()}")
            logging.info(f"Test Dataframe Head : /n{test_df.head().to_string()}")

            logging.info('Obtaining preprocessing object')
            
            ## Calling our method inside an object
            preprocessor_object = self.get_data_transformer_object()

            target_column_name = 'math_score'
            

            ## features into independent and dependent features
            input_feature_train_df = train_df.drop(columns=target_column_name,axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=target_column_name,axis=1)
            target_feature_test_df = test_df[target_column_name]

            ## Now we just need to call our data_transformer object so we apply transformation
            input_feature_train_arr = preprocessor_object.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_object.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")

            ## Now we concatenate the inputs array with targets because we want them into single array
            ## Because from this we get last column a target columns so in model training we split it again
            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            ## we save object by calling it from utils
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_object

            )
            
            logging.info('Processor is completed and saved') 


            return(train_arr
                   ,test_arr
                   ,self.data_transformation_config.preprocessor_obj_file_path)
        
        except Exception as e:
            logging.info("Exception occured in the initiate_dataTransformation")
            raise CustomException(e,sys)

## Now to check wheteher it is running properly
'''
from src.components.data_ingestion import DataIngestion

if __name__ == '__main__':
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)

    '''


        
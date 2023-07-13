## so our main aim is to read the the data from any data source 

## 1 . we try to get data from local itself

import os 
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split

from dataclasses import dataclass


# whenever we made a data ingestion component we have to make a separate folder
# in which we store the outputs gain from this component like train_data, test_data etc.


## By using dataclasses we have not to write __init__ to define our class variable
#  we directly write this

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts','train.csv')
    test_data_path: str = os.path.join('artifacts','test.csv')
    raw_data_path: str = os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()  

    ## so we craeted this variable and inside all three paths are stored
    
    def initiate_data_ingestion(self):
        ## here we read the dataset from the database sources
        ## intially we read this data from our local
        logging.info('Entered the data ingestion method and process is initialized')
        
        try:
            df = pd.read_csv('Notebooks\data\stud.csv')
            logging.info('Read the dataset as a dataframe')


            ## Now we know above the path of our raw data in artifacts folder but
            ## we neded to create a directory(artifacts) for which inside our raw data will stored.
            ## So we call makedirs funt and give path of raw and also we give directory name 
            ## and exist_ok=True means  argument ensures that 
            ##  if the directory already exists, it doesn't raise an error.

            # matlb mere paas path h ab mujhe location or maal hona path banane ke liye

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            

            # logging.info('Raw data is stored in artifacts folder')

            ## Now we start train test split
            logging.info('Train test Split initiated')
            train_set, test_set = train_test_split(df,test_size=0.3,random_state=42)

            ## similary we also give train and test data to their respective paths
            ## Here we don't need to create directory(artifacts) because above it is already created

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of the data is completed')


            return(
                self.ingestion_config.train_data_path,    
                self.ingestion_config.test_data_path      
            )                                             
         ## we know inside this our train and test data is stored and This 2 (train and test) we use in data transformation

        
        except Exception as e:
            logging.info('Error occured in Data Ingestion config')
            raise CustomException(e,sys)
    

if __name__ == '__main__':
    obj = DataIngestion()
    obj.initiate_data_ingestion()
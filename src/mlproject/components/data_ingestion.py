import pandas as pd
import numpy as np
import re
import sys

import nltk
#nltk.download('all')

from dataclasses import dataclass
import os
from sklearn.model_selection import train_test_split

from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.config.configclass import DataIngestionConfig


class DataIngestion:
    def __init__(self):

        try:
            self.data_ingestion_config = DataIngestionConfig()
        except Exception as e:
            raise CustomException(e, sys)

    def download_Data(self):

        try:
            df = pd.read_csv('https://raw.githubusercontent.com/Vishnuu011/datastore/refs/heads/main/Twitter_Data.csv')

            df.reset_index(drop=True, inplace=True)

            df.drop(columns=df.columns[[0,2]], axis=1, inplace=True)
            df.columns = ['text','sentiment']
            df.drop([0], axis=0, inplace=True)
            #df.sentiment = df.sentiment.map({"neutral":0,"positive":1,"negative":2})
            df.dropna(inplace=True)

            return df

        except Exception as e:
            raise CustomException(e, sys)

    def intiate_data_ingestion(self):

        try:
            df = self.download_Data()

            logging.info("Data download ..............")

            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)

            df.to_csv(self.data_ingestion_config.raw_data_path, index=False)

            train, test = train_test_split(df, test_size=0.1)
            train.to_csv(self.data_ingestion_config.train_data_path, index=False)
            test.to_csv(self.data_ingestion_config.test_data_path, index=False)
            logging.info("Data Ingestion Completed ..............")
            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)



import os
import numpy as np
import pandas as pd
import sys

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import torch

from src.mlproject.config.configclass import DataTransformationConfig
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.utils import save_object
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize




class DataTratanformation:

    def __init__(self):
        try:
            self.data_transformation_config = DataTransformationConfig()
        except Exception as e:
            raise CustomException(e, sys)

    def preprocess(self, df):
        try:
            df['text'] = df['text'].astype(str).apply(self._preprocess_text)  
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def _preprocess_text(self, text): 
        try:
            text = re.sub(r"http\\S+", "", text)
            text = re.sub(r"@\\w+", "", text)
            text = re.sub(r"#", "", text)
            text = re.sub(r"[^\w\s]", "", text)
            text = str(text).lower()
            text = text.split()
            lemmatizer = WordNetLemmatizer()
            clean_text = [lemmatizer.lemmatize(word) for word in text if not word in set(stopwords.words('english'))]
            clean_text = ' '.join(clean_text)
            return clean_text
        except Exception as e:
            raise CustomException(e, sys)

    def fit_transformation_tensor(self, train, test):

        try:
            train_data = pd.read_csv(train)
            test_data = pd.read_csv(test)

            # Preprocess before splitting
            train_data = self.preprocess(train_data)
            test_data = self.preprocess(test_data)

            le = LabelEncoder()
            train_data['sentiment'] = le.fit_transform(train_data['sentiment'])
            test_data['sentiment'] = le.transform(test_data['sentiment'])
            print(train_data.sentiment.value_counts(normalize=True)*100)
            save_object(
                self.data_transformation_config.le_encoder_path,
                le
            )

            # Split the data after preprocessing
            X_train, X_test, y_train, y_test = train_test_split(train_data['text'], train_data['sentiment'])
            vectorizer = TfidfVectorizer(max_features=5000)
            X_train_vectorized = vectorizer.fit_transform(X_train)
            X_test_vectorized = vectorizer.transform(X_test)
            X_train_vectorized = X_train_vectorized.toarray()
            X_test_vectorized = X_test_vectorized.toarray()

            save_object(
                self.data_transformation_config.vectorizer_path,
                vectorizer
            )

            X_train_tensor = torch.tensor(X_train_vectorized, dtype=torch.float32)
            X_test_tensor = torch.tensor(X_test_vectorized, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
            y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

            return (
                X_train_tensor,
                X_test_tensor,
                y_train_tensor,
                y_test_tensor
            )
        except Exception as e:
            raise CustomException(e, sys)
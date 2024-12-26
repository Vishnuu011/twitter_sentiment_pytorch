import os
import sys
import torch
import re

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from src.mlproject.utils import load_object



class PredictionPipeline:
    def __init__(self):

        pass

    @staticmethod
    def preprocess_text(text):
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

    def predict(self, text):

        try:
           
            vectorizer = os.path.join("artifact", "vectorizer.pkl")
            vectorizer = load_object(vectorizer)

            model_path = os.path.join("artifact", "model.pkl")
            model_1 = load_object(model_path)

            text_clean = PredictionPipeline.preprocess_text(text)
            text_vectorized = vectorizer.transform([text_clean]).toarray()
            text_tensor = torch.FloatTensor(text_vectorized)
            model_1.eval()
            with torch.no_grad():
                output = model_1(text_tensor)
                predicted = torch.argmax(output.data, 1)

            return predicted

        except Exception as e:
            raise CustomException(e, sys)
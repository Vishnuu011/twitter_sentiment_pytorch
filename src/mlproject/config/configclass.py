from dataclasses import dataclass
import os

class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifact", "raw.csv")
    train_data_path: str = os.path.join("artifact", "train.csv")
    test_data_path: str = os.path.join("artifact", "test.csv")


class DataTransformationConfig:
    vectorizer_path: str = os.path.join("artifact", "vectorizer.pkl")
    le_encoder_path: str = os.path.join("artifact", "le_encoder.pkl")    


class ModelTrainerConfig:
    model_path: str = os.path.join("artifact", "model.pkl")    
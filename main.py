from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_transformation import DataTratanformation
from src.mlproject.LSTM_model.lstm_model import LSTMModel
from src.mlproject.components.model_trainer import ModelTrainer
import numpy as np

data_ingestion = DataIngestion()
train, test = data_ingestion.intiate_data_ingestion()

data_tansformation = DataTratanformation()
xtrain_tensor, xtest_tensor, ytrain_tensor_, ytest_tensor_ = data_tansformation.fit_transformation_tensor(train, test)

print(xtrain_tensor.shape, xtest_tensor.shape, ytrain_tensor_.shape, ytest_tensor_.shape)

input_size = xtrain_tensor.shape[1] 
hidden_size = 64                        
num_classes = len(np.unique(ytrain_tensor_))  

# Instantiate the model
model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)

model_trainer = ModelTrainer(model, xtrain_tensor,
                             ytrain_tensor_,
                             xtest_tensor,
                             ytest_tensor_
                            )
model_trainer.fit_trainer()

model_trainer.fit_evaluation()
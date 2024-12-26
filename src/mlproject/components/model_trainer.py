import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.mlproject.config.configclass import ModelTrainerConfig
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.utils import save_object




class ModelTrainer:
    def __init__(self, model, X_train, y_train, X_test, y_test):
        try:
            self.model_trainer_config = ModelTrainerConfig()
            self.model = model
            self.X_train = X_train
            self.y_train = y_train
            self.X_test = X_test
            self.y_test = y_test
        except Exception as e:
            raise CustomException(e, sys)


    def fit_trainer(self):
        try:
            self._train_model()
            save_object(
                self.model_trainer_config.model_path,
                self.model
            )
        except Exception as e:
            raise CustomException(e, sys)


    def fit_evaluation(self):

        try:
            self._model_evaluation()
        except Exception as e:
            raise CustomException(e, sys)

    def _train_model(self):

        try:
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            epochs = 5
            for epoch in range(epochs):
                self.model.train()

                running_loss = 0.0

                for i in tqdm(range(0, len(self.X_train), 64)):
                    X_batch = self.X_train[i:i + 64]
                    y_batch = self.y_train[i:i + 64]

                    optimizer.zero_grad()
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    accuracy = (outputs.argmax(dim=1) == y_batch).float().mean()
                    optimizer.step()
                    running_loss += loss.item()
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(self.X_train):.4f}, Accuracy: {accuracy:.4f}')

        except Exception as e:
            raise CustomException(e, sys)

    def _model_evaluation(self):

        try:
            self.model.eval()

            with torch.no_grad():
                outputs = self.model(self.X_test)
                _, predicted = torch.max(outputs.data, 1)

            accuracy = (predicted == self.y_test).sum().item() / len(self.y_test) * 100
            print(f'Test Accuracy: {accuracy:.2f}%')

        except Exception as e:
            raise CustomException(e, sys)
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_transformation import DataTratanformation

data_ingestion = DataIngestion()
train, test = data_ingestion.intiate_data_ingestion()

data_tansformation = DataTratanformation()
xtrain_tensor, xtest_tensor, ytrain_tensor_, ytest_tensor_ = data_tansformation.fit_transformation_tensor(train, test)

print(xtrain_tensor.shape, xtest_tensor.shape, ytrain_tensor_.shape, ytest_tensor_.shape)
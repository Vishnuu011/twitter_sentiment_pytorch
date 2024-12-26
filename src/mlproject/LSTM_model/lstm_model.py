import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm






class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        x = x.unsqueeze(1)

        out, (hn, cn) = self.lstm(x)

        out = self.fc1(hn[-1])

        out = self.leaky_relu(out)

        out = self.fc2(out)

        return out


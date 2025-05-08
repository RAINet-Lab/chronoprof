import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np


class Model(nn.Module):
    def __init__(self,configs):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, configs.pred_len)
    def forward(self, x):
        x, (hn,cn) = self.lstm(x)
        x=x[:,-1,:].squeeze()
        x = self.linear(x)
        return x.unsqueeze(dim=-1)
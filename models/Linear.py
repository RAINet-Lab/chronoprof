__all__= ['Linear']
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    """
    Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.linear= nn.Linear(self.seq_len,self.pred_len)

    def forward(self, x):
        return self.linear(x.permute(0,2,1)).permute(0,2,1)

import torch
import torch.nn.functional as F
from torch import nn


class FFN(nn.Module):
    def __init__(self, d_model: int = 512) -> None:
        super(FFN, self).__init__()

        self.d_model = d_model

        self.linear_1 = nn.Linear(self.d_model, 2048)
        self.linear_2 = nn.Linear(2048, self.d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        out = self.relu(self.linear_1(x))
        out = self.linear_2(out)
        out = self.dropout(out)
        return out

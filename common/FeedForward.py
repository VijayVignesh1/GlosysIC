import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Feed Forward net work is used for traversing the inputs w.r.t hidden layers and is just a straight forward network
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__() 
    
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        # Whatever Input is it just returns back the same
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

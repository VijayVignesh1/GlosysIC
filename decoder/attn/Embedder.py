import torch
import torch.nn as nn

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        # Input is ( batch size, seqlength)
        # Output is ( batch size, seqlength, d_model)
        return self.embed(x)
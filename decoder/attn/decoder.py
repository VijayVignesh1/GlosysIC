import torch
import torch.nn as nn 
import copy
from decoder.attn.MultiHeadAttention import MultiHeadAttention
from decoder.attn.PositionalEncoder import PositionalEncoder
from decoder.attn.Embedder import Embedder
from common.FeedForward import FeedForward
from common.norm import Norm

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N = 6, heads = 8, dropout=0.5):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
        self.out=nn.Linear(d_model,vocab_size)
    def forward(self, trg, e_outputs, trg_mask,caption_lengths):
        caption_lengths, sort_ind=caption_lengths.squeeze(1).sort(dim=0, descending=True)
        trg=trg[sort_ind]
        e_outputs=e_outputs[sort_ind]
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, trg_mask)
        y = self.norm(x)
        return self.out(y), trg ,caption_lengths, sort_ind

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.5):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    # Used for forward traversal from the decoder and the encoders final output
    def forward(self, x, e_outputs, trg_mask):
        # Input is (batch, seq length)
        # Output is (batch, seq length, d_model)
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x
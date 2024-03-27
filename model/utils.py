import torch
import torch.nn.functional as F
from torch import nn

from model.attention import MultiHeadAttentionLayer
from model.moe import FFN, Router


class LayerNorm(nn.Module):
    def __init__(self, ndims, bias) -> None:
        super(LayerNorm, self).__init__()

        self.weight = nn.Parameter(torch.ones(ndims))
        self.bias = nn.Parameter(torch.zeros(ndims)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class Block(nn.Module):
    def __init__(self, d_model, h, is_moe) -> None:
        super(Block, self).__init__()
        self.d_model = d_model
        self.h = h

        self.ln_1 = LayerNorm(self.d_model, bias=True)
        self.attn = MultiHeadAttentionLayer(self.d_model, self.h)
        self.ln_2 = LayerNorm(self.d_model, bias=True)
        if is_moe:
            self.ffn = Router(8, 2, d_model)
        else:
            self.ffn = FFN(self.d_model)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))[0]
        x = x + self.ffn(self.ln_2(x))
        return x

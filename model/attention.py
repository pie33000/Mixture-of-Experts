import torch
import torch.nn.functional as F
from torch import nn


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model: int = 512, h: int = 64):
        super(MultiHeadAttentionLayer, self).__init__()
        self.h = h
        self.d_model = d_model
        self.d_k = self.d_model // self.h
        self.d_v = self.d_model // self.h

        assert self.d_k > 0

        self.proj_attn = nn.Linear(self.d_model, 3 * self.d_model, bias=True)
        self.linear_output = nn.Linear(self.d_model, self.d_model, bias=True)
        self.residual_dropout = nn.Dropout(p=0.1)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        """
        :param query: (batch_size, query_len, d_model)
        :param key: (batch_size, key_len, d_model)
        :param value: (batch_size, key_len, d_model)
        :param mask: (batch_size, query_len, key_len)
        :return
        """
        assert x.dim() == 3
        B, T, C = x.size()

        mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T)

        q, k, v = self.proj_attn(x).split(self.d_model, dim=2)
        k = k.view(B, T, self.h, C // self.h).transpose(1, 2)  # (B, n*h, T, h*C)
        q = q.view(B, T, self.h, C // self.h).transpose(1, 2)
        v = v.view(B, T, self.h, C // self.h).transpose(1, 2)

        out, prob = self.attention(q, k, v, mask, T)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.residual_dropout(out)
        return out, prob

    def attention(self, query, key, value, mask=None, T=None):
        d_k = key.size(-1)
        scores = query.matmul(key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k))

        scores = scores.masked_fill(mask[:, :, :T, :T] == 0, float("-inf"))

        scores_proba = F.softmax(scores, dim=-1)
        scores_proba = self.dropout(scores_proba)
        out = scores_proba.matmul(value)

        return out, scores_proba

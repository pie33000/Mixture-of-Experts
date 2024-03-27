import math

import torch
import torch.nn.functional as F
from torch import nn

from model.utils import Block, LayerNorm


class DecoderOnly(nn.Module):
    def __init__(
        self, d_model, h, vocab_size, block_size, is_moe: bool = False
    ) -> None:
        super().__init__()

        self.block_size = block_size

        self.decoder = nn.ModuleDict(
            dict(
                wte=nn.Embedding(vocab_size, d_model),
                wpe=nn.Embedding(block_size, d_model),
                drop=nn.Dropout(p=0.1),
                h=nn.ModuleList(
                    [Block(d_model, h, is_moe) for _ in range(d_model // h)]
                ),
                ln_f=LayerNorm(d_model, bias=True),
            )
        )

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.decoder.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

        self.apply(self._init_weights)

        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("linear_output.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * d_model // h)
                )

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.decoder.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)
        # forward the GPT model itself
        tok_emb = self.decoder.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.decoder.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.decoder.drop(tok_emb + pos_emb)
        for block in self.decoder.h:
            x = block(x)
        x = self.decoder.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

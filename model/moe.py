import torch
import torch.nn as nn
import torch.nn.functional as F


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


class Router(nn.Module):
    def __init__(
        self,
        num_experts: int,
        num_expert_per_token: int,
        d_model: int,
    ) -> None:
        super().__init__()
        self.experts = [FFN(d_model) for _ in range(num_experts)]
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.num_experts = num_experts
        self.num_expert_per_token = num_expert_per_token

    def forward(self, x: torch.tensor):
        # x (B, T, C)
        x_squashed = x.view(-1, x.shape[-1])
        gate_logits = self.gate(x_squashed)  # (B * T, num_experts)
        weights, selected_experts = torch.topk(
            gate_logits, self.num_expert_per_token
        )  # values (B * T, num_expert_per_token), indices (B * T, num_expert_per_token)
        weights = F.softmax(weights, dim=1, dtype=torch.float)
        results = torch.zeros_like(x_squashed)
        for i, expert in enumerate(self.experts):
            B, nth_expert = torch.where(selected_experts == i)
            results[B] += weights[B, nth_expert, None] * expert(x_squashed[B])
        return results.view_as(x)

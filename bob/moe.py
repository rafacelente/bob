from typing import List
import torch.nn as nn
import torch
import torch.nn.functional as F

class MoeLayer(nn.Module):
    """
    A mixture of experts layer that selects a subset of experts based on a gate tensor.
    Based on the Mixtral architecture:
      - https://github.com/mistralai/mistral-src/blob/main/mistral/moe.py

    Args:
        experts (List[nn.Module]): List of expert modules.
        gate (nn.Module): Gate module.
        num_experts (int): Number of experts.
        num_experts_per_token (int): Number of experts to select per token.
    """
    def __init__(
            self,
            experts: List[nn.Module],
            gate: nn.Module,
            num_experts: int,
            num_experts_per_token: int,
    ):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token

    def forward(self, x):
        gate = self.gate(x)
        weights, selected_experts = torch.topk(gate, self.num_experts_per_token)
        weights = F.softmax(weights, dim=-1, dtype=torch.float).to(x.dtype)
        output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            output[batch_idx] = weights[batch_idx, nth_expert, None] * expert(x[batch_idx])
        return output

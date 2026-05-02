"""Dueling DQN — compact trunk (128) matching Traffic-Control-RL / sumo-rl obs scale."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingDQN(nn.Module):
    """Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)."""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        self.input = nn.Linear(state_size, hidden_size)
        self.main = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
        )
        self.value = nn.Linear(hidden_size, 1)
        self.advantage = nn.Linear(hidden_size, action_size)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)
        h = F.relu(self.input(x))
        h = self.main(h)
        v = self.value(h)
        a = self.advantage(h)
        q = v + a - a.mean(dim=1, keepdim=True)
        return q.squeeze(0) if single else q

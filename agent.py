"""
Double Dueling DQN with uniform replay (sumo-rl Phase A).
Follows the simple update in Traffic-Control-RL (CodeKnight314) + soft target updates.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from model import DuelingDQN
from replay import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        max_memory: int = 100_000,
        max_grad: float = 1.0,
        hidden_size: int = 128,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.max_grad = max_grad
        self.action_size = action_size

        self.online = DuelingDQN(state_size, action_size, hidden_size).to(self.device)
        self.target = DuelingDQN(state_size, action_size, hidden_size).to(self.device)
        self.target.load_state_dict(self.online.state_dict())

        self.buffer = ReplayBuffer(max_memory)
        self.optim = AdamW(self.online.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

    def act(self, state: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            s = torch.FloatTensor(state).to(self.device)
            q = self.online(s)
            return int(q.argmax().item())

    def push(self, *args: Any) -> None:
        self.buffer.push(*args)

    def learn(self, batch_size: int) -> float | None:
        if len(self.buffer) < batch_size:
            return None
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        with torch.no_grad():
            next_actions = self.online(next_states).argmax(1, keepdim=True)
            next_q = self.target(next_states).gather(1, next_actions)
            targets = rewards + (1.0 - dones) * self.gamma * next_q

        current = self.online(states).gather(1, actions.unsqueeze(1))
        loss = self.loss_fn(current, targets)

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.max_grad)
        self.optim.step()
        return float(loss.item())

    def soft_update_target(self, tau: float) -> None:
        with torch.no_grad():
            for t, o in zip(self.target.parameters(), self.online.parameters()):
                t.data.mul_(1.0 - tau).add_(tau * o.data)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
            },
            path,
        )

    def load(self, path: str | Path) -> bool:
        path = Path(path)
        if not path.is_file():
            return False
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])
        return True

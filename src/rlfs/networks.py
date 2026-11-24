"""Neural network modules: MLP, Q-networks (plain + dueling), actor-critic."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Categorical


def mlp(
    in_dim: int, out_dim: int, hidden: tuple[int, ...] = (64, 64), act=nn.Tanh
) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), act()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


def orthogonal_init(module: nn.Module, gain: float = math.sqrt(2)) -> None:
    """Orthogonal weights / zero bias — the standard PPO initialisation."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain)
            nn.init.zeros_(m.bias)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden=(128, 128), dueling: bool = False):
        super().__init__()
        self.dueling = dueling
        if dueling:
            self.trunk = mlp(obs_dim, hidden[-1], hidden[:-1], act=nn.ReLU)
            self.value = nn.Linear(hidden[-1], 1)
            self.adv = nn.Linear(hidden[-1], n_actions)
        else:
            self.net = mlp(obs_dim, n_actions, hidden, act=nn.ReLU)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if not self.dueling:
            return self.net(obs)
        h = torch.relu(self.trunk(obs))
        a = self.adv(h)
        # Mean-subtracted advantage keeps V and A identifiable (Wang et al., 2016).
        return self.value(h) + a - a.mean(dim=-1, keepdim=True)


class ActorCritic(nn.Module):
    """Separate policy and value MLPs for discrete action spaces."""

    def __init__(self, obs_dim: int, n_actions: int, hidden=(64, 64)):
        super().__init__()
        self.actor = mlp(obs_dim, n_actions, hidden)
        self.critic = mlp(obs_dim, 1, hidden)
        orthogonal_init(self)
        # Small final-layer gains: near-uniform initial policy, near-zero initial value.
        nn.init.orthogonal_(self.actor[-1].weight, 0.01)
        nn.init.orthogonal_(self.critic[-1].weight, 1.0)

    def dist(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.actor(obs))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def act(self, obs: torch.Tensor):
        d = self.dist(obs)
        a = d.sample()
        return a, d.log_prob(a), self.value(obs)

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        d = self.dist(obs)
        return d.log_prob(actions), d.entropy(), self.value(obs)

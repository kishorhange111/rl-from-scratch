"""Shared utilities: seeding, evaluation, training history."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class History:
    """Per-episode training returns indexed by the env step at which they finished."""

    steps: list[int] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)

    def log_episode(self, step: int, ret: float) -> None:
        self.steps.append(int(step))
        self.returns.append(float(ret))

    def recent_mean(self, k: int = 20) -> float:
        if not self.returns:
            return float("nan")
        return float(np.mean(self.returns[-k:]))

    def to_dict(self) -> dict:
        return {"steps": self.steps, "returns": self.returns}


@torch.no_grad()
def evaluate(
    policy: Callable[[np.ndarray], int],
    env_id: str,
    episodes: int = 10,
    seed: int = 10_000,
) -> tuple[float, float]:
    """Run a deterministic policy; returns (mean, std) of undiscounted episode return.

    Evaluation seeds are disjoint from training seeds so we never score on
    training initial states.
    """
    env = gym.make(env_id)
    rets = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, total = False, 0.0
        while not done:
            obs, r, term, trunc, _ = env.step(policy(obs))
            total += float(r)
            done = term or trunc
        rets.append(total)
    env.close()
    return float(np.mean(rets)), float(np.std(rets))

"""Experience storage: uniform replay (off-policy) and rollout buffer with GAE (on-policy)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: np.ndarray,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalized Advantage Estimation (Schulman et al., 2016).

    Arrays are shaped ``[T, N]`` (time, envs). ``dones[t]`` marks that the episode
    ended *after* step ``t`` so the bootstrap from ``t+1`` is cut.

    Returns ``(advantages, returns)`` where ``returns = advantages + values``.
    """
    T = rewards.shape[0]
    adv = np.zeros_like(rewards, dtype=np.float64)
    last_gae = np.zeros_like(last_value, dtype=np.float64)
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        adv[t] = last_gae
    adv = adv.astype(np.float32)
    return adv, (adv + values).astype(np.float32)


class ReplayBuffer:
    """Fixed-size circular buffer with uniform sampling."""

    def __init__(self, capacity: int, obs_dim: int, seed: int = 0) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(self, obs, action, reward, next_obs, done) -> None:
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        idx = self.rng.integers(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[idx]),
            "actions": torch.as_tensor(self.actions[idx]),
            "rewards": torch.as_tensor(self.rewards[idx]),
            "next_obs": torch.as_tensor(self.next_obs[idx]),
            "dones": torch.as_tensor(self.dones[idx]),
        }


@dataclass
class RolloutBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor


class RolloutBuffer:
    """Stores ``T x N`` on-policy transitions from ``N`` vectorized envs."""

    def __init__(self, n_steps: int, n_envs: int, obs_dim: int) -> None:
        self.n_steps, self.n_envs = n_steps, n_envs
        self.obs = np.zeros((n_steps, n_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_envs), dtype=np.int64)
        self.log_probs = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.t = 0

    def add(self, obs, actions, log_probs, rewards, dones, values) -> None:
        t = self.t
        self.obs[t], self.actions[t], self.log_probs[t] = obs, actions, log_probs
        self.rewards[t], self.dones[t], self.values[t] = rewards, dones, values
        self.t += 1

    @property
    def full(self) -> bool:
        return self.t >= self.n_steps

    def finalize(self, last_value: np.ndarray, gamma: float, lam: float) -> RolloutBatch:
        adv, ret = compute_gae(self.rewards, self.values, self.dones, last_value, gamma, lam)
        flat = lambda a: torch.as_tensor(a.reshape(self.n_steps * self.n_envs, *a.shape[2:]))  # noqa: E731
        self.t = 0
        return RolloutBatch(
            obs=flat(self.obs),
            actions=flat(self.actions),
            log_probs=flat(self.log_probs),
            advantages=flat(adv),
            returns=flat(ret),
            values=flat(self.values),
        )

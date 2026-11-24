"""Deep Q-Network with Double-DQN targets, optional dueling head, and Polyak target updates.

References:
    Mnih et al., 2015 — Human-level control through deep RL (DQN).
    van Hasselt et al., 2016 — Deep RL with Double Q-learning.
    Wang et al., 2016 — Dueling network architectures.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from .buffers import ReplayBuffer
from .common import History, set_seed
from .networks import QNetwork


@dataclass
class DQNConfig:
    env_id: str = "CartPole-v1"
    total_steps: int = 50_000
    buffer_size: int = 50_000
    batch_size: int = 128
    gamma: float = 0.99
    lr: float = 5e-4
    learning_starts: int = 1_000
    train_freq: int = 1
    gradient_steps: int = 1
    tau: float = 0.005  # Polyak coefficient; 1.0 == hard copy
    target_update_every: int = 1
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_fraction: float = 0.2
    double: bool = True
    dueling: bool = False
    hidden: tuple[int, ...] = (128, 128)
    max_grad_norm: float = 10.0
    seed: int = 0


def epsilon_at(step: int, cfg: DQNConfig) -> float:
    """Linear decay from eps_start to eps_end over eps_fraction * total_steps."""
    horizon = max(1, int(cfg.eps_fraction * cfg.total_steps))
    frac = min(1.0, step / horizon)
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)


def td_target(
    q_online: QNetwork,
    q_target: QNetwork,
    rewards: torch.Tensor,
    next_obs: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    double: bool,
) -> torch.Tensor:
    """One-step TD target. Double DQN selects with the online net, evaluates with target."""
    with torch.no_grad():
        q_next_target = q_target(next_obs)
        if double:
            a_star = q_online(next_obs).argmax(dim=-1, keepdim=True)
            q_next = q_next_target.gather(-1, a_star).squeeze(-1)
        else:
            q_next = q_next_target.max(dim=-1).values
        return rewards + gamma * (1.0 - dones) * q_next


@torch.no_grad()
def polyak_update(target: torch.nn.Module, online: torch.nn.Module, tau: float) -> None:
    for tp, op in zip(target.parameters(), online.parameters(), strict=True):
        tp.mul_(1.0 - tau).add_(op, alpha=tau)


class DQNAgent:
    def __init__(self, cfg: DQNConfig, obs_dim: int, n_actions: int):
        self.cfg = cfg
        self.n_actions = n_actions
        self.q = QNetwork(obs_dim, n_actions, cfg.hidden, dueling=cfg.dueling)
        self.q_target = copy.deepcopy(self.q).requires_grad_(False)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.rng = np.random.default_rng(cfg.seed)

    @torch.no_grad()
    def greedy(self, obs: np.ndarray) -> int:
        return int(self.q(torch.as_tensor(obs, dtype=torch.float32)[None]).argmax())

    def act(self, obs: np.ndarray, eps: float) -> int:
        if self.rng.random() < eps:
            return int(self.rng.integers(self.n_actions))
        return self.greedy(obs)

    def update(self, batch: dict[str, torch.Tensor]) -> float:
        target = td_target(
            self.q,
            self.q_target,
            batch["rewards"],
            batch["next_obs"],
            batch["dones"],
            self.cfg.gamma,
            self.cfg.double,
        )
        q_sa = self.q(batch["obs"]).gather(-1, batch["actions"][:, None]).squeeze(-1)
        loss = F.smooth_l1_loss(q_sa, target)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.max_grad_norm)
        self.opt.step()
        return float(loss)


def train_dqn(cfg: DQNConfig) -> tuple[DQNAgent, History]:
    set_seed(cfg.seed)
    env = gym.make(cfg.env_id)
    obs_dim = env.observation_space.shape[0]
    agent = DQNAgent(cfg, obs_dim, env.action_space.n)
    buf = ReplayBuffer(cfg.buffer_size, obs_dim, seed=cfg.seed)
    hist = History()

    obs, _ = env.reset(seed=cfg.seed)
    ep_ret = 0.0
    for step in range(1, cfg.total_steps + 1):
        action = agent.act(obs, epsilon_at(step, cfg))
        next_obs, r, term, trunc, _ = env.step(action)
        # Only true termination cuts the bootstrap; time-limit truncation does not.
        buf.add(obs, action, r, next_obs, term)
        ep_ret += float(r)
        obs = next_obs
        if term or trunc:
            hist.log_episode(step, ep_ret)
            obs, _ = env.reset()
            ep_ret = 0.0

        if step >= cfg.learning_starts and step % cfg.train_freq == 0:
            for _ in range(cfg.gradient_steps):
                hist.losses.append(agent.update(buf.sample(cfg.batch_size)))
            if step % cfg.target_update_every == 0:
                polyak_update(agent.q_target, agent.q, cfg.tau)
    env.close()
    return agent, hist

"""REINFORCE with a learned state-value baseline (Williams, 1992; Sutton & Barto ch. 13).

Kept deliberately minimal: it is the variance baseline that PPO's advantage
estimation and clipping improve upon, which the benchmarks make visible.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch

from .common import History, set_seed
from .networks import ActorCritic


@dataclass
class ReinforceConfig:
    env_id: str = "CartPole-v1"
    total_steps: int = 100_000
    gamma: float = 0.99
    lr: float = 1e-3
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    baseline: bool = True
    normalize_returns: bool = True
    hidden: tuple[int, ...] = (64, 64)
    seed: int = 0


def discounted_returns(rewards: list[float], gamma: float) -> np.ndarray:
    out = np.zeros(len(rewards), dtype=np.float32)
    g = 0.0
    for t in reversed(range(len(rewards))):
        g = rewards[t] + gamma * g
        out[t] = g
    return out


class ReinforceAgent:
    def __init__(self, cfg: ReinforceConfig, obs_dim: int, n_actions: int):
        self.cfg = cfg
        self.ac = ActorCritic(obs_dim, n_actions, cfg.hidden)
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=cfg.lr)

    @torch.no_grad()
    def greedy(self, obs: np.ndarray) -> int:
        return int(self.ac.actor(torch.as_tensor(obs, dtype=torch.float32)[None]).argmax())

    def update(self, obs, actions, rewards) -> float:
        cfg = self.cfg
        obs_t = torch.as_tensor(np.array(obs), dtype=torch.float32)
        act_t = torch.as_tensor(actions)
        ret = torch.as_tensor(discounted_returns(rewards, cfg.gamma))
        logp, ent, v = self.ac.evaluate(obs_t, act_t)
        adv = ret - v.detach() if cfg.baseline else ret
        if cfg.normalize_returns and len(adv) > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        loss = -(logp * adv).mean() - cfg.ent_coef * ent.mean()
        if cfg.baseline:
            loss = loss + cfg.vf_coef * 0.5 * (v - ret).pow(2).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()


def train_reinforce(cfg: ReinforceConfig) -> tuple[ReinforceAgent, History]:
    set_seed(cfg.seed)
    env = gym.make(cfg.env_id)
    agent = ReinforceAgent(cfg, env.observation_space.shape[0], env.action_space.n)
    hist = History()
    step = 0
    ep = 0
    while step < cfg.total_steps:
        obs, _ = env.reset(seed=cfg.seed if ep == 0 else None)
        ep += 1
        ep_obs, ep_act, ep_rew = [], [], []
        done = False
        while not done:
            with torch.no_grad():
                a = int(agent.ac.dist(torch.as_tensor(obs, dtype=torch.float32)[None]).sample())
            nxt, r, term, trunc, _ = env.step(a)
            ep_obs.append(obs)
            ep_act.append(a)
            ep_rew.append(float(r))
            obs = nxt
            step += 1
            done = term or trunc
        hist.log_episode(step, sum(ep_rew))
        hist.losses.append(agent.update(ep_obs, ep_act, ep_rew))
    env.close()
    return agent, hist

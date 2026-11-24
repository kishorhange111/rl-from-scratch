"""Proximal Policy Optimization (clip variant) for discrete actions.

Implements the details that matter in practice (Huang et al., 2022, "The 37
Implementation Details of PPO"): vectorized envs, GAE, per-minibatch advantage
normalisation, clipped value loss, entropy bonus, global grad-norm clipping,
linear LR annealing, approximate-KL early stopping, and correct bootstrapping
on time-limit truncation.

Reference: Schulman et al., 2017 — Proximal Policy Optimization Algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import torch

from .buffers import RolloutBatch, RolloutBuffer
from .common import History, set_seed
from .networks import ActorCritic


@dataclass
class PPOConfig:
    env_id: str = "CartPole-v1"
    total_steps: int = 100_000
    n_envs: int = 8
    n_steps: int = 128
    epochs: int = 4
    minibatches: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    clip_vloss: bool = True
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    lr: float = 2.5e-4
    anneal_lr: bool = True
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    hidden: tuple[int, ...] = (64, 64)
    seed: int = 0


@dataclass
class UpdateStats:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_frac: float
    explained_var: float


@dataclass
class PPOHistory(History):
    updates: list[UpdateStats] = field(default_factory=list)


def ppo_policy_loss(
    new_logp: torch.Tensor, old_logp: torch.Tensor, adv: torch.Tensor, clip: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Clipped surrogate objective (negated, to minimise).

    Returns (loss, approx_kl, clip_fraction). approx_kl uses the low-variance
    estimator ``E[(r - 1) - log r]`` (Schulman, 2020 blog "Approximating KL").
    """
    log_ratio = new_logp - old_logp
    ratio = log_ratio.exp()
    unclipped = ratio * adv
    clipped = ratio.clamp(1.0 - clip, 1.0 + clip) * adv
    loss = -torch.min(unclipped, clipped).mean()
    with torch.no_grad():
        approx_kl = ((ratio - 1.0) - log_ratio).mean()
        clip_frac = ((ratio - 1.0).abs() > clip).float().mean()
    return loss, approx_kl, clip_frac


def value_loss(
    new_v: torch.Tensor, old_v: torch.Tensor, returns: torch.Tensor, clip: float | None
) -> torch.Tensor:
    if clip is None:
        return 0.5 * (new_v - returns).pow(2).mean()
    v_clipped = old_v + (new_v - old_v).clamp(-clip, clip)
    return 0.5 * torch.max((new_v - returns).pow(2), (v_clipped - returns).pow(2)).mean()


def explained_variance(pred: np.ndarray, target: np.ndarray) -> float:
    var = np.var(target)
    return float("nan") if var == 0 else float(1.0 - np.var(target - pred) / var)


def make_vec_env(env_id: str, n_envs: int, seed: int) -> gym.vector.VectorEnv:
    env = gym.vector.SyncVectorEnv(
        [lambda: gym.make(env_id) for _ in range(n_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    env.reset(seed=seed)
    return env


class PPOAgent:
    def __init__(self, cfg: PPOConfig, obs_dim: int, n_actions: int):
        self.cfg = cfg
        self.ac = ActorCritic(obs_dim, n_actions, cfg.hidden)
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=cfg.lr, eps=1e-5)

    @torch.no_grad()
    def greedy(self, obs: np.ndarray) -> int:
        logits = self.ac.actor(torch.as_tensor(obs, dtype=torch.float32)[None])
        return int(logits.argmax())

    def update(self, batch: RolloutBatch) -> UpdateStats:
        cfg = self.cfg
        n = batch.obs.shape[0]
        mb_size = n // cfg.minibatches
        stats = []
        for _ in range(cfg.epochs):
            perm = torch.randperm(n)
            stop = False
            for start in range(0, n, mb_size):
                idx = perm[start : start + mb_size]
                new_logp, ent, new_v = self.ac.evaluate(batch.obs[idx], batch.actions[idx])
                adv = batch.advantages[idx]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                pl, kl, cf = ppo_policy_loss(new_logp, batch.log_probs[idx], adv, cfg.clip)
                vl = value_loss(
                    new_v,
                    batch.values[idx],
                    batch.returns[idx],
                    cfg.clip if cfg.clip_vloss else None,
                )
                loss = pl + cfg.vf_coef * vl - cfg.ent_coef * ent.mean()
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ac.parameters(), cfg.max_grad_norm)
                self.opt.step()
                stats.append((pl.item(), vl.item(), ent.mean().item(), kl.item(), cf.item()))
                if cfg.target_kl is not None and float(kl) > cfg.target_kl:
                    stop = True
                    break
            if stop:
                break
        s = np.mean(np.array(stats), axis=0)
        ev = explained_variance(batch.values.numpy(), batch.returns.numpy())
        return UpdateStats(*map(float, s), explained_var=ev)


def train_ppo(cfg: PPOConfig) -> tuple[PPOAgent, PPOHistory]:
    set_seed(cfg.seed)
    envs = make_vec_env(cfg.env_id, cfg.n_envs, cfg.seed)
    obs_dim = envs.single_observation_space.shape[0]
    agent = PPOAgent(cfg, obs_dim, envs.single_action_space.n)
    buf = RolloutBuffer(cfg.n_steps, cfg.n_envs, obs_dim)
    hist = PPOHistory()

    obs, _ = envs.reset(seed=cfg.seed)
    ep_ret = np.zeros(cfg.n_envs)
    n_updates = cfg.total_steps // (cfg.n_envs * cfg.n_steps)
    global_step = 0
    for update in range(n_updates):
        if cfg.anneal_lr:
            for g in agent.opt.param_groups:
                g["lr"] = cfg.lr * (1.0 - update / n_updates)
        while not buf.full:
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                a, logp, v = agent.ac.act(obs_t)
            next_obs, r, term, trunc, info = envs.step(a.numpy())
            global_step += cfg.n_envs
            r = r.astype(np.float32)
            ep_ret += r  # logged returns use the raw env reward
            # Time-limit truncation: bootstrap from V(final_obs) instead of treating it as death.
            if trunc.any():
                final = np.stack([info["final_obs"][i] for i in np.where(trunc)[0]])
                with torch.no_grad():
                    fv = agent.ac.value(torch.as_tensor(final, dtype=torch.float32)).numpy()
                r = r.copy()
                r[trunc] += cfg.gamma * fv
            done = np.logical_or(term, trunc)
            buf.add(obs, a.numpy(), logp.numpy(), r, done.astype(np.float32), v.numpy())
            for i in np.where(done)[0]:
                hist.log_episode(global_step, float(ep_ret[i]))
                ep_ret[i] = 0.0
            obs = next_obs
        with torch.no_grad():
            last_v = agent.ac.value(torch.as_tensor(obs, dtype=torch.float32)).numpy()
        batch = buf.finalize(last_v, cfg.gamma, cfg.lam)
        hist.updates.append(agent.update(batch))
    envs.close()
    return agent, hist

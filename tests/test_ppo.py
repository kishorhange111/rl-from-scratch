import numpy as np
import pytest
import torch

from rlfs.common import evaluate
from rlfs.ppo import PPOConfig, explained_variance, ppo_policy_loss, train_ppo, value_loss


def test_ratio_one_gives_negative_mean_advantage_and_zero_kl():
    logp = torch.log(torch.tensor([0.3, 0.6, 0.1]))
    adv = torch.tensor([1.0, -2.0, 0.5])
    loss, kl, cf = ppo_policy_loss(logp, logp.clone(), adv, clip=0.2)
    assert loss.item() == pytest.approx(-adv.mean().item())
    assert kl.item() == pytest.approx(0.0, abs=1e-7)
    assert cf.item() == 0.0


def test_clipping_removes_incentive_beyond_trust_region():
    old = torch.zeros(1)
    adv = torch.ones(1)
    # ratio = e^1 ≈ 2.72 > 1.2 with positive advantage -> objective clipped at 1.2
    new = torch.ones(1, requires_grad=True)
    loss, _, cf = ppo_policy_loss(new, old, adv, clip=0.2)
    assert loss.item() == pytest.approx(-1.2)
    loss.backward()
    assert new.grad.item() == 0.0  # no gradient pushes ratio further
    assert cf.item() == 1.0


def test_clipping_keeps_gradient_when_it_helps_pessimistically():
    # Negative advantage with ratio > 1+eps: min() picks the unclipped (worse) term,
    # so the gradient still pulls the ratio back down.
    new = torch.ones(1, requires_grad=True)
    loss, _, _ = ppo_policy_loss(new, torch.zeros(1), -torch.ones(1), clip=0.2)
    loss.backward()
    assert new.grad.item() > 0


def test_clipped_value_loss_is_pessimistic():
    old_v, ret = torch.tensor([0.0]), torch.tensor([1.0])
    new_v = torch.tensor([0.9])
    # clipped prediction 0.2 is further from the return -> its error dominates
    assert value_loss(new_v, old_v, ret, clip=0.2).item() == pytest.approx(0.5 * 0.8**2)
    assert value_loss(new_v, old_v, ret, clip=None).item() == pytest.approx(0.5 * 0.1**2)


def test_explained_variance():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert explained_variance(y, y) == pytest.approx(1.0)
    assert explained_variance(np.full(4, y.mean()), y) == pytest.approx(0.0)


def test_ppo_learns_cartpole():
    cfg = PPOConfig(total_steps=40_960, n_envs=8, n_steps=128, seed=0)
    agent, hist = train_ppo(cfg)
    assert len(hist.updates) == cfg.total_steps // (cfg.n_envs * cfg.n_steps)
    assert all(np.isfinite(u.approx_kl) for u in hist.updates)
    mean, _ = evaluate(agent.greedy, cfg.env_id, episodes=5)
    assert mean > 150, f"greedy eval return too low: {mean}"


def test_target_kl_early_stopping_runs():
    cfg = PPOConfig(total_steps=2048, n_envs=4, n_steps=128, target_kl=1e-6, seed=0)
    _, hist = train_ppo(cfg)
    assert len(hist.updates) == 4

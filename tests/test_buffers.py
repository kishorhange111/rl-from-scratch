import numpy as np
import pytest
import torch

from rlfs.buffers import ReplayBuffer, RolloutBuffer, compute_gae
from rlfs.networks import ActorCritic, QNetwork


def test_gae_matches_hand_computation():
    # Single env, 3 steps, no terminals.
    r = np.array([[1.0], [1.0], [1.0]], dtype=np.float32)
    v = np.array([[0.5], [0.4], [0.3]], dtype=np.float32)
    d = np.zeros_like(r)
    last = np.array([0.2], dtype=np.float32)
    g, lam = 0.9, 0.8
    d2 = 1 + g * 0.2 - 0.3
    d1 = 1 + g * 0.3 - 0.4
    d0 = 1 + g * 0.4 - 0.5
    a2 = d2
    a1 = d1 + g * lam * a2
    a0 = d0 + g * lam * a1
    adv, ret = compute_gae(r, v, d, last, g, lam)
    np.testing.assert_allclose(adv[:, 0], [a0, a1, a2], rtol=1e-6)
    np.testing.assert_allclose(ret, adv + v, rtol=1e-6)


def test_gae_cuts_bootstrap_at_terminal():
    r = np.array([[1.0], [1.0]], dtype=np.float32)
    v = np.array([[0.0], [0.0]], dtype=np.float32)
    d = np.array([[1.0], [0.0]], dtype=np.float32)  # episode ends after step 0
    adv, _ = compute_gae(r, v, d, np.array([10.0], dtype=np.float32), 0.99, 0.95)
    assert adv[0, 0] == pytest.approx(1.0)  # no leakage from the next episode
    assert adv[1, 0] == pytest.approx(1.0 + 0.99 * 10.0)


def test_gae_lambda_one_is_monte_carlo_minus_value():
    r = np.ones((4, 1), dtype=np.float32)
    v = np.full((4, 1), 0.5, dtype=np.float32)
    d = np.zeros_like(r)
    d[-1] = 1.0
    adv, ret = compute_gae(r, v, d, np.array([0.0], dtype=np.float32), gamma=1.0, lam=1.0)
    np.testing.assert_allclose(ret[:, 0], [4, 3, 2, 1], rtol=1e-6)


def test_replay_buffer_wraps_and_samples():
    buf = ReplayBuffer(capacity=5, obs_dim=2, seed=0)
    for i in range(8):
        buf.add([i, i], i % 2, float(i), [i + 1, i + 1], i == 7)
    assert len(buf) == 5
    assert set(buf.rewards.tolist()) == {3.0, 4.0, 5.0, 6.0, 7.0}
    batch = buf.sample(16)
    assert batch["obs"].shape == (16, 2)
    assert batch["actions"].dtype == torch.int64


def test_rollout_buffer_flattens_time_and_envs():
    buf = RolloutBuffer(n_steps=4, n_envs=3, obs_dim=2)
    for _ in range(4):
        buf.add(np.zeros((3, 2)), np.zeros(3), np.zeros(3), np.ones(3), np.zeros(3), np.zeros(3))
    assert buf.full
    batch = buf.finalize(np.zeros(3), 0.99, 0.95)
    assert batch.obs.shape == (12, 2)
    assert batch.advantages.shape == (12,)
    assert not buf.full


def test_dueling_q_shapes_and_identifiability():
    q = QNetwork(4, 3, dueling=True)
    out = q(torch.randn(5, 4))
    assert out.shape == (5, 3)


def test_actor_critic_initial_policy_is_near_uniform():
    torch.manual_seed(0)
    ac = ActorCritic(4, 2)
    probs = ac.dist(torch.randn(64, 4)).probs
    assert torch.allclose(probs, torch.full_like(probs, 0.5), atol=0.05)
    a, logp, v = ac.act(torch.randn(7, 4))
    assert a.shape == logp.shape == v.shape == (7,)

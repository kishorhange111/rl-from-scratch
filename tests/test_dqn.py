import pytest
import torch

from rlfs.common import evaluate
from rlfs.dqn import DQNConfig, epsilon_at, polyak_update, td_target, train_dqn
from rlfs.networks import QNetwork


class FixedQ(torch.nn.Module):
    """Returns a fixed Q table regardless of input (for target math tests)."""

    def __init__(self, table):
        super().__init__()
        self.table = torch.tensor(table, dtype=torch.float32)

    def forward(self, obs):
        return self.table.expand(obs.shape[0], -1)


def test_epsilon_schedule_endpoints():
    cfg = DQNConfig(total_steps=1000, eps_fraction=0.5)
    assert epsilon_at(0, cfg) == 1.0
    assert abs(epsilon_at(250, cfg) - 0.525) < 1e-9
    assert epsilon_at(500, cfg) == pytest.approx(cfg.eps_end)
    assert epsilon_at(999, cfg) == pytest.approx(cfg.eps_end)


def test_double_dqn_decouples_selection_and_evaluation():
    online = FixedQ([[1.0, 5.0]])  # online picks action 1
    target = FixedQ([[9.0, 2.0]])  # target would pick action 0 under vanilla max
    r, nxt, d = torch.zeros(1), torch.zeros(1, 4), torch.zeros(1)
    assert td_target(online, target, r, nxt, d, 1.0, double=True).item() == 2.0
    assert td_target(online, target, r, nxt, d, 1.0, double=False).item() == 9.0


def test_terminal_transition_has_no_bootstrap():
    q = FixedQ([[100.0, 100.0]])
    t = td_target(q, q, torch.tensor([1.5]), torch.zeros(1, 4), torch.ones(1), 0.99, True)
    assert t.item() == 1.5


def test_polyak_update_interpolates():
    a, b = QNetwork(2, 2), QNetwork(2, 2)
    for p in a.parameters():
        p.data.fill_(0.0)
    for p in b.parameters():
        p.data.fill_(1.0)
    polyak_update(a, b, tau=0.25)
    assert all(torch.allclose(p, torch.full_like(p, 0.25)) for p in a.parameters())


def test_dqn_learns_cartpole_quickly():
    """Smoke-level learning check: clearly beats a random policy (~22) in a short budget."""
    cfg = DQNConfig(total_steps=12_000, learning_starts=500, eps_fraction=0.25, seed=1)
    agent, hist = train_dqn(cfg)
    mean, _ = evaluate(agent.greedy, cfg.env_id, episodes=5)
    assert len(hist.returns) > 0
    assert mean > 60, f"greedy eval return too low: {mean}"

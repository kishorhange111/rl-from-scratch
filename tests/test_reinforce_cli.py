import json

import numpy as np
import pytest

from rlfs.cli import iqm, main
from rlfs.reinforce import discounted_returns


def test_discounted_returns():
    np.testing.assert_allclose(discounted_returns([1, 1, 1], 0.5), [1.75, 1.5, 1.0])


def test_iqm_ignores_outliers():
    assert iqm(np.array([0, 10, 10, 10, 10, 10, 10, 1000])) == pytest.approx(10.0)


def test_cli_writes_results(tmp_path):
    main(
        [
            "--algo",
            "reinforce",
            "--steps",
            "800",
            "--seeds",
            "0",
            "1",
            "--eval-episodes",
            "2",
            "--out",
            str(tmp_path),
        ]
    )
    data = json.loads((tmp_path / "reinforce_CartPole-v1.json").read_text())
    assert data["summary"]["n_seeds"] == 2
    assert len(data["runs"]) == 2
    assert data["runs"][0]["history"]["returns"]


def test_cli_overrides(tmp_path):
    main(
        [
            "--algo",
            "dqn",
            "--steps",
            "600",
            "--seeds",
            "0",
            "--eval-episodes",
            "1",
            "--out",
            str(tmp_path),
            "--tag",
            "vanilla",
            "--set",
            "double=false",
            "learning_starts=100",
        ]
    )
    data = json.loads((tmp_path / "dqn_CartPole-v1_vanilla.json").read_text())
    assert data["summary"]["overrides"] == {"double": False, "learning_starts": 100}

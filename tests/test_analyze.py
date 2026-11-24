import importlib.util
import json
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "analyze", Path(__file__).parents[1] / "benchmarks" / "analyze.py"
)
analyze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze)


def test_rolling_mean():
    np.testing.assert_allclose(analyze.rolling(np.array([2.0, 4.0, 6.0]), k=2), [2, 3, 5])


def test_steps_to_threshold_hit_and_miss():
    steps, rets = [10, 20, 30], [0.0, 500.0, 500.0]
    assert analyze.steps_to_threshold(steps, rets, 300.0, 100) == 30  # rolling mean 333 at idx 2
    assert analyze.steps_to_threshold(steps, rets, 1e9, 100) == 100


def test_analyze_file_roundtrip(tmp_path):
    run = {"eval_mean": 500.0, "history": {"steps": [100, 200], "returns": [480.0, 500.0]}}
    data = {
        "summary": {
            "algo": "ppo",
            "env": "CartPole-v1",
            "total_steps": 200,
            "n_seeds": 1,
            "eval_mean_over_seeds": 500.0,
            "eval_std_over_seeds": 0.0,
            "eval_iqm": 500.0,
            "mean_wall_seconds": 1.0,
        },
        "runs": [run],
    }
    p = tmp_path / "ppo_CartPole-v1.json"
    p.write_text(json.dumps(data))
    row = analyze.analyze_file(p)
    assert row["label"] == "ppo" and row["solved_seeds"] == 1
    assert "| CartPole-v1 | ppo |" in analyze.table([row])

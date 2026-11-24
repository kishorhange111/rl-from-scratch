"""Command-line entry point: train one algorithm over several seeds and write JSON results.

Example::

    rlfs --algo ppo --env CartPole-v1 --steps 100000 --seeds 0 1 2 3 4 --out benchmarks/results
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from .common import evaluate
from .dqn import DQNConfig, train_dqn
from .ppo import PPOConfig, train_ppo
from .reinforce import ReinforceConfig, train_reinforce

ALGOS = {
    "dqn": (DQNConfig, train_dqn),
    "ppo": (PPOConfig, train_ppo),
    "reinforce": (ReinforceConfig, train_reinforce),
}


def run_one(algo: str, env_id: str, steps: int, seed: int, eval_episodes: int, **overrides):
    cfg_cls, train = ALGOS[algo]
    cfg = cfg_cls(env_id=env_id, total_steps=steps, seed=seed, **overrides)
    t0 = time.perf_counter()
    agent, hist = train(cfg)
    wall = time.perf_counter() - t0
    mean, std = evaluate(agent.greedy, env_id, episodes=eval_episodes)
    return {
        "algo": algo,
        "env": env_id,
        "seed": seed,
        "total_steps": steps,
        "eval_mean": mean,
        "eval_std": std,
        "train_last20_mean": hist.recent_mean(20),
        "wall_seconds": round(wall, 2),
        "history": hist.to_dict(),
    }


def summarize(runs: list[dict]) -> dict:
    evals = np.array([r["eval_mean"] for r in runs])
    return {
        "algo": runs[0]["algo"],
        "env": runs[0]["env"],
        "total_steps": runs[0]["total_steps"],
        "n_seeds": len(runs),
        "eval_mean_over_seeds": float(evals.mean()),
        "eval_std_over_seeds": float(evals.std(ddof=1)) if len(runs) > 1 else 0.0,
        "eval_iqm": float(iqm(evals)),
        "mean_wall_seconds": float(np.mean([r["wall_seconds"] for r in runs])),
    }


def iqm(x: np.ndarray) -> float:
    """Interquartile mean (Agarwal et al., 2021) — robust to outlier seeds."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return float(x[lo:hi].mean()) if hi > lo else float(x.mean())


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="rlfs")
    p.add_argument("--algo", choices=sorted(ALGOS), required=True)
    p.add_argument("--env", default="CartPole-v1")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    p.add_argument("--tag", default="")
    p.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    args = p.parse_args(argv)

    overrides = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        overrides[k] = json.loads(v)

    torch.set_num_threads(1)
    runs = []
    for s in args.seeds:
        r = run_one(args.algo, args.env, args.steps, s, args.eval_episodes, **overrides)
        print(f"{args.algo} {args.env} seed={s} eval={r['eval_mean']:.1f} ({r['wall_seconds']}s)")
        runs.append(r)
    summary = summarize(runs)
    summary["overrides"] = overrides
    summary["hardware"] = f"{platform.processor() or platform.machine()} (CPU, 1 thread)"
    summary["torch"] = torch.__version__
    args.out.mkdir(parents=True, exist_ok=True)
    name = f"{args.algo}_{args.env}{('_' + args.tag) if args.tag else ''}.json"
    (args.out / name).write_text(json.dumps({"summary": summary, "runs": runs}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

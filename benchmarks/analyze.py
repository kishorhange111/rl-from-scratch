"""Turn benchmark JSON into a results table and learning-curve plots.

Final greedy return saturates on CartPole (cap = 500), so the table also reports
sample-efficiency metrics computed from the training curves:

* ``steps_to_475`` — first env step at which the 20-episode rolling mean of
  training returns reaches 475 (the CartPole-v1 "solved" threshold). Seeds that
  never get there count as the full budget, which *understates* the gap.
* ``auc`` — mean training return over the budget, interpolated on a common step
  grid (higher = learns faster).

Usage::

    python benchmarks/analyze.py benchmarks/results --plots docs/img
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SOLVED = {"CartPole-v1": 475.0, "Acrobot-v1": -100.0}


def rolling(x: np.ndarray, k: int = 20) -> np.ndarray:
    if len(x) == 0:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.empty(len(x))
    for i in range(len(x)):
        lo = max(0, i - k + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def curve_on_grid(steps, returns, budget: int, n: int = 200) -> np.ndarray:
    grid = np.linspace(0, budget, n)
    if not steps:
        return np.full(n, np.nan)
    sm = rolling(np.asarray(returns, dtype=float))
    return np.interp(grid, np.asarray(steps, dtype=float), sm)


def steps_to_threshold(steps, returns, threshold: float, budget: int) -> int:
    sm = rolling(np.asarray(returns, dtype=float))
    hit = np.nonzero(sm >= threshold)[0]
    return int(steps[hit[0]]) if len(hit) else budget


def analyze_file(path: Path) -> dict:
    data = json.loads(path.read_text())
    s = data["summary"]
    budget, env = s["total_steps"], s["env"]
    thr = SOLVED.get(env)
    curves, stt = [], []
    for r in data["runs"]:
        h = r["history"]
        curves.append(curve_on_grid(h["steps"], h["returns"], budget))
        if thr is not None:
            stt.append(steps_to_threshold(h["steps"], h["returns"], thr, budget))
    curves = np.array(curves)
    label = path.stem.replace(f"_{env}", "")
    return {
        "label": label,
        "env": env,
        "budget": budget,
        "eval_mean": s["eval_mean_over_seeds"],
        "eval_std": s["eval_std_over_seeds"],
        "eval_iqm": s["eval_iqm"],
        "auc": float(np.nanmean(curves)),
        "steps_to_solve_median": float(np.median(stt)) if stt else None,
        "solved_seeds": int(sum(x < budget for x in stt)) if stt else None,
        "n_seeds": s["n_seeds"],
        "wall": s["mean_wall_seconds"],
        "_curves": curves,
    }


def table(rows: list[dict]) -> str:
    out = [
        "| env | method | budget | greedy eval (mean ± std) | IQM | train AUC | "
        "median steps to solve | seeds solved | wall/seed |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: (r["env"], r["label"])):
        sts = "—" if r["steps_to_solve_median"] is None else f"{r['steps_to_solve_median']:,.0f}"
        solved = "—" if r["solved_seeds"] is None else f"{r['solved_seeds']}/{r['n_seeds']}"
        out.append(
            f"| {r['env']} | {r['label']} | {r['budget']:,} | {r['eval_mean']:.1f} ± "
            f"{r['eval_std']:.1f} | {r['eval_iqm']:.1f} | {r['auc']:.1f} | {sts} | {solved} | "
            f"{r['wall']:.0f}s |"
        )
    return "\n".join(out)


def plot(rows: list[dict], out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for env in sorted({r["env"] for r in rows}):
        fig, ax = plt.subplots(figsize=(6.4, 4))
        for r in sorted((r for r in rows if r["env"] == env), key=lambda r: r["label"]):
            c = r["_curves"]
            x = np.linspace(0, r["budget"], c.shape[1])
            m, sd = np.nanmean(c, 0), np.nanstd(c, 0)
            ax.plot(x, m, label=f"{r['label']} (n={r['n_seeds']})", lw=1.6)
            ax.fill_between(x, m - sd, m + sd, alpha=0.15)
        if env in SOLVED:
            ax.axhline(SOLVED[env], ls="--", c="gray", lw=0.8)
        ax.set(
            title=f"{env}: training return (20-ep rolling mean, ±1 sd over seeds)",
            xlabel="environment steps",
            ylabel="episode return",
        )
        ax.title.set_fontsize(9)
        ax.legend(fontsize=7, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        p = out_dir / f"curves_{env}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)
    return paths


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--plots", type=Path, default=None)
    ap.add_argument("--table-out", type=Path, default=None)
    a = ap.parse_args(argv)
    rows = [analyze_file(p) for p in sorted(a.results.glob("*.json"))]
    md = table(rows)
    print(md)
    if a.table_out:
        a.table_out.write_text(md + "\n")
    if a.plots:
        for p in plot(rows, a.plots):
            print("wrote", p)


if __name__ == "__main__":
    main()

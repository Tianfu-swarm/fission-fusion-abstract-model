#!/usr/bin/env python3
"""
Redraw the ARGoS real-controller desired-size sweep as the same two figures
robustness_groupsize_distribution.py draws for the abstract model (sweep line
and violin sweep), so they can sit side by side in the paper with identical
format and style.

Inputs, as the real-controller analysis already wrote them:
  output/<sweep>/result/summary_by_param.csv   size, runs, mean_group_size, std_group_size
  output/<sweep>/result/pooled_distribution.csv
      columns desired_size, size, probability (the header repeats "size" for
      the first two columns, so they are read by position, not by name)

Outputs:
  output/<sweep>/figure/groupsize_sweep_argos.png/pdf
  output/<sweep>/figure/groupsize_violin_sweep_argos.png/pdf
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from robustness_groupsize_distribution import plot_n_star_sweep, plot_violin_sweep

_HERE = Path(__file__).resolve().parent
DEFAULT_SWEEP_DIR = _HERE / ".." / "output" / "real_controller_sweep_desired_size_7to42_native_groupsize"


def read_summary(path: Path, num_agents: int) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            n_star = int(float(r["size"]))
            rows.append({"n_star": n_star, "mean_size": float(r["mean_group_size"]),
                         "std_size": float(r["std_group_size"]), "remainder": num_agents % n_star})
    return sorted(rows, key=lambda r: r["n_star"])


def read_pooled_distribution(path: Path) -> dict[int, dict[int, float]]:
    all_dist: dict[int, dict[int, float]] = defaultdict(dict)
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header: size,size,probability
        for desired, size, prob in reader:
            p = float(prob)
            if p > 0:
                all_dist[int(float(desired))][int(float(size))] = p
    return dict(all_dist)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR,
                    help="Real-controller desired-size sweep output dir (holds result/ and figure/).")
    ap.add_argument("--num-agents", type=int, default=42)
    ap.add_argument("--no-exclude-target-from-violin", dest="exclude_target_from_violin",
                    action="store_false", help="Include the n_star bucket in each violin body.")
    args = ap.parse_args()

    rows = read_summary(args.sweep_dir / "result" / "summary_by_param.csv", args.num_agents)
    all_dist = read_pooled_distribution(args.sweep_dir / "result" / "pooled_distribution.csv")
    stds_full = {r["n_star"]: r["std_size"] for r in rows}
    figure_dir = args.sweep_dir / "figure"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_n_star_sweep(rows, args.num_agents, figure_dir, stem="groupsize_sweep_argos")
    plot_violin_sweep(all_dist, figure_dir, args.exclude_target_from_violin,
                      stem="groupsize_violin_sweep_argos", stds_full=stds_full)
    print(f"Plot: {figure_dir / 'groupsize_sweep_argos.png'}")
    print(f"Plot: {figure_dir / 'groupsize_violin_sweep_argos.png'}")


if __name__ == "__main__":
    main()

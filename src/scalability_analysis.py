#!/usr/bin/env python3
"""
Analyze how the abstract model's convergence time scales with population
size. Two experiments, both keeping every point exactly divisible (no
leftover-remainder confound like robustness_groupsize_distribution.py deals
with) so exact convergence stays reachable everywhere:

  1. --mode fixed-n-star (default, the original experiment): desired group
     size n_star=14 fixed (PhysSimParams' own default), population sizes
     chosen as exact multiples of 14. num_agents is the only thing varied.
  2. --mode fixed-num-groups: number of groups held fixed at --num-groups
     (default 3) instead of n_star; population size is now the swept
     variable directly (--population-sizes, each must be an exact multiple
     of --num-groups), and desired_size = population_size / num_groups is
     derived per point rather than fixed. This asks the mirror-image
     question of experiment 1: does convergence time still scale the same
     way when it's the *target group size* that grows with the population,
     rather than the number of groups?

All other parameters are left at PhysSimParams' own defaults
(alpha_ctrl=beta_ctrl=0.5 included) in both modes.

Individual-agent effort (decisions to reach the target) is expected to stay
roughly flat as population grows, since each agent only ever needs to find
~13 local partners regardless of how many other groups exist elsewhere.
Convergence time is the more interesting question: the whole trial only
finishes once *every* group has independently reached exactly n_star, so
the total time is set by whichever group is slowest — an extreme-value
effect that can make convergence time grow with the number of groups even
though each group, in isolation, is no harder to form. Plotting mean
convergence time against the number of groups (num_agents / n_star) on a
log-x axis is meant to reveal which regime this is in: flat/log-ish growth
says the self-organization protocol has no real bottleneck at scale; faster
growth says it does.

Uses simulate_one_trial(...) unchanged (its normal done.all() convergence
check, not run_to_horizon — exact convergence is always reachable here
since num_agents is always a multiple of n_star), imported from
fission_fusion_abstract_model.py.

Outputs are written under output/scalability/, split into result/ (CSV) and
figure/ (PNG/PDF), matching the layout used by the other analysis scripts
in this repo. The two modes write distinctly-named files side by side
(scalability_sweep.* vs scalability_fixed_groups_sweep.*) so both can live
under the same --out-dir.
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from concurrent import futures
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fission_fusion_abstract_model import PhysSimParams, simulate_one_trial

_HERE = Path(__file__).resolve().parent

DEFAULT_GROUP_COUNTS = [1, 3, 5, 7, 9, 11, 13, 15, 18, 20, 22, 24, 26, 28, 30, 32]  # 16 points, ~evenly spaced in population (k*14)

# For --mode fixed-num-groups: desired_size = 5..20 (population_size / 3),
# 16 points spanning population 15..60. Scaled down from an earlier, much
# larger attempt (desired_size up to 150) after profiling showed convergence
# time grows roughly as desired_size^2.6-2.9 in *simulated* time — exact
# convergence requires every one of the 3 groups to simultaneously sit at
# exactly n_star, and larger groups have more members independently
# perturbing their own size, making that simultaneous coincidence rarer the
# bigger n_star gets (not a code performance issue — see scalability_analysis
# session notes). desired_size=150 alone was projected at ~11 days per trial.
# n*=14 (population=42) deliberately matches the fixed-n-star experiment's
# n_star=14 default, so the two experiments share that one data point.
# Tried extending to population=102 (n*=34) with --max-time 2,000,000, but
# even that budget wasn't enough — n*=30 came back at 95% conv_rate and
# n*=34 at just 26%, badly censoring the top two points' means — so pulled
# back to this range, where --max-time 20000 keeps every point >=75%
# converged (worst case n*=20; everything below n*=17 is ~100%).
DEFAULT_POPULATION_SIZES_FIXED_GROUPS = [15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60]

BOX_STYLE = dict(
    widths=0.3,
    patch_artist=True,
    boxprops=dict(facecolor="#cde2fb", edgecolor="#2a78d6", linewidth=1.2),
    medianprops=dict(color="#184f95", linewidth=1.5),
    whiskerprops=dict(color="#2a78d6"),
    capprops=dict(color="#2a78d6"),
    flierprops=dict(markeredgecolor="#2a78d6", markersize=4),
)


# ── multiprocessing worker ──────────────────────────────────────────────────

def _run_one(args: tuple) -> dict:
    p, seed = args
    rng = np.random.default_rng(seed)
    r = simulate_one_trial(p, rng)
    return {
        'converged': bool(r['converged']),
        'total_time': float(r['total_time']),
        'mean_decisions': float(r['decisions'].mean()),
    }


# ── sweep ────────────────────────────────────────────────────────────────────

def sweep_scalability(n_star: int, group_counts: list[int], trials_per_point: int,
                       seed: int = 0, processes: int | None = None,
                       base_kwargs: dict | None = None) -> tuple[list[dict], dict[int, list[float]]]:
    base_kwargs = base_kwargs or {}
    seed_seq = np.random.SeedSequence(seed)

    jobs = []
    job_num_agents = []
    for k in group_counts:
        num_agents = k * n_star
        p = PhysSimParams(num_agents=num_agents, n_star=n_star, **base_kwargs)
        child_seeds = seed_seq.spawn(trials_per_point)
        for cs in child_seeds:
            jobs.append((p, cs))
            job_num_agents.append(num_agents)

    processes = processes or os.cpu_count() or 1
    print(f"Sweeping population size over {len(group_counts)} values x {trials_per_point} trials "
          f"({len(jobs)} trials total, {processes} processes) …", flush=True)

    results: list[dict] = [None] * len(jobs)
    if processes == 1:
        for i, job in enumerate(jobs):
            results[i] = _run_one(job)
            if (i + 1) % max(1, len(jobs) // 20) == 0:
                print(f"  {i + 1}/{len(jobs)} trials done", flush=True)
    else:
        with futures.ProcessPoolExecutor(max_workers=processes) as pool:
            for i, r in enumerate(pool.map(_run_one, jobs, chunksize=4)):
                results[i] = r
                if (i + 1) % max(1, len(jobs) // 20) == 0:
                    print(f"  {i + 1}/{len(jobs)} trials done", flush=True)

    by_num_agents: dict[int, list[dict]] = defaultdict(list)
    for num_agents, r in zip(job_num_agents, results):
        by_num_agents[num_agents].append(r)

    rows = []
    per_group_times: dict[int, list[float]] = {}
    for k in group_counts:
        num_agents = k * n_star
        trials = by_num_agents[num_agents]
        converged = [t for t in trials if t['converged']]
        times = [t['total_time'] for t in converged]
        decs = [t['mean_decisions'] for t in converged]
        per_group_times[k] = times
        rows.append({
            'num_agents': num_agents,
            'num_groups': k,
            'conv_rate': len(converged) / len(trials) if trials else float('nan'),
            'mean_convergence_time_s': float(np.mean(times)) if times else float('nan'),
            'std_convergence_time_s': float(np.std(times)) if times else float('nan'),
            'mean_decisions': float(np.mean(decs)) if decs else float('nan'),
            'std_decisions': float(np.std(decs)) if decs else float('nan'),
        })
    return rows, per_group_times


def sweep_scalability_fixed_num_groups(num_groups: int, population_sizes: list[int], trials_per_point: int,
                                        seed: int = 0, processes: int | None = None,
                                        base_kwargs: dict | None = None
                                        ) -> tuple[list[dict], dict[int, list[float]]]:
    """Mirror of sweep_scalability with the roles reversed: num_groups is
    held fixed and population_size is the swept variable directly (each
    must already be an exact multiple of num_groups); desired_size =
    population_size // num_groups is derived per point so every trial still
    divides evenly and exact convergence stays reachable."""
    base_kwargs = base_kwargs or {}
    seed_seq = np.random.SeedSequence(seed)

    for pop in population_sizes:
        if pop % num_groups != 0:
            raise ValueError(f"population_size={pop} is not an exact multiple of num_groups={num_groups}")

    jobs = []
    job_population = []
    for pop in population_sizes:
        n_star = pop // num_groups
        p = PhysSimParams(num_agents=pop, n_star=n_star, **base_kwargs)
        child_seeds = seed_seq.spawn(trials_per_point)
        for cs in child_seeds:
            jobs.append((p, cs))
            job_population.append(pop)

    processes = processes or os.cpu_count() or 1
    print(f"Sweeping population size (num_groups={num_groups} fixed) over {len(population_sizes)} values x "
          f"{trials_per_point} trials ({len(jobs)} trials total, {processes} processes) …", flush=True)

    results: list[dict] = [None] * len(jobs)
    if processes == 1:
        for i, job in enumerate(jobs):
            results[i] = _run_one(job)
            if (i + 1) % max(1, len(jobs) // 20) == 0:
                print(f"  {i + 1}/{len(jobs)} trials done", flush=True)
    else:
        with futures.ProcessPoolExecutor(max_workers=processes) as pool:
            for i, r in enumerate(pool.map(_run_one, jobs, chunksize=4)):
                results[i] = r
                if (i + 1) % max(1, len(jobs) // 20) == 0:
                    print(f"  {i + 1}/{len(jobs)} trials done", flush=True)

    by_population: dict[int, list[dict]] = defaultdict(list)
    for pop, r in zip(job_population, results):
        by_population[pop].append(r)

    rows = []
    per_population_times: dict[int, list[float]] = {}
    for pop in population_sizes:
        n_star = pop // num_groups
        trials = by_population[pop]
        converged = [t for t in trials if t['converged']]
        times = [t['total_time'] for t in converged]
        decs = [t['mean_decisions'] for t in converged]
        per_population_times[pop] = times
        rows.append({
            'num_agents': pop,
            'num_groups': num_groups,
            'desired_size': n_star,
            'conv_rate': len(converged) / len(trials) if trials else float('nan'),
            'mean_convergence_time_s': float(np.mean(times)) if times else float('nan'),
            'std_convergence_time_s': float(np.std(times)) if times else float('nan'),
            'mean_decisions': float(np.mean(decs)) if decs else float('nan'),
            'std_decisions': float(np.std(decs)) if decs else float('nan'),
        })
    return rows, per_population_times


# ── output ───────────────────────────────────────────────────────────────────

def read_cached_results(out_dir: Path, group_counts: list[int]):
    """Re-read a prior run's result CSVs instead of re-simulating. Returns
    None (cache miss) if either CSV is missing or the summary's num_groups
    values don't match the requested set; a hit does NOT verify
    --n-star/--alpha/--beta/--trials-per-point/--max-time match the cached
    run, since those aren't stored in the CSVs — pass --force after
    changing any of those."""
    summary_path = out_dir / "result" / "scalability_sweep.csv"
    per_trial_path = out_dir / "result" / "per_trial_convergence.csv"
    if not (summary_path.exists() and per_trial_path.exists()):
        return None

    rows = []
    with open(summary_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                'num_agents': int(row['num_agents']),
                'num_groups': int(row['num_groups']),
                'conv_rate': float(row['conv_rate']),
                'mean_convergence_time_s': float(row['mean_convergence_time_s']),
                'std_convergence_time_s': float(row['std_convergence_time_s']),
                'mean_decisions': float(row['mean_decisions']),
                'std_decisions': float(row['std_decisions']),
            })
    if sorted(r['num_groups'] for r in rows) != sorted(group_counts):
        return None

    per_group_times: dict[int, list[float]] = defaultdict(list)
    with open(per_trial_path, newline="") as f:
        for row in csv.DictReader(f):
            per_group_times[int(row['num_groups'])].append(float(row['convergence_time_s']))

    return rows, dict(per_group_times)


def write_outputs(rows: list[dict], per_group_times: dict[int, list[float]],
                   n_star: int, out_dir: Path) -> None:
    result_dir = out_dir / "result"
    figure_dir = out_dir / "figure"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    with open(result_dir / "scalability_sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["num_agents", "num_groups", "conv_rate", "mean_convergence_time_s",
                    "std_convergence_time_s", "mean_decisions", "std_decisions"])
        for r in rows:
            w.writerow([r['num_agents'], r['num_groups'], round(r['conv_rate'], 4),
                        round(r['mean_convergence_time_s'], 3), round(r['std_convergence_time_s'], 3),
                        round(r['mean_decisions'], 3), round(r['std_decisions'], 3)])

    with open(result_dir / "per_trial_convergence.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["num_agents", "num_groups", "convergence_time_s"])
        for r in rows:
            k = r['num_groups']
            for t in per_group_times.get(k, []):
                w.writerow([r['num_agents'], k, round(t, 3)])

    # Box plot, not mean±std: convergence time is heavily right-skewed
    # (std is close to the mean), so a symmetric error bar can even dip
    # below zero — a box plot's median/IQR/outliers represent that shape
    # honestly. Positioned at the real population-size value (not
    # categorical slots) so the true spacing between points is visible —
    # DEFAULT_GROUP_COUNTS is chosen to be ~evenly spaced in population for
    # exactly this reason. Box width is a fixed data-unit size, not
    # proportional, since spacing is already ~uniform here (contrast the
    # log-scaled --mode fixed-num-groups sweep, which spans a much wider
    # range and would need width scaling instead).
    groups = [r['num_groups'] for r in rows]
    populations = [r['num_agents'] for r in rows]
    data = [per_group_times.get(g, []) for g in groups]

    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=300)
    box_width = 0.6 * min(np.diff(sorted(populations))) if len(populations) > 1 else 10
    box_style = {**BOX_STYLE, "widths": box_width}
    ax.boxplot(data, positions=populations, **box_style)
    ax.set_xlim(min(populations) - box_width, max(populations) + box_width)
    ax.set_xticks(populations)
    ax.set_xticklabels([str(g) for g in groups])
    ax.set_xlabel(f"Number of groups (desired size={n_star} fixed)", fontsize=15)
    ax.set_ylabel("Convergence time (s)", fontsize=15)
    ax.grid(linestyle="--", alpha=0.35, axis="y")
    ax.tick_params(labelsize=13)
    fig.tight_layout()
    fig.savefig(figure_dir / "scalability_sweep.png")
    fig.savefig(figure_dir / "scalability_sweep.pdf")
    plt.close(fig)

    print(f"\nCSV : {result_dir / 'scalability_sweep.csv'}")
    print(f"CSV : {result_dir / 'per_trial_convergence.csv'}")
    print(f"Plot: {figure_dir / 'scalability_sweep.png'}")


def read_cached_results_fixed_num_groups(out_dir: Path, population_sizes: list[int]):
    """Mirror of read_cached_results for --mode fixed-num-groups."""
    summary_path = out_dir / "result" / "scalability_fixed_groups_sweep.csv"
    per_trial_path = out_dir / "result" / "per_trial_convergence_fixed_groups.csv"
    if not (summary_path.exists() and per_trial_path.exists()):
        return None

    rows = []
    with open(summary_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                'num_agents': int(row['num_agents']),
                'num_groups': int(row['num_groups']),
                'desired_size': int(row['desired_size']),
                'conv_rate': float(row['conv_rate']),
                'mean_convergence_time_s': float(row['mean_convergence_time_s']),
                'std_convergence_time_s': float(row['std_convergence_time_s']),
                'mean_decisions': float(row['mean_decisions']),
                'std_decisions': float(row['std_decisions']),
            })
    if sorted(r['num_agents'] for r in rows) != sorted(population_sizes):
        return None

    per_population_times: dict[int, list[float]] = defaultdict(list)
    with open(per_trial_path, newline="") as f:
        for row in csv.DictReader(f):
            per_population_times[int(row['num_agents'])].append(float(row['convergence_time_s']))

    return rows, dict(per_population_times)


def write_outputs_fixed_num_groups(rows: list[dict], per_population_times: dict[int, list[float]],
                                    num_groups: int, out_dir: Path) -> None:
    """Mirror of write_outputs for --mode fixed-num-groups — same box-plot
    style, written to distinctly-named files so both modes can share one
    --out-dir."""
    result_dir = out_dir / "result"
    figure_dir = out_dir / "figure"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    with open(result_dir / "scalability_fixed_groups_sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["num_agents", "num_groups", "desired_size", "conv_rate", "mean_convergence_time_s",
                    "std_convergence_time_s", "mean_decisions", "std_decisions"])
        for r in rows:
            w.writerow([r['num_agents'], r['num_groups'], r['desired_size'], round(r['conv_rate'], 4),
                        round(r['mean_convergence_time_s'], 3), round(r['std_convergence_time_s'], 3),
                        round(r['mean_decisions'], 3), round(r['std_decisions'], 3)])

    with open(result_dir / "per_trial_convergence_fixed_groups.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["num_agents", "num_groups", "convergence_time_s"])
        for r in rows:
            pop = r['num_agents']
            for t in per_population_times.get(pop, []):
                w.writerow([pop, num_groups, round(t, 3)])

    # Positioned at the real population value (= num_groups * desired_size),
    # same as write_outputs, so true spacing is visible — but labeled by
    # desired_size (the swept variable) rather than the population number,
    # since desired_size is what's actually varying here (num_groups is
    # fixed) and its shorter labels don't need a reduced font size.
    populations = [r['num_agents'] for r in rows]
    desired_sizes = [r['desired_size'] for r in rows]
    data = [per_population_times.get(p, []) for p in populations]

    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=300)
    box_width = 0.6 * min(np.diff(sorted(populations))) if len(populations) > 1 else 1
    box_style = {**BOX_STYLE, "widths": box_width}
    ax.boxplot(data, positions=populations, **box_style)
    ax.set_xlim(min(populations) - box_width, max(populations) + box_width)
    ax.set_xticks(populations)
    ax.set_xticklabels([str(n) for n in desired_sizes])
    ax.set_xlabel(f"Desired group size (num_groups={num_groups} fixed)", fontsize=15)
    ax.set_ylabel("Convergence time (s)", fontsize=15)
    ax.grid(linestyle="--", alpha=0.35, axis="y")
    ax.tick_params(labelsize=13)
    fig.tight_layout()
    fig.savefig(figure_dir / "scalability_fixed_groups_sweep.png")
    fig.savefig(figure_dir / "scalability_fixed_groups_sweep.pdf")
    plt.close(fig)

    print(f"\nCSV : {result_dir / 'scalability_fixed_groups_sweep.csv'}")
    print(f"CSV : {result_dir / 'per_trial_convergence_fixed_groups.csv'}")
    print(f"Plot: {figure_dir / 'scalability_fixed_groups_sweep.png'}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["fixed-n-star", "fixed-num-groups"], default="fixed-n-star",
                     help="fixed-n-star (default): sweep population size at fixed n_star (--n-star, "
                          "--group-counts). fixed-num-groups: sweep population size directly at fixed "
                          "--num-groups, deriving desired_size = population_size / num_groups per point "
                          "(--population-sizes).")
    ap.add_argument("--n-star", type=int, default=14, help="[fixed-n-star mode]")
    ap.add_argument("--group-counts", type=str, default=",".join(map(str, DEFAULT_GROUP_COUNTS)),
                     help="[fixed-n-star mode] Comma-separated list of num_agents/n_star values to test.")
    ap.add_argument("--num-groups", type=int, default=3, help="[fixed-num-groups mode]")
    ap.add_argument("--population-sizes", type=str,
                     default=",".join(map(str, DEFAULT_POPULATION_SIZES_FIXED_GROUPS)),
                     help="[fixed-num-groups mode] Comma-separated list of population sizes to test; "
                          "each must be an exact multiple of --num-groups.")
    ap.add_argument("--trials-per-point", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--processes", type=int, default=None)
    ap.add_argument("--max-time", type=float, default=None,
                     help="Override PhysSimParams' own max_time default (7200s) if the largest "
                          "population size shows conv_rate < 100%% (see the red-x marker).")
    ap.add_argument("--alpha", type=float, default=0.5, help="alpha_ctrl (default 0.5).")
    ap.add_argument("--beta", type=float, default=0.5, help="beta_ctrl (default 0.5).")
    ap.add_argument("--out-dir", type=Path, default=_HERE / ".." / "output" / "scalability")
    ap.add_argument("--force", action="store_true",
                     help="Re-simulate even if a cached result CSV already matches this "
                          "group-count/population-size set (needed after changing --alpha/--beta/"
                          "--max-time/--n-star/--num-groups/--trials-per-point, since those aren't "
                          "recorded in the cache).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    base_kwargs = {'alpha_ctrl': args.alpha, 'beta_ctrl': args.beta}
    if args.max_time is not None:
        base_kwargs['max_time'] = args.max_time

    if args.mode == "fixed-num-groups":
        population_sizes = [int(x) for x in args.population_sizes.split(",")]
        cached = None if args.force else read_cached_results_fixed_num_groups(args.out_dir, population_sizes)
        if cached is not None:
            rows, per_population_times = cached
            print(f"Cache: existing result CSVs under {args.out_dir} match this population-size set — "
                  f"skipped simulation (pass --force to re-simulate)")
        else:
            rows, per_population_times = sweep_scalability_fixed_num_groups(
                args.num_groups, population_sizes, args.trials_per_point,
                seed=args.seed, processes=args.processes, base_kwargs=base_kwargs,
            )
            for r in rows:
                flag = "" if r['conv_rate'] == 1.0 else f"  <-- conv_rate={r['conv_rate']:.0%}, consider --max-time"
                print(f"  num_agents={r['num_agents']:4d} (n*={r['desired_size']:3d})  "
                      f"mean_t={r['mean_convergence_time_s']:.1f}s ± {r['std_convergence_time_s']:.1f}{flag}")

        write_outputs_fixed_num_groups(rows, per_population_times, args.num_groups, args.out_dir)
    else:
        group_counts = [int(x) for x in args.group_counts.split(",")]
        cached = None if args.force else read_cached_results(args.out_dir, group_counts)
        if cached is not None:
            rows, per_group_times = cached
            print(f"Cache: existing result CSVs under {args.out_dir} match this group-count set — "
                  f"skipped simulation (pass --force to re-simulate)")
        else:
            rows, per_group_times = sweep_scalability(
                args.n_star, group_counts, args.trials_per_point,
                seed=args.seed, processes=args.processes, base_kwargs=base_kwargs,
            )
            for r in rows:
                flag = "" if r['conv_rate'] == 1.0 else f"  <-- conv_rate={r['conv_rate']:.0%}, consider --max-time"
                print(f"  num_agents={r['num_agents']:4d} (k={r['num_groups']:3d})  "
                      f"mean_t={r['mean_convergence_time_s']:.1f}s ± {r['std_convergence_time_s']:.1f}{flag}")

        write_outputs(rows, per_group_times, args.n_star, args.out_dir)

    print(f"Output: {args.out_dir}")


if __name__ == "__main__":
    main()

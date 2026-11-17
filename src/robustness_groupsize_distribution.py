#!/usr/bin/env python3
"""
Analyze the abstract model's behavior as desired group size (n_star) sweeps
across a range that does not always evenly divide the population (N=42
fixed by default). Exact convergence (done.all(): every agent's group size
== n_star simultaneously) is structurally impossible whenever
num_agents % n_star != 0 — the sum of exact-n_star-sized groups is always a
multiple of n_star, which can never equal num_agents otherwise — so the
usual convergence-time metric is meaningless here. Instead every trial is
run to a *fixed* time horizon (simulate_one_trial(..., run_to_horizon=True),
imported unchanged from fission_fusion_abstract_model.py) and every agent's
true group size is tracked throughout, regardless of whether any group ever
hits n_star exactly.

This file merges what used to be two separate scripts —
n_star_divisibility_analysis.py and violin_groupsize_test.py — into one: the
violin figure now reuses the exact same simulated trials as the other two
outputs instead of re-simulating independently, so there's only ever one
sweep per run.

Three outputs:
  1. Summary sweep line: for each n_star in the swept range, the
     time-weighted mean group size experienced by a uniformly random agent,
     averaged over --trials-per-point trials, plotted against n_star with a
     y=x reference line — the vertical gap directly shows over/undershoot.
  2. For one representative n_star (the one whose remainder against N sits
     farthest from a clean divide, i.e. worst-case non-divisibility in the
     swept range), two distributions:
       - system-level: time-weighted group-size distribution pooled across
         all --trials-per-point trials at that n_star
       - one example agent's own time-distribution across sizes, from one
         additional tracked trial
  3. Violin sweep: the full time-weighted group-size distribution (not just
     its mean) pooled across --trials-per-point trials, for every n_star in
     the swept range, drawn as a smoothed violin per n_star — the same
     information the sweep line in (1) collapses into a single mean point,
     shown in full. By default the n_star bucket itself is dropped from each
     violin body (the mean line still uses the full distribution) so the
     "off-target" spread isn't dwarfed by a huge spike at the target —
     pass --no-exclude-target-from-violin to keep it inside the violins.

Outputs are written under output/robustness_groupsize_distribution/, split
into result/ and figure/ (result/ for CSVs, figure/ for PNG/PDF), matching
the layout used by the other analysis scripts in this repo.
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fission_fusion_abstract_model import PhysSimParams, simulate_one_trial

_HERE = Path(__file__).resolve().parent

# ── palette (fission_fusion dataviz skill reference palette) ────────────────
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
FILL = "#9ec1ee"       # light step of the blue ramp — violin body, recessive
EDGE = "#2a78d6"       # categorical slot 1 (blue) — violin outline
ACCENT = "#eb6834"     # categorical slot 2 (orange) — mean group size line
VIOLIN_BW = 0.6        # Gaussian smoothing bandwidth (in group-size units) for the violin bodies

# Every trial is measured only over its trailing TAIL_FRAC of the horizon —
# no stability test. A stability/burn-in test (try detecting when the
# trajectory settles and discarding everything before it) sounds more
# principled, but for the n_star values that need a *very* long horizon to
# organize at all (e.g. one big group covering half the population), it
# often never detects any stable tail and returns NaN — dropping that point
# from the sweep entirely. A fixed trailing window always returns something:
# for an n_star that's already settled, the trailing window sits inside the
# settled regime just like a detected one would; for one that's still
# organizing, it reports the still-approaching value instead of nothing,
# which is more useful than a gap in the plot.
TAIL_FRAC = 0.7  # discard the first 30% of the horizon as warm-up, measure the last 70%


# ── time-weighted stats from a trial's raw output ───────────────────────────

def _segment_durations(times: list[float], horizon: float) -> list[float]:
    """Duration each entry in a (t, ...) log held, given the trial's
    effective horizon (its `total_time`) as the closing boundary."""
    bounds = times[1:] + [horizon]
    return [b - t for t, b in zip(times, bounds)]


def tail_start(horizon: float, tail_frac: float = TAIL_FRAC) -> float:
    """Start of the trailing measurement window: everything before this is
    discarded as burn-in/organizing time."""
    return horizon * (1.0 - tail_frac)


def time_weighted_mean_size(snaps: list[tuple[float, list[int]]], horizon: float,
                             num_agents: int, t_start: float = 0.0) -> float:
    """Time-weighted mean size experienced by a uniformly random agent,
    restricted to [t_start, horizon] (pass t_start=tail_start(horizon) to
    only measure the trailing window). `snaps` logs (t, all current group
    sizes) at every
    S/A/D event; group membership only ever changes inside the same event
    that logs the next snap, so the configuration is constant between
    consecutive snap times. At any instant, `size` agents are each
    experiencing group size `size`, so integrating sum(size^2 over groups)
    / num_agents over time gives the per-agent time-weighted mean."""
    window = [(t, sizes) for t, sizes in snaps if t >= t_start]
    if not window or horizon <= t_start:
        return float("nan")
    times = [t for t, _ in window]
    durations = _segment_durations(times, horizon)
    total = sum(dt * sum(s * s for s in sizes) for dt, (_, sizes) in zip(durations, window) if dt > 0)
    return total / ((horizon - t_start) * num_agents)


def time_weighted_size_distribution(snaps: list[tuple[float, list[int]]], horizon: float,
                                     num_agents: int, t_start: float = 0.0) -> dict[int, float]:
    """Time-weighted probability distribution over group size, for a
    uniformly random agent (system-level, aggregated over all agents),
    restricted to [t_start, horizon]."""
    window = [(t, sizes) for t, sizes in snaps if t >= t_start]
    if not window or horizon <= t_start:
        return {}
    times = [t for t, _ in window]
    durations = _segment_durations(times, horizon)
    mass: dict[int, float] = defaultdict(float)
    for dt, (_, sizes) in zip(durations, window):
        if dt <= 0:
            continue
        for s in sizes:
            mass[s] += dt * s          # `s` agents each experience size `s` for dt
    total_mass = (horizon - t_start) * num_agents
    return {s: m / total_mass for s, m in mass.items()}


def agent_time_distribution(size_history: list[tuple[float, int]], horizon: float,
                             t_start: float = 0.0) -> dict[int, float]:
    """One agent's own time-distribution across sizes, from its
    `size_history` log (t, its group's size at t), restricted to
    [t_start, horizon]."""
    window = [(t, size) for t, size in size_history if t >= t_start]
    if not window or horizon <= t_start:
        return {}
    times = [t for t, _ in window]
    durations = _segment_durations(times, horizon)
    mass: dict[int, float] = defaultdict(float)
    for dt, (_, size) in zip(durations, window):
        if dt > 0:
            mass[size] += dt
    return {s: m / (horizon - t_start) for s, m in mass.items()}


def pooled_time_weighted_distribution(trials: list[dict], num_agents: int) -> dict[int, float]:
    """Duration-weighted average of time_weighted_size_distribution across
    trials, each restricted to its own trailing tail_start(horizon) window —
    shared by the representative-n_star distribution figure and the
    full-sweep violin figure, so both use identical pooling math."""
    pooled_mass: dict[int, float] = defaultdict(float)
    pooled_window = 0.0
    for r in trials:
        horizon = r['total_time']
        t_start = tail_start(horizon)
        window = horizon - t_start
        dist = time_weighted_size_distribution(r['snaps'], horizon, num_agents, t_start=t_start)
        for s, prob in dist.items():
            pooled_mass[s] += prob * window
        pooled_window += window
    return {s: m / pooled_window for s, m in pooled_mass.items()} if pooled_window > 0 else {}


# ── multiprocessing worker ──────────────────────────────────────────────────

def _run_one(args: tuple) -> dict:
    p, seed, track_agent = args
    rng = np.random.default_rng(seed)
    r = simulate_one_trial(p, rng, run_to_horizon=True, track_agent=track_agent)
    return {
        'total_time': r['total_time'],
        'snaps': r['snaps'],
        'size_history': r['size_history'],
    }


# ── sweep ────────────────────────────────────────────────────────────────────

def sweep_n_star(num_agents: int, n_star_values: list[int], trials_per_point: int,
                  seed: int = 0, processes: int | None = None,
                  base_kwargs: dict | None = None
                  ) -> tuple[list[dict], int, list[dict], dict, dict[int, list[dict]]]:
    """Runs the full sweep. Returns:
      - rows: one dict per n_star (n_star, remainder, dist_to_multiple, mean, std)
      - representative n_star (worst-case remainder in the swept range)
      - that n_star's trial dicts (each with total_time/snaps/size_history),
        for the representative-n_star distribution figure
      - base_kwargs, unchanged, passed back for convenience
      - by_n_star: every n_star's trial dicts, keyed by n_star — feeds the
        full-sweep violin figure so it doesn't need a second simulation pass
    """
    base_kwargs = base_kwargs or {}
    seed_seq = np.random.SeedSequence(seed)

    jobs = []
    job_n_star = []
    for n_star in n_star_values:
        p = PhysSimParams(num_agents=num_agents, n_star=n_star, **base_kwargs)
        child_seeds = seed_seq.spawn(trials_per_point)
        for cs in child_seeds:
            jobs.append((p, cs, None))
            job_n_star.append(n_star)

    processes = processes or os.cpu_count() or 1
    print(f"Sweeping n_star over {len(n_star_values)} values x {trials_per_point} trials "
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

    by_n_star: dict[int, list[dict]] = defaultdict(list)
    for n_star, r in zip(job_n_star, results):
        by_n_star[n_star].append(r)

    rows = []
    for n_star in n_star_values:
        trials = by_n_star[n_star]
        means = [time_weighted_mean_size(r['snaps'], r['total_time'], num_agents,
                                          t_start=tail_start(r['total_time']))
                 for r in trials]
        remainder = num_agents % n_star
        dist = min(remainder, n_star - remainder) if remainder else 0
        rows.append({
            'n_star': n_star,
            'remainder': remainder,
            'dist_to_multiple': dist,
            'mean_size': float(np.mean(means)),
            'std_size': float(np.std(means)),
        })

    representative = max(rows, key=lambda r: r['dist_to_multiple'])['n_star']
    rep_trials = by_n_star[representative]

    return rows, representative, rep_trials, base_kwargs, by_n_star


def representative_distributions(num_agents: int, representative: int, rep_trials: list[dict],
                                  base_kwargs: dict, seed: int = 0,
                                  example_agent: int = 0) -> tuple[dict[int, float], dict[int, float]]:
    """System-level distribution pooled across rep_trials' snaps, and one
    example agent's own distribution from one additional tracked trial —
    both restricted to each trial's own trailing tail_start(horizon)
    window."""
    system_dist = pooled_time_weighted_distribution(rep_trials, num_agents)

    p = PhysSimParams(num_agents=num_agents, n_star=representative, **base_kwargs)
    tracked_seed = np.random.SeedSequence(seed).spawn(1)[0]
    rng = np.random.default_rng(tracked_seed)
    r = simulate_one_trial(p, rng, run_to_horizon=True, track_agent=example_agent)
    agent_dist = agent_time_distribution(r['size_history'], r['total_time'], t_start=tail_start(r['total_time']))

    return system_dist, agent_dist


def all_n_star_distributions(num_agents: int, n_star_values: list[int],
                              by_n_star: dict[int, list[dict]]) -> dict[int, dict[int, float]]:
    """Pooled time-weighted distribution for every n_star in the sweep, not
    just the representative one — feeds the violin figure. Reuses the same
    trials sweep_n_star already ran, so this costs no extra simulation."""
    return {n_star: pooled_time_weighted_distribution(by_n_star[n_star], num_agents)
            for n_star in n_star_values}


# ── violin figure ────────────────────────────────────────────────────────────

def draw_smooth_violin(ax, x: float, dist: dict[int, float], halfwidth: float = 0.42,
                        bw: float = VIOLIN_BW) -> None:
    """Smoothed violin over a discrete distribution: each observed integer
    size contributes a Gaussian bump of std `bw` weighted by its
    probability, so single-size or sparse distributions still give a
    rounded body instead of a rectangle or a spike. The body is cut at
    0.5 below (no group is smaller than 1) and `2*bw` past the largest
    observed size above. Widths are scaled so the widest point of this
    violin spans `halfwidth`."""
    if not dist:
        return
    sizes = np.array(sorted(dist), dtype=float)
    probs = np.array([dist[int(s)] for s in sizes], dtype=float)
    y = np.linspace(max(0.5, sizes.min() - 2 * bw), sizes.max() + 2 * bw, 400)
    dens = (probs[:, None] * np.exp(-0.5 * ((y[None, :] - sizes[:, None]) / bw) ** 2)).sum(axis=0)
    w = dens / dens.max() * halfwidth
    ax.fill_betweenx(y, x - w, x + w, facecolor=FILL, edgecolor=EDGE, linewidth=0.8, zorder=3)


# ── output ───────────────────────────────────────────────────────────────────

def read_cached_results(out_dir: Path, n_star_values: list[int]):
    """Re-read a prior run's result CSVs instead of re-simulating — for
    e.g. a plot-style-only change. Returns None (cache miss) if any of the
    three CSVs is missing or its n_star values don't match the requested
    range; a hit does NOT verify --alpha/--beta/--t0/--trials-per-point/
    --num-agents match the cached run, since those aren't stored in the
    CSVs — pass --force after changing any of those."""
    sweep_path = out_dir / "result" / "n_star_sweep.csv"
    dist_path = out_dir / "result" / "representative_distributions.csv"
    all_dist_path = out_dir / "result" / "all_n_star_distributions.csv"
    if not (sweep_path.exists() and dist_path.exists() and all_dist_path.exists()):
        return None

    rows = []
    with open(sweep_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                'n_star': int(row['n_star']),
                'remainder': int(row['remainder']),
                'dist_to_multiple': int(row['dist_to_multiple']),
                'mean_size': float(row['mean_time_weighted_size']),
                'std_size': float(row['std_time_weighted_size']),
            })
    if sorted(r['n_star'] for r in rows) != sorted(n_star_values):
        return None

    representative = max(rows, key=lambda r: r['dist_to_multiple'])['n_star']
    system_dist: dict[int, float] = {}
    agent_dist: dict[int, float] = {}
    with open(dist_path, newline="") as f:
        for row in csv.DictReader(f):
            s = int(row['size'])
            sp, ap = float(row['system_probability']), float(row['agent_probability'])
            if sp > 0:
                system_dist[s] = sp
            if ap > 0:
                agent_dist[s] = ap

    all_dist: dict[int, dict[int, float]] = {n_star: {} for n_star in n_star_values}
    with open(all_dist_path, newline="") as f:
        for row in csv.DictReader(f):
            n_star = int(row['n_star'])
            if n_star not in all_dist:
                continue
            prob = float(row['probability'])
            if prob > 0:
                all_dist[n_star][int(row['size'])] = prob

    return rows, representative, system_dist, agent_dist, all_dist


def plot_n_star_sweep(rows: list[dict], num_agents: int, figure_dir: Path,
                      stem: str = "n_star_sweep") -> None:
    """Figure 1, split out like plot_violin_sweep(). Each row needs n_star,
    mean_size, std_size and remainder (num_agents % n_star)."""
    # figure 1: sweep line with y=x reference
    n_stars = [r['n_star'] for r in rows]
    means = [r['mean_size'] for r in rows]
    stds = [r['std_size'] for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=300)
    ax.plot([min(n_stars), max(n_stars)], [min(n_stars), max(n_stars)],
            linestyle="--", color="#999999", linewidth=1.2, label="y = x (exact target)")
    ax.errorbar(n_stars, means, yerr=stds, marker="o", markersize=5, linewidth=1.8,
                color="#2a78d6", ecolor="#2a78d6", elinewidth=1.0, capsize=3,
                label="mean size")
    divisors = [r['n_star'] for r in rows if r['remainder'] == 0]
    if divisors:
        ax.scatter(divisors, [r['mean_size'] for r in rows if r['remainder'] == 0],
                   s=70, facecolor="none", edgecolor="#e07b1f", linewidth=1.8, zorder=5,
                   label=f"num_agents={num_agents} divides evenly")
    ax.set_xlabel("Desired group size", fontsize=15)
    ax.set_ylabel("Mean group size", fontsize=15)
    ax.grid(linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(figure_dir / f"{stem}.png")
    fig.savefig(figure_dir / f"{stem}.pdf")
    plt.close(fig)


def plot_violin_sweep(all_dist: dict[int, dict[int, float]], figure_dir: Path,
                      exclude_target_from_violin: bool = True,
                      stem: str = "groupsize_violin_sweep",
                      stds_full: dict[int, float] | None = None) -> None:
    """Figure 3, split out so analyses of other data sources (e.g. ARGoS
    real-controller logs) can draw the exact same violin figure from their
    own {n_star: {size: probability}} distributions.

    `stds_full`, if given, is a {n_star: std} map of the run-to-run standard
    deviation of the full-distribution mean (e.g. std_group_size from a
    real-controller summary_by_param.csv) and is drawn as error bars on the
    mean-group-size line. Left as None (default), the line is drawn exactly
    as before — the abstract model has no such run-to-run std worth showing
    (it stays under 0.08 across the whole sweep, invisible at this axis
    scale), only the real-controller sweep passes this."""
    # figure 3: violin sweep across the full n_star range — same information
    # figure 1 collapses into one mean point, shown in full per n_star: violins
    # of the off-target distribution, the target line, and the mean line.
    n_star_values_sorted = sorted(all_dist)
    x_lo, x_hi = min(n_star_values_sorted), max(n_star_values_sorted)
    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=300, constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)

    means_full, means_violin = [], []
    for n_star in n_star_values_sorted:
        dist = all_dist[n_star]
        means_full.append(np.average(list(dist.keys()), weights=list(dist.values())) if dist else float("nan"))
        violin_dist = {s: p for s, p in dist.items() if s != n_star} if exclude_target_from_violin else dist
        means_violin.append(np.average(list(violin_dist.keys()), weights=list(violin_dist.values()))
                            if violin_dist else float("nan"))
        draw_smooth_violin(ax, n_star, violin_dist)

    # mean of what each violin shows, as a white dot inside the body
    ax.scatter(n_star_values_sorted, means_violin, s=75, facecolor="white", edgecolor=EDGE,
               linewidth=0.8, zorder=4)
    ax.plot([x_lo - 0.5, x_hi + 0.5], [x_lo - 0.5, x_hi + 0.5], color=INK_MUTED, linewidth=1.0,
            linestyle=(0, (4, 3)), zorder=2)
    mean_errorbar = None
    if stds_full is not None:
        yerr = [stds_full.get(n_star, 0.0) for n_star in n_star_values_sorted]
        # Drawn as two layered calls: the errorbar itself (line + whiskers + caps, no
        # marker) so capthick actually controls cap thickness, then the marker with its
        # own SURFACE-colored edge on top — passing markeredgewidth to errorbar() would
        # silently override capthick (matplotlib applies the last of the two).
        mean_errorbar = ax.errorbar(n_star_values_sorted, means_full, yerr=yerr, color=ACCENT, ecolor="#da4922",
                                    elinewidth=1.3, capsize=7, capthick=2.5, linewidth=4.0, marker="none",
                                    zorder=6, label="mean group size ± std across 100 runs")
        ax.plot(n_star_values_sorted, means_full, color=ACCENT, linewidth=0, marker="o", markersize=8.7,
                markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=7)
    else:
        ax.plot(n_star_values_sorted, means_full, color=ACCENT, linewidth=4.0, marker="o", markersize=8.7,
                markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=5)

    for spine in ax.spines.values():
        spine.set_color(INK_SECONDARY)
        spine.set_linewidth(1.0)
    ax.tick_params(colors=INK_MUTED, length=3, color=BASELINE)

    ax.set_facecolor(SURFACE)
    ax.set_xlim(x_lo - 0.7, x_hi + 0.7)
    ax.set_ylim(0, x_hi + 2)
    tick_step = 5 if len(n_star_values_sorted) > 10 else 1
    ax.set_xticks(range(x_lo, x_hi + 1, tick_step))
    ax.tick_params(labelsize=13)
    ax.set_xlabel("Desired group size, n*", fontsize=15, color=INK_SECONDARY)
    ax.set_ylabel("Group size experienced", fontsize=15, color=INK_SECONDARY)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    mean_handle = mean_errorbar if mean_errorbar is not None else Line2D(
        [], [], color=ACCENT, linewidth=4.0, marker="o", markersize=9.5,
        markeredgecolor=SURFACE, label="mean group size")
    handles = [
        Line2D([], [], color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)), label="target n*"),
        mean_handle,
        Patch(facecolor=FILL, edgecolor=EDGE,
              label="off-target distribution" if exclude_target_from_violin else "group-size distribution"),
        Line2D([], [], linestyle="none", marker="o", markersize=9.5, markerfacecolor="white",
               markeredgecolor=EDGE, markeredgewidth=0.8,
               label="mean of off-target distribution" if exclude_target_from_violin
               else "mean of distribution"),
    ]
    ax.legend(handles=handles, fontsize=12, frameon=False, loc="upper left", labelcolor=INK_SECONDARY)
    fig.savefig(figure_dir / f"{stem}.png", facecolor=fig.get_facecolor())
    fig.savefig(figure_dir / f"{stem}.pdf", facecolor=fig.get_facecolor())
    plt.close(fig)


def write_outputs(rows: list[dict], representative: int, system_dist: dict[int, float],
                   agent_dist: dict[int, float], all_dist: dict[int, dict[int, float]],
                   num_agents: int, out_dir: Path, exclude_target_from_violin: bool = True) -> None:
    result_dir = out_dir / "result"
    figure_dir = out_dir / "figure"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    with open(result_dir / "n_star_sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_star", "remainder", "dist_to_multiple", "mean_time_weighted_size",
                    "std_time_weighted_size"])
        for r in rows:
            w.writerow([r['n_star'], r['remainder'], r['dist_to_multiple'],
                        round(r['mean_size'], 4), round(r['std_size'], 4)])

    with open(result_dir / "representative_distributions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["size", "system_probability", "agent_probability"])
        sizes = sorted(set(system_dist) | set(agent_dist))
        for s in sizes:
            w.writerow([s, round(system_dist.get(s, 0.0), 6), round(agent_dist.get(s, 0.0), 6)])

    with open(result_dir / "all_n_star_distributions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_star", "size", "probability"])
        for n_star in sorted(all_dist):
            for s in sorted(all_dist[n_star]):
                w.writerow([n_star, s, round(all_dist[n_star][s], 6)])

    plot_n_star_sweep(rows, num_agents, figure_dir)

    # figure 2: representative n_star's two distributions, side by side.
    # System panel: binned to the nearest integer over the *full* observed
    # range (unobserved sizes shown as explicit zero-height bars rather than
    # skipped) so it reads as a proper histogram. Individual panel: exact
    # observed sizes only, no filled-in gaps (usually just 1-2 bars anyway).
    from matplotlib.ticker import MaxNLocator

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)

    if system_dist:
        max_size = max(system_dist)
        full_sizes = list(range(1, max_size + 1))
        full_probs = [system_dist.get(s, 0.0) for s in full_sizes]
    else:
        full_sizes, full_probs = [], []
    ax1.bar(full_sizes, full_probs, width=1.0, color="#2a78d6", edgecolor="white", alpha=0.85)

    agent_sizes = sorted(agent_dist)
    agent_probs = [agent_dist[s] for s in agent_sizes]
    ax2.bar(agent_sizes, agent_probs, width=1.0, color="#e07b1f", edgecolor="white", alpha=0.85)

    for ax, title in [
        (ax1, f"System-level (n_star={representative}, pooled trials)"),
        (ax2, f"One example agent (n_star={representative})"),
    ]:
        ax.axvline(representative, color="#333333", linestyle="--", linewidth=1.2,
                   label=f"n_star={representative}")
        ax.set_xlabel("Group size", fontsize=12)
        ax.set_ylabel("Time-weighted probability", fontsize=12)
        ax.set_title(title, fontsize=11)
        ax.grid(linestyle="--", alpha=0.4, axis="y")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(figure_dir / "representative_distributions.png")
    fig.savefig(figure_dir / "representative_distributions.pdf")
    plt.close(fig)

    plot_violin_sweep(all_dist, figure_dir, exclude_target_from_violin)

    print(f"\nCSV : {result_dir / 'n_star_sweep.csv'}")
    print(f"CSV : {result_dir / 'representative_distributions.csv'}")
    print(f"CSV : {result_dir / 'all_n_star_distributions.csv'}")
    print(f"Plot: {figure_dir / 'n_star_sweep.png'}")
    print(f"Plot: {figure_dir / 'representative_distributions.png'}")
    print(f"Plot: {figure_dir / 'groupsize_violin_sweep.png'}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-agents", type=int, default=42)
    ap.add_argument("--n-star-start", type=int, default=7)
    ap.add_argument("--n-star-end", type=int, default=42, help="inclusive")
    ap.add_argument("--trials-per-point", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--processes", type=int, default=None)
    ap.add_argument("--max-time", type=float, default=72000.0,
                     help="Fixed horizon per trial (default 72000s, needed for large n_star to "
                          "organize at all). Only the trailing TAIL_FRAC of this window is "
                          "actually measured (see tail_start()).")
    ap.add_argument("--alpha", type=float, default=1.0, help="alpha_ctrl (default 1.0, faster reaction).")
    ap.add_argument("--beta", type=float, default=1.0, help="beta_ctrl (default 1.0, faster reaction).")
    ap.add_argument("--t0", type=float, default=75.0, help="T_0, the STAY wait-time baseline (default 75.0).")
    ap.add_argument("--out-dir", type=Path,
                     default=_HERE / ".." / "output" / "robustness_groupsize_distribution")
    ap.add_argument("--exclude-target-from-violin", dest="exclude_target_from_violin",
                    action="store_true", default=True,
                    help="Drop the n_star bucket itself from each violin body in figure 3, so the "
                         "off-target spread isn't dwarfed by a huge spike at the target (default: on). "
                         "The mean-size line always "
                         "uses the full distribution regardless.")
    ap.add_argument("--no-exclude-target-from-violin", dest="exclude_target_from_violin",
                    action="store_false",
                    help="Include the n_star bucket in each violin body.")
    ap.add_argument("--force", action="store_true",
                     help="Re-simulate even if cached result CSVs already match this n_star range "
                          "(needed after changing --alpha/--beta/--t0/--max-time/--trials-per-point/"
                          "--num-agents, since those aren't recorded in the cache).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    n_star_values = list(range(args.n_star_start, args.n_star_end + 1))
    base_kwargs = {'max_time': args.max_time, 'alpha_ctrl': args.alpha,
                    'beta_ctrl': args.beta, 'T_0': args.t0}

    cached = None if args.force else read_cached_results(args.out_dir, n_star_values)
    if cached is not None:
        rows, representative, system_dist, agent_dist, all_dist = cached
        print(f"Cache: existing result CSVs under {args.out_dir} match this n_star range — "
              f"skipped simulation (pass --force to re-simulate)")
    else:
        rows, representative, rep_trials, base_kwargs, by_n_star = sweep_n_star(
            args.num_agents, n_star_values, args.trials_per_point,
            seed=args.seed, processes=args.processes, base_kwargs=base_kwargs,
        )
        print(f"\nRepresentative n_star (worst-case remainder in range): {representative} "
              f"(num_agents % n_star = {args.num_agents % representative})")

        system_dist, agent_dist = representative_distributions(
            args.num_agents, representative, rep_trials, base_kwargs, seed=args.seed,
        )
        all_dist = all_n_star_distributions(args.num_agents, n_star_values, by_n_star)

    write_outputs(rows, representative, system_dist, agent_dist, all_dist, args.num_agents, args.out_dir,
                  exclude_target_from_violin=args.exclude_target_from_violin)
    print(f"Output: {args.out_dir}")


if __name__ == "__main__":
    main()

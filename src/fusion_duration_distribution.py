#!/usr/bin/env python3
"""
fusion_duration_distribution.py
--------------------------------
Aggregates every bot*.csv under a data run, extracts the duration of each
FUSION episode (state == 'FUSION'), keeps only episodes that end by joining a
group (next state == 'STAY') — episodes that instead fall back to
RANDOM_WALK (target lost / rejected, no join) are excluded.

Two data layouts are supported, auto-detected under ../data (newest wins):
  data/result_*/bot*.csv                              — single run
  data/fusion_time_follow_range_*/fr<range>/result_*/bot*.csv
                                                        — one run per follow
                                                          range (fr5.0, fr10.0, …)

Outputs, per run/range:
  - a bar chart (normalized probability per duration bin) with every raw
    duration point scattered uniformly at random inside its own bar
Outputs, once, only when multiple follow ranges are present:
  - a line plot overlaying each range's binned distribution, one color per
    range, for comparison

CSV format (no header), one row per 0.01s tick, comma-separated:
  bot_name, time, c3, c4, c5, x, y, c8, c9, c10, state, c12
Only columns 2 (time) and 11 (state) are used here.
"""
import argparse
import csv
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# ── data discovery ───────────────────────────────────────────────────────────

def _latest(dirs, prefix: str) -> Path | None:
    matches = sorted(p for p in dirs if p.is_dir() and p.name.startswith(prefix))
    return matches[-1] if matches else None


def _latest_result_dir(parent: Path) -> Path:
    run = _latest(parent.iterdir(), "result_")
    if run is None:
        raise FileNotFoundError(f"No result_* run directory found under {parent}")
    return run


def discover_range_groups(data_root: Path, multi_dir: Path | None = None) -> dict[float, Path]:
    """{follow_range: run_dir} for one fusion_time_follow_range_* batch —
    `multi_dir` if given, otherwise the newest one under data_root (empty
    dict if none exists)."""
    if multi_dir is None:
        multi_dir = _latest(data_root.iterdir(), "fusion_time_follow_range_")
    if multi_dir is None:
        return {}
    groups = {}
    for fr_dir in sorted(multi_dir.iterdir()):
        m = re.fullmatch(r"fr([0-9.]+)", fr_dir.name)
        if not fr_dir.is_dir() or not m:
            continue
        groups[float(m.group(1))] = _latest_result_dir(fr_dir)
    if not groups:
        raise FileNotFoundError(f"No fr<range> subdirectories found under {multi_dir}")
    return groups


# ── duration extraction ──────────────────────────────────────────────────────

def _fusion_to_stay_durations(csv_path: Path) -> tuple[list[float], float]:
    """Read one bot's log and return (durations, run_length) — durations of
    FUSION runs that are immediately followed by STAY (FUSION -> RANDOM_WALK
    runs are dropped; a FUSION run still open at end-of-file is dropped too,
    since its outcome is unknown), and run_length = the log's last timestamp
    (how long this bot was recorded for)."""
    times, states = [], []
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 11:
                continue  # truncated row — file was still being written when read
            times.append(float(row[1]))
            states.append(row[10])

    durations = []
    n = len(states)
    i = 0
    while i < n:
        if states[i] == "FUSION":
            start_t = times[i]
            j = i
            while j < n and states[j] == "FUSION":
                j += 1
            if j < n and states[j] == "STAY":
                durations.append(times[j] - start_t)
            i = j
        else:
            i += 1
    run_length = times[-1] if times else 0.0
    return durations, run_length


def collect_durations(run_dir: Path) -> list[float]:
    bot_files = sorted(run_dir.glob("bot*.csv"))
    if not bot_files:
        raise FileNotFoundError(f"No bot*.csv files found under {run_dir}")
    all_durations = []
    run_lengths = []
    for bot_csv in bot_files:
        d, run_length = _fusion_to_stay_durations(bot_csv)
        all_durations.extend(d)
        run_lengths.append(run_length)

    run_lengths = np.asarray(run_lengths)
    if run_lengths.max() - run_lengths.min() < 0.01:  # effectively all the same
        length_str = f"{run_lengths.mean():.1f}s"
    else:
        length_str = f"{run_lengths.min():.1f}-{run_lengths.max():.1f}s (mean {run_lengths.mean():.1f}s)"
    print(f"  {run_dir}: {len(bot_files)} agents, run length {length_str}, "
          f"{len(all_durations)} FUSION→STAY episodes kept")
    return all_durations


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_distribution(durations: list[float], bins: int, out_dir: Path, suffix: str = ""):
    """Bar chart of normalized probability per bin, with every raw duration
    point scattered uniformly at random inside its own bar. Unchanged from
    the single-run version; `suffix` (e.g. '_fr5.0') just tags the filenames
    when this is one of several follow-range runs."""
    durations = np.asarray(durations)
    counts, edges = np.histogram(durations, bins=bins, range=(0.0, durations.max()))
    probs = counts / counts.sum()          # normalized probability (sums to 1)
    centers = (edges[:-1] + edges[1:]) / 2

    result_dir = out_dir / "result"
    figure_dir = out_dir / "figure"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / f"fusion_duration_distribution{suffix}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin_center_s", "bin_left_s", "bin_right_s", "count", "probability"])
        for c, lo, hi, cnt, p in zip(centers, edges[:-1], edges[1:], counts, probs):
            w.writerow([round(c, 4), round(lo, 4), round(hi, 4), int(cnt), round(p, 6)])

    # Bars = normalized probability per bin. Every raw duration point is then
    # scattered uniformly at random inside its own bar's rectangle (random x
    # across the bin width, random y between 0 and the bar height) — so the
    # cloud density visually confirms the bar height instead of just stacking.
    bin_idx = np.clip(np.digitize(durations, edges) - 1, 0, len(edges) - 2)
    bin_width = edges[1] - edges[0]
    rng = np.random.default_rng(0)
    xs = edges[bin_idx] + rng.uniform(0.08, 0.92, size=len(durations)) * bin_width
    ys = rng.uniform(0.04, 0.96, size=len(durations)) * probs[bin_idx]

    fig, ax = plt.subplots(figsize=(6.5, 5.0), dpi=300)
    ax.bar(centers, probs, width=bin_width * 0.95, color="#cde2fb",
           edgecolor="#2a78d6", linewidth=1.0, zorder=1)
    ax.scatter(xs, ys, s=14, color="#184f95", alpha=0.75,
               edgecolor="none", zorder=3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Fusion duration (s)", fontsize=13)
    ax.set_ylabel("Normalized probability", fontsize=13)
    ax.grid(linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    fig.savefig(figure_dir / f"fusion_duration_distribution{suffix}.pdf")
    plt.close(fig)

    print(f"  n={len(durations)}  mean={durations.mean():.3f}s  median={np.median(durations):.3f}s  "
          f"max={durations.max():.3f}s")
    print(f"  [Plot] -> {figure_dir / f'fusion_duration_distribution{suffix}.pdf'}")
    print(f"  [Data] -> {result_dir / f'fusion_duration_distribution{suffix}.csv'}")


def plot_range_comparison(durations_by_range: dict[float, list[float]], bins: int, out_dir: Path):
    """Line plot of normalized probability vs. duration, one colored line per
    follow range, all sharing the same bin edges so the shapes are comparable."""
    global_max = max(max(d) for d in durations_by_range.values())
    edges = np.linspace(0.0, global_max, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    result_dir = out_dir / "result"
    figure_dir = out_dir / "figure"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "fusion_duration_by_range.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["follow_range", "bin_center_s", "count", "probability"])
        fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=300)
        for i, r in enumerate(sorted(durations_by_range)):
            d = np.asarray(durations_by_range[r])
            counts, _ = np.histogram(d, bins=edges)
            probs = counts / counts.sum()
            for c, cnt, p in zip(centers, counts, probs):
                w.writerow([r, round(c, 4), int(cnt), round(p, 6)])
            ax.plot(centers, probs, marker="o", markersize=5, linewidth=1.8,
                    color=f"C{i}", label=f"follow range = {r:g}")

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Fusion duration (s)", fontsize=13)
    ax.set_ylabel("Normalized probability", fontsize=13)
    ax.grid(linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(figure_dir / "fusion_duration_by_range.pdf")
    plt.close(fig)

    print(f"\n[Plot] -> {figure_dir / 'fusion_duration_by_range.pdf'}")
    print(f"[Data] -> {result_dir / 'fusion_duration_by_range.csv'}")


def plot_distribution_grouped(durations_by_range: dict[float, list[float]], bins: int, out_dir: Path):
    """One bar chart for all follow ranges together: each bin is split into
    N side-by-side sub-bars (one per range, own color), with that range's raw
    duration points scattered uniformly at random inside its own sub-bar."""
    ranges = sorted(durations_by_range)
    global_max = max(max(d) for d in durations_by_range.values())
    edges = np.linspace(0.0, global_max, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    bin_width = edges[1] - edges[0]

    group_width = bin_width * 0.85
    n = len(ranges)
    sub_width = group_width / n
    rng = np.random.default_rng(0)

    result_dir = out_dir / "result"
    figure_dir = out_dir / "figure"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=300)

    x_smooth = np.linspace(1e-6, global_max, 300)
    fit_rows = []
    with open(result_dir / "fusion_duration_distribution.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["follow_range", "bin_center_s", "bin_left_s", "bin_right_s", "count", "probability"])
        for i, r in enumerate(ranges):
            d = np.asarray(durations_by_range[r])
            counts, _ = np.histogram(d, bins=edges)
            probs = counts / counts.sum()
            for c, lo, hi, cnt, p in zip(centers, edges[:-1], edges[1:], counts, probs):
                w.writerow([r, round(c, 4), round(lo, 4), round(hi, 4), int(cnt), round(p, 6)])

            sub_left = centers - group_width / 2 + i * sub_width
            color = f"C{i}"
            ax.bar(sub_left + sub_width / 2, probs, width=sub_width * 0.95,
                   color=color, alpha=0.35, edgecolor=color, linewidth=1.2,
                   zorder=1, label=f"follow range = {r:g}")

            bin_idx = np.clip(np.digitize(d, edges) - 1, 0, len(edges) - 2)
            xs = sub_left[bin_idx] + rng.uniform(0.08, 0.92, size=len(d)) * sub_width
            ys = rng.uniform(0.04, 0.96, size=len(d)) * probs[bin_idx]
            ax.scatter(xs, ys, s=12, color=color, alpha=0.85, edgecolor="none", zorder=3)

            # Gamma MLE fit (loc pinned to 0), overlaid as pdf(x) * bin_width so
            # it lands on the same "probability per bin" scale as the bars.
            shape, loc, scale = stats.gamma.fit(d[d > 0], floc=0)
            ax.plot(x_smooth, stats.gamma.pdf(x_smooth, shape, loc, scale) * bin_width,
                    color=color, linewidth=2.0, zorder=4)
            fit_rows.append((r, shape, scale))

    print("\nGamma fits (shape, scale; loc=0):")
    for r, shape, scale in fit_rows:
        print(f"  follow range={r:g}: shape={shape:.3f}  scale={scale:.3f}  "
              f"(scale/range={scale/r:.4f})")

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Fusion duration (s)", fontsize=13)
    ax.set_ylabel("Normalized probability", fontsize=13)
    ax.grid(linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(figure_dir / "fusion_duration_distribution.pdf")
    plt.close(fig)

    print(f"\n[Plot] -> {figure_dir / 'fusion_duration_distribution.pdf'}")
    print(f"[Data] -> {result_dir / 'fusion_duration_distribution.csv'}")


# ── goodness of fit: is Gamma actually a reasonable choice? ─────────────────

_GOF_CANDIDATES = {
    # name -> (scipy dist, fit kwargs). All three are non-negative-support
    # families, so loc is pinned to 0 for all of them (duration can't start
    # before zero).
    "exponential": (stats.expon,   {"floc": 0}),
    "gamma":       (stats.gamma,   {"floc": 0}),
    "lognorm":     (stats.lognorm, {"floc": 0}),
}


def goodness_of_fit(durations_by_range: dict[float, list[float]], out_dir: Path) -> dict:
    """Fit Exponential / Gamma / Log-normal (loc pinned to 0) to
    each follow range's positive durations, score by AIC (lower better) and
    a KS test (p > 0.05 -> not rejected), and save a CSV. Answers "is Gamma
    a reasonable choice" quantitatively rather than by eye. Returns
    {range: gamma_params} for reuse by the CDF plot."""
    result_dir = out_dir / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    gamma_params = {}
    print("\nGoodness of fit (lower AIC = better; KS p > 0.05 = not rejected):")
    for r in sorted(durations_by_range):
        d = np.asarray(durations_by_range[r])
        d = d[d > 0]
        fits = []
        for name, (dist, fit_kwargs) in _GOF_CANDIDATES.items():
            params = dist.fit(d, **fit_kwargs)
            ll = np.sum(dist.logpdf(d, *params))
            k_free = len(params) - len(fit_kwargs)
            aic = 2 * k_free - 2 * ll
            ks_stat, ks_p = stats.kstest(d, dist.cdf, args=params)
            fits.append((name, aic, ks_stat, ks_p, params))
            if name == "gamma":
                gamma_params[r] = params
        fits.sort(key=lambda t: t[1])
        print(f"  follow range={r:g}  (n={len(d)}):")
        for name, aic, ks_stat, ks_p, params in fits:
            print(f"    {name:12s} AIC={aic:9.2f}  KS_stat={ks_stat:.4f}  KS_p={ks_p:.4f}")
            rows.append([r, name, round(aic, 3), round(ks_stat, 4), round(ks_p, 4)]
                        + [round(v, 4) for v in params])

    with open(result_dir / "fusion_duration_goodness_of_fit.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["follow_range", "distribution", "AIC", "KS_stat", "KS_p",
                     "param1_shape_or_s", "param2_loc", "param3_scale"])
        w.writerows(rows)
    print(f"\n[Data] -> {result_dir / 'fusion_duration_goodness_of_fit.csv'}")
    return gamma_params


def plot_goodness_of_fit(durations_by_range: dict[float, list[float]], gamma_params: dict, out_dir: Path):
    """All follow ranges on one set of axes: empirical CDF (solid step) vs
    the fitted Gamma CDF (dashed curve), same color per range — if a step
    function hugs its matching dashed curve, the Gamma fit is a good
    description of that range's data, which is the direct visual evidence
    the AIC/KS numbers only summarize."""
    from matplotlib.lines import Line2D

    figure_dir = out_dir / "figure"
    figure_dir.mkdir(parents=True, exist_ok=True)
    ranges = sorted(durations_by_range)
    fig, ax = plt.subplots(figsize=(7.0, 5.5), dpi=300)
    global_max = 0.0
    for r, color in zip(ranges, [f"C{i}" for i in range(len(ranges))]):
        d = np.sort(np.asarray(durations_by_range[r]))
        d = d[d > 0]
        global_max = max(global_max, d[-1])
        ecdf = np.arange(1, len(d) + 1) / len(d)
        ax.step(d, ecdf, where="post", color=color, linewidth=1.8,
                 label=f"follow range = {r:g}")
        shape, loc, scale = gamma_params[r]
        x = np.linspace(0, d[-1], 300)
        ax.plot(x, stats.gamma.cdf(x, shape, loc, scale), color=color,
                 linewidth=1.8, linestyle="--")

    style_handles = [Line2D([0], [0], color="#333333", linewidth=1.8, label="Empirical"),
                      Line2D([0], [0], color="#333333", linewidth=1.8, linestyle="--", label="Gamma fit")]
    range_legend = ax.legend(loc="lower right", fontsize=10)
    ax.add_artist(range_legend)
    ax.legend(handles=style_handles, loc="center right", fontsize=10)

    ax.set_xlabel("Fusion duration (s)", fontsize=13)
    ax.set_ylabel("Cumulative probability", fontsize=13)
    ax.set_xlim(0, global_max)
    ax.set_ylim(0, 1)
    ax.grid(linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    fig.savefig(figure_dir / "fusion_duration_gof.pdf")
    plt.close(fig)
    print(f"[Plot] -> {figure_dir / 'fusion_duration_gof.pdf'}")


def main():
    here = Path(__file__).resolve().parent
    data_root = here / ".." / "data"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=None,
                     help="A single result_* run directory (containing bot*.csv). "
                          "Overrides auto-detection; disables the range comparison plot.")
    ap.add_argument("--batch-dir", type=Path, default=None,
                     help="A specific data/fusion_time_follow_range_* directory to use "
                          "(containing fr<range>/result_*/bot*.csv). Defaults to the "
                          "newest such directory under ../data.")
    ap.add_argument("--bins", type=int, default=30)
    ap.add_argument("--out-dir", type=Path,
                     default=here / ".." / "output" / "fusion_duration_distribution")
    args = ap.parse_args()

    if args.data_dir is not None:
        print(f"Reading {args.data_dir}")
        durations = collect_durations(args.data_dir)
        plot_distribution(durations, args.bins, args.out_dir)
        return

    groups = discover_range_groups(data_root, args.batch_dir)
    if not groups:
        run_dir = _latest_result_dir(data_root)
        print(f"Reading {run_dir}")
        durations = collect_durations(run_dir)
        plot_distribution(durations, args.bins, args.out_dir)
        return

    print(f"Found {len(groups)} follow-range group(s): "
          + ", ".join(f"fr{r:g}" for r in sorted(groups)))
    durations_by_range = {}
    for r, run_dir in sorted(groups.items()):
        print(f"fr{r:g}:")
        durations_by_range[r] = collect_durations(run_dir)

    plot_distribution_grouped(durations_by_range, args.bins, args.out_dir)
    if len(durations_by_range) > 1:
        plot_range_comparison(durations_by_range, args.bins, args.out_dir)

    gamma_params = goodness_of_fit(durations_by_range, args.out_dir)
    plot_goodness_of_fit(durations_by_range, gamma_params, args.out_dir)


if __name__ == "__main__":
    main()

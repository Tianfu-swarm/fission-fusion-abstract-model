#!/usr/bin/env python3
"""
Analyze real-controller sweep logs.

Expected layout:
  data/sweep_alpha_<timestamp>/alpha_<value>/run_*/bot*.csv
  data/sweep_beta_<timestamp>/beta_<value>/run_*/bot*.csv

Each row is expected to contain (no header):
  id,time,estimated_size,filtered_rounded_size,window_size,x,y,k,roundid,
  propagation_hops,mode,follow_range

Size is NOT read from the CSV's own estimated/filtered-rounded-size columns.
Instead, at every timestamp, agents currently in STAY mode are clustered by
position (connected components, edge when Euclidean distance <= cluster
radius); an agent's size is the size of its position cluster. Agents not in
STAY mode don't get a fresh size sample and keep holding whatever cluster
size they last had (same hold-last-value semantics used for ordinary gaps).

The convergence/oversize *counting method* is unchanged from the original
CSV-size-column version:
  - convergence_time_s: first time every agent's latest known (now:
    cluster-derived) size is == target_size simultaneously
  - oversize_time_s: sum over agents of time spent with their latest known
    (cluster-derived) size > target_size
Matches the abstract model's simulate_one_trial, which halts the instant
every agent reaches the target simultaneously: analysis of a run stops the
moment convergence_time_s is found, so oversize_time_s only reflects time
spent oversize *before* convergence, not whatever the log kept recording
afterward. A run that never converges is analyzed to the end of its log.

Outputs are written under output/<name>/, split into result/ and figure/:
  - result/per_run_convergence_oversize.csv
  - result/summary_by_param.csv (averages converged runs only)
  - figure/convergence_time.png/pdf, figure/oversize_time.png/pdf — box
    plots, one box per parameter value, converged runs only
The summary table is also printed to the console.

The alpha sweep (beta=0.5 fixed) and beta sweep (alpha=0.5 fixed) are
analyzed independently with the plain command above (once per sweep_dir).
Once both have been analyzed, `--combine` overlays their per-run results
into one pair of combined box plots — output/real_controller_combined_alpha_beta/
figure/{convergence_time,oversize_time}_combined.png/pdf — alpha sweep in
blue (legend "beta=0.5"), beta sweep in orange (legend "alpha=0.5"), sharing
one x-axis of parameter value since both sweeps use the same value grid.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import csv
import heapq
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, pstdev

_HERE = Path(__file__).resolve().parent  # anchor data/output roots here, not on cwd at launch

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
CSV_START_RE = re.compile(r"((?:bot)?[A-Za-z0-9_-]+)\s*,\s*([-+0-9.eE]+)\s*,")


@dataclass(frozen=True)
class FullRecord:
    agent_id: str
    time_s: float
    x: float
    y: float
    mode: str


@dataclass(frozen=True)
class RunResult:
    param_value: float
    param_dir: str
    run: str
    rows: int
    agents: int
    start_time_s: float | None
    end_time_s: float | None
    convergence_time_s: float | None
    oversize_time_s: float


@dataclass(frozen=True)
class SummaryRow:
    param_value: float
    runs: int
    converged_runs: int
    convergence_rate: float
    mean_convergence_time_s: float | None
    median_convergence_time_s: float | None
    std_convergence_time_s: float | None
    mean_oversize_time_s: float | None
    median_oversize_time_s: float | None
    std_oversize_time_s: float | None


def _latest_dir(parent: Path, prefix: str) -> Path | None:
    matches = sorted(p for p in parent.iterdir() if p.is_dir() and p.name.startswith(prefix))
    return matches[-1] if matches else None


def _parse_param(path: Path, param_name: str) -> float:
    m = re.fullmatch(rf"{re.escape(param_name)}_([-+0-9.]+)", path.name)
    if not m:
        raise ValueError(f"Cannot parse {param_name} from directory name: {path}")
    return float(m.group(1))


def infer_param_name(sweep_dir: Path, requested: str | None = None) -> str:
    if requested:
        return requested

    m = re.match(r"sweep_([A-Za-z0-9]+)_", sweep_dir.name)
    if m:
        return m.group(1)

    prefixes = sorted({
        child.name.split("_", 1)[0]
        for child in sweep_dir.iterdir()
        if child.is_dir() and "_" in child.name
    })
    if len(prefixes) == 1:
        return prefixes[0]
    raise ValueError(f"Cannot infer swept parameter name from {sweep_dir}; pass --param-name")


def _maybe_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _parse_csv_candidate(text: str) -> list[str] | None:
    try:
        row = next(csv.reader([text]))
    except csv.Error:
        return None
    row = [cell.strip() for cell in row]
    return row if len(row) >= 4 else None


def _row_from_line(line: str) -> list[str] | None:
    if "," not in line:
        return None

    line = ANSI_RE.sub("", line).strip()
    if not line:
        return None

    row = _parse_csv_candidate(line)
    if row and _maybe_float(row[1]) is not None:
        return row

    match = CSV_START_RE.search(line)
    if match:
        row = _parse_csv_candidate(line[match.start(1):])
        if row and _maybe_float(row[1]) is not None:
            return row
    return None


def iter_full_records(path: Path, id_col: int, time_col: int, x_col: int, y_col: int, mode_col: int):
    """Yield FullRecord(id, time, x, y, mode) for every parsable row. No
    header handling — the real bot*.csv logs never have one; column
    positions are fixed and passed in explicitly."""
    max_col = max(id_col, time_col, x_col, y_col, mode_col)
    with open(path, newline="", errors="replace") as f:
        for line in f:
            row = _row_from_line(line)
            if row is None or max_col >= len(row):
                continue
            t = _maybe_float(row[time_col])
            x = _maybe_float(row[x_col])
            y = _maybe_float(row[y_col])
            if t is None or x is None or y is None:
                continue
            yield FullRecord(agent_id=row[id_col], time_s=t, x=x, y=y, mode=row[mode_col].strip())


def _cluster_sizes(positions: dict[str, tuple[float, float]], radius: float) -> dict[str, int]:
    """Connected components over `positions`: an edge links two agents when
    their Euclidean distance <= radius. Returns {agent: size of its
    component}. positions with 0 or 1 agents are handled directly."""
    agents = list(positions)
    n = len(agents)
    if n == 0:
        return {}
    if n == 1:
        return {agents[0]: 1}

    pts = np.array([positions[a] for a in agents])
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff)
    adj = (d2 <= radius * radius)
    np.fill_diagonal(adj, False)

    _, labels = connected_components(csr_matrix(adj), directed=False)
    sizes = np.bincount(labels)
    return {agents[i]: int(sizes[labels[i]]) for i in range(n)}


def _run_paths(run_path: Path) -> list[Path]:
    if run_path.is_file():
        return [run_path]
    return sorted(p for p in run_path.rglob("*") if p.is_file() and p.suffix.lower() == ".csv")


def _expected_agents_from_paths(paths: list[Path]) -> int | None:
    bot_files = [p for p in paths if re.fullmatch(r"bot\d+", p.stem)]
    return len(bot_files) if len(bot_files) > 1 else None


# Generic two-parameter combined layout: <param1>_<value1>_<param2>_<value2>/run_*
# e.g. alpha_0.1_beta_0.5, or fr_10.0_fd_10.0. Non-greedy name group so it
# doesn't swallow the numeric value that follows.
COMBINED_DIR_RE = re.compile(r"^([A-Za-z_]+?)_([-+0-9.]+)_([A-Za-z_]+?)_([-+0-9.]+)$")


def is_combined_layout(sweep_dir: Path) -> bool:
    """True when sweep_dir holds a merged two-parameter sweep layout
    (<param1>_<v1>_<param2>_<v2>/run_*/bot*.csv, one combo per folder,
    sharing a midpoint where both params equal their fixed value) rather
    than the original single-parameter layout (<param>_<value>/run_*)."""
    return any(COMBINED_DIR_RE.fullmatch(p.name) for p in sweep_dir.iterdir() if p.is_dir())


def infer_combined_param_names(sweep_dir: Path) -> tuple[str, str]:
    """(param1_name, param2_name) from the first matching combo folder name,
    e.g. "alpha_0.1_beta_0.5" -> ("alpha", "beta"), "fr_10.0_fd_10.0" -> ("fr", "fd")."""
    for p in sorted(sweep_dir.iterdir()):
        if not p.is_dir():
            continue
        m = COMBINED_DIR_RE.fullmatch(p.name)
        if m:
            return m.group(1), m.group(3)
    raise FileNotFoundError(f"No <param1>_<v1>_<param2>_<v2> folders found under {sweep_dir}")


def discover_combined_runs(
    sweep_dir: Path, vary_param: str, fixed_param: str, fixed_value: float,
) -> list[tuple[float, Path, Path]]:
    """Same return shape as discover_runs: (varying_param_value, param_dir,
    run_path). Folders are named <p1>_<v1>_<p2>_<v2> (p1/p2 order fixed per
    sweep_dir, see infer_combined_param_names); vary_param/fixed_param name
    which of the two varies (its value becomes param_value) vs. is held at
    fixed_value — e.g. vary_param="alpha", fixed_param="beta" keeps only
    folders where beta == fixed_value, and param_value is that folder's alpha."""
    runs: list[tuple[float, Path, Path]] = []
    for param_dir in sorted(p for p in sweep_dir.iterdir() if p.is_dir()):
        m = COMBINED_DIR_RE.fullmatch(param_dir.name)
        if not m:
            continue
        names_values = {m.group(1): float(m.group(2)), m.group(3): float(m.group(4))}
        if vary_param not in names_values or fixed_param not in names_values:
            continue
        if names_values[fixed_param] != fixed_value:
            continue
        varying_v = names_values[vary_param]

        direct_dirs = sorted(p for p in param_dir.iterdir() if p.is_dir())
        dir_stems = {p.name for p in direct_dirs}
        direct_files = sorted(
            p for p in param_dir.iterdir()
            if (
                p.is_file()
                and p.suffix.lower() == ".csv"
                and p.stem not in dir_stems
            )
        )
        for run_path in direct_files + direct_dirs:
            runs.append((varying_v, param_dir, run_path))
    if not runs:
        raise FileNotFoundError(
            f"No run files/directories found under {sweep_dir} "
            f"varying {vary_param} with {fixed_param} fixed at {fixed_value}"
        )
    return runs


def discover_runs(sweep_dir: Path, param_name: str) -> list[tuple[float, Path, Path]]:
    runs: list[tuple[float, Path, Path]] = []
    for param_dir in sorted(p for p in sweep_dir.iterdir() if p.is_dir() and p.name.startswith(f"{param_name}_")):
        param_value = _parse_param(param_dir, param_name)
        direct_dirs = sorted(p for p in param_dir.iterdir() if p.is_dir())
        dir_stems = {p.name for p in direct_dirs}
        direct_files = sorted(
            p for p in param_dir.iterdir()
            if (
                p.is_file()
                and p.suffix.lower() == ".csv"
                and p.stem not in dir_stems
            )
        )
        for run_path in direct_files + direct_dirs:
            runs.append((param_value, param_dir, run_path))
    if not runs:
        raise FileNotFoundError(f"No run files/directories found under {sweep_dir}/{param_name}_*")
    return runs


def limit_runs_per_param(
    runs: list[tuple[float, Path, Path]],
    run_limit: int | None,
) -> list[tuple[float, Path, Path]]:
    if run_limit is None:
        return runs
    kept: list[tuple[float, Path, Path]] = []
    counts: dict[float, int] = defaultdict(int)
    for item in runs:
        param_value = item[0]
        if counts[param_value] < run_limit:
            kept.append(item)
            counts[param_value] += 1
    return kept


def analyze_run(
    param_value: float,
    param_dir: Path,
    run_path: Path,
    target_size: float,
    num_agents: int | None,
    id_col: int,
    time_col: int,
    x_col: int,
    y_col: int,
    mode_col: int,
    cluster_radius: float,
    stay_mode: str,
) -> RunResult:
    """Size comes from position-clustering STAY-mode agents (radius =
    cluster_radius), not from any size column in the CSV. Convergence/
    oversize are counted exactly as before: convergence_time_s is the first
    timestamp at which every agent's latest known size == target_size
    simultaneously; oversize_time_s sums, per agent, the time between
    consecutive size samples where the *earlier* sample was > target_size.
    Only STAY-mode rows produce a fresh size sample for their agent — a
    non-STAY row leaves that agent's last cluster size in force, the same
    hold-last-value behavior a plain data gap would get.

    Matches the abstract model's simulate_one_trial, which halts the
    instant every agent reaches the target simultaneously: once
    convergence_time_s is found, analysis of this run stops right there —
    rows/agents/end_time_s and oversize_time_s only reflect data up to and
    including that moment, not whatever the log kept recording afterward.
    A run that never converges is analyzed to the end of its log, since
    there's no convergence point to cut off at."""
    paths = _run_paths(run_path)
    rows = 0
    agents: set[str] = set()
    start_time_s: float | None = None
    end_time_s: float | None = None
    expected_agents = num_agents or _expected_agents_from_paths(paths)

    stay_pos: dict[str, tuple[float, float]] = {}   # currently-STAY agents -> position
    latest: dict[str, float] = {}                   # agent -> latest known cluster size
    prev_sample: dict[str, tuple[float, float]] = {}  # agent -> (time, size) of its last STAY sample
    conv_t: float | None = None
    over_t = 0.0

    streams = [iter_full_records(path, id_col, time_col, x_col, y_col, mode_col) for path in paths]
    merged = heapq.merge(*streams, key=lambda r: (r.time_s, r.agent_id))

    batch: list[FullRecord] = []
    batch_t: float | None = None

    def flush(batch_time: float):
        nonlocal conv_t, over_t
        # An agent can appear more than once in the same tick with its mode
        # changing mid-batch (e.g. STAY -> FISSION at the exact same
        # timestamp) — only that agent's LAST record this tick decides its
        # STAY/position status, not every row seen along the way.
        last_by_agent: dict[str, FullRecord] = {}
        for rec in batch:
            last_by_agent[rec.agent_id] = rec

        touched = []
        for aid, rec in last_by_agent.items():
            if rec.mode == stay_mode:
                stay_pos[aid] = (rec.x, rec.y)
                touched.append(aid)
            else:
                stay_pos.pop(aid, None)
        if touched:
            sizes = _cluster_sizes(stay_pos, cluster_radius)
            for aid in touched:
                size = float(sizes[aid])
                prev = prev_sample.get(aid)
                if prev is not None and prev[1] > target_size:
                    over_t += max(0.0, batch_time - prev[0])
                prev_sample[aid] = (batch_time, size)
                latest[aid] = size

        if conv_t is None:
            current_expected = expected_agents or len(agents)
            if current_expected and len(latest) >= current_expected and all(
                size == target_size for size in latest.values()
            ):
                conv_t = batch_time

    for record in merged:
        if batch_t is not None and record.time_s != batch_t:
            flush(batch_t)
            if conv_t is not None:
                break
            batch = []

        rows += 1
        agents.add(record.agent_id)
        start_time_s = record.time_s if start_time_s is None else min(start_time_s, record.time_s)
        end_time_s = record.time_s if end_time_s is None else max(end_time_s, record.time_s)
        batch_t = record.time_s
        batch.append(record)

    if conv_t is None and batch:
        flush(batch_t)

    return RunResult(
        param_value=param_value,
        param_dir=param_dir.name,
        run=run_path.name,
        rows=rows,
        agents=len(agents),
        start_time_s=start_time_s,
        end_time_s=end_time_s,
        convergence_time_s=conv_t,
        oversize_time_s=over_t,
    )


def analyze_run_job(args: tuple) -> RunResult:
    return analyze_run(*args)


def _round_or_blank(value: float | None, ndigits: int = 6) -> str:
    if value is None or math.isnan(value):
        return ""
    return str(round(value, ndigits))


def _blank_to_float(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    return float(value)


def read_summary(summary_path: Path, param_name: str) -> list[SummaryRow]:
    with open(summary_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append(SummaryRow(
                param_value=float(row[param_name]),
                runs=int(row["runs"]),
                converged_runs=int(row["converged_runs"]),
                convergence_rate=float(row["convergence_rate"]),
                mean_convergence_time_s=_blank_to_float(row["mean_convergence_time_s"]),
                median_convergence_time_s=_blank_to_float(row["median_convergence_time_s"]),
                std_convergence_time_s=_blank_to_float(row["std_convergence_time_s"]),
                mean_oversize_time_s=_blank_to_float(row["mean_oversize_time_s"]),
                median_oversize_time_s=_blank_to_float(row["median_oversize_time_s"]),
                std_oversize_time_s=_blank_to_float(row["std_oversize_time_s"]),
            ))
    return rows


def _fmt(value: float | None, ndigits: int = 1) -> str:
    return "-" if value is None else f"{value:.{ndigits}f}"


def print_summary(summary_rows: list[SummaryRow], param_name: str) -> None:
    header = (
        f"{param_name:>8} {'runs':>5} {'conv':>5} {'rate%':>6}  "
        f"{'conv_mean':>10} {'conv_med':>9} {'conv_std':>9}  "
        f"{'over_mean':>10} {'over_med':>9} {'over_std':>9}"
    )
    print("\n" + header)
    print("-" * len(header))
    for r in summary_rows:
        print(
            f"{r.param_value:>8g} {r.runs:>5} {r.converged_runs:>5} {r.convergence_rate * 100:>6.1f}  "
            f"{_fmt(r.mean_convergence_time_s):>10} {_fmt(r.median_convergence_time_s):>9} "
            f"{_fmt(r.std_convergence_time_s):>9}  "
            f"{_fmt(r.mean_oversize_time_s):>10} {_fmt(r.median_oversize_time_s):>9} "
            f"{_fmt(r.std_oversize_time_s):>9}"
        )
    print()


def read_per_run(per_run_path: Path, param_name: str) -> list[tuple[float, float | None, float]]:
    """(param_value, convergence_time_s_or_None, oversize_time_s) per row, for
    rebuilding the box plots from the cached CSV without re-reading raw logs."""
    with open(per_run_path, newline="") as f:
        reader = csv.DictReader(f)
        out = []
        for row in reader:
            out.append((
                float(row[param_name]),
                _blank_to_float(row["convergence_time_s"]),
                float(row["oversize_time_s"]),
            ))
    return out


def _scale_exp_tag(out_dir: Path) -> str | None:
    """'scale_exp1' / 'scale_exp2' when out_dir is one of the scalability
    experiment folders (e.g. real_controller_sweep_scale_exp1), else None."""
    m = re.search(r"scale_exp\d+", out_dir.name)
    return m.group(0) if m else None


def _single_sweep_figsize(param_name: str, out_dir: Path) -> tuple[float, float]:
    """Scalability experiments use 6.5x6.5 in (matching the abstract model's
    scalability figures); tmax figures use 6.5x5.0 in; other single-param
    sweeps keep 7.0x4.8 in."""
    if _scale_exp_tag(out_dir):
        return (6.5, 6.5)
    return (6.5, 5.0) if param_name == "tmax" else (7.0, 4.8)


# Scalability experiments: x-axis labels and font sizes match the abstract
# model's scalability figures (scalability_analysis.py) so the two sets of
# figures can sit side by side in the paper.
SCALE_EXP_XLABEL = {
    "scale_exp1": "Desired group size (num_groups=3 fixed)",
    "scale_exp2": "Number of groups (desired size=14 fixed)",
}


def _label_axes(ax, out_dir: Path, param_name: str, ylabel: str) -> None:
    """Axis labels for a single-sweep figure; scalability experiments get
    the abstract model's label wording and font sizes."""
    tag = _scale_exp_tag(out_dir)
    if tag in SCALE_EXP_XLABEL:
        ax.set_xlabel(SCALE_EXP_XLABEL[tag], fontsize=15)
        ax.set_ylabel(ylabel, fontsize=15)
        ax.tick_params(labelsize=13)
    else:
        ax.set_xlabel(param_name)
        ax.set_ylabel(ylabel)


def _figure_stem(stem: str, out_dir: Path) -> str:
    """Append the _scale_expN suffix for scalability experiment folders."""
    tag = _scale_exp_tag(out_dir)
    return f"{stem}_{tag}" if tag else stem


# Scalability box plots drop points above Q3 + 2.5*IQR.
# Only the figure is affected; the result CSVs keep every run.
FAR_OUTLIER_IQR_FACTOR = 2.5


def _drop_far_outliers(values: list[float]) -> list[float]:
    if len(values) < 4:
        return values
    q1, q3 = np.percentile(values, [25, 75])
    upper = q3 + FAR_OUTLIER_IQR_FACTOR * (q3 - q1)
    return [v for v in values if v <= upper]


def plot_boxplots(rows: list[tuple[float, float | None, float]], out_dir: Path, param_name: str) -> None:
    """Two box plots — convergence_time.png/pdf and oversize_time.png/pdf —
    one box per parameter value. Only converged runs contribute to either
    box: oversize_time_s from a run that never converged reflects its whole
    log rather than the pre-convergence window, so it isn't comparable.
    `out_dir` is the category folder; plots are written under its figure/
    subfolder."""
    figure_dir = out_dir / "figure"
    figure_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "real_controller_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_param: dict[float, list[tuple[float, float]]] = defaultdict(list)
    for pv, conv, over in rows:
        if conv is not None:
            by_param[pv].append((conv, over))
    param_values = sorted(by_param)
    if not param_values:
        return
    labels = [f"{pv:g}" for pv in param_values]

    box_style = dict(
        widths=0.3,
        patch_artist=True,
        boxprops=dict(facecolor="#cde2fb", edgecolor="#2a78d6", linewidth=1.2),
        medianprops=dict(color="#184f95", linewidth=1.5),
        whiskerprops=dict(color="#2a78d6"),
        capprops=dict(color="#2a78d6"),
        flierprops=dict(markeredgecolor="#2a78d6"),
    )

    if _scale_exp_tag(out_dir):
        # match scalability_analysis.py's BOX_STYLE / 0.6-spacing box width
        box_style["widths"] = 0.6
        box_style["flierprops"] = dict(markeredgecolor="#2a78d6", markersize=4)

    conv_data = [[c for c, _ in by_param[pv]] for pv in param_values]
    if _scale_exp_tag(out_dir):
        conv_data = [_drop_far_outliers(d) for d in conv_data]
    fig, ax = plt.subplots(figsize=_single_sweep_figsize(param_name, out_dir), dpi=300)
    ax.boxplot(conv_data, tick_labels=labels, **box_style)
    _label_axes(ax, out_dir, param_name, "Convergence time (s)")
    if len(param_values) > 20:
        ax.tick_params(axis="x", labelsize=10)  # keep two-digit labels from running together
    ax.grid(linestyle="--", alpha=0.35, axis="y")
    fig.tight_layout()
    fig.savefig(figure_dir / f"{_figure_stem('convergence_time', out_dir)}.png")
    fig.savefig(figure_dir / f"{_figure_stem('convergence_time', out_dir)}.pdf")
    plt.close(fig)


def plot_mean_line(summary_rows: list[SummaryRow], out_dir: Path, param_name: str) -> None:
    """Line plot of mean convergence time vs. param_value with error bars
    showing the standard deviation (converged runs only — mean/std here
    come straight from SummaryRow, which is already convergence-only).
    Written as figure/mean_convergence_time.png/pdf under out_dir."""
    figure_dir = out_dir / "figure"
    figure_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "real_controller_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in summary_rows if r.mean_convergence_time_s is not None]
    if not rows:
        return
    rows.sort(key=lambda r: r.param_value)
    x = [r.param_value for r in rows]
    y = [r.mean_convergence_time_s for r in rows]
    yerr = [r.std_convergence_time_s or 0.0 for r in rows]

    fig, ax = plt.subplots(figsize=_single_sweep_figsize(param_name, out_dir), dpi=300)
    ax.errorbar(
        x, y, yerr=yerr,
        marker="o", markersize=5, linewidth=1.6, color="#2a78d6",
        ecolor="#2a78d6", elinewidth=1.2, capsize=4,
    )
    _label_axes(ax, out_dir, param_name, "Mean convergence time (s)")
    ax.grid(linestyle="--", alpha=0.35, axis="y")
    fig.tight_layout()
    fig.savefig(figure_dir / f"{_figure_stem('mean_convergence_time', out_dir)}.png")
    fig.savefig(figure_dir / f"{_figure_stem('mean_convergence_time', out_dir)}.pdf")
    plt.close(fig)


def plot_success_rate(summary_rows: list[SummaryRow], out_dir: Path, param_name: str) -> None:
    """Standalone line plot of convergence rate (%) vs. param_value.
    Written as figure/convergence_rate.png/pdf under out_dir."""
    figure_dir = out_dir / "figure"
    figure_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "real_controller_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(summary_rows, key=lambda r: r.param_value)
    if not rows:
        return
    x = [r.param_value for r in rows]
    rate = [r.convergence_rate * 100 for r in rows]

    fig, ax = plt.subplots(figsize=_single_sweep_figsize(param_name, out_dir), dpi=300)
    ax.plot(
        x, rate,
        marker="s", markersize=5, linewidth=1.6, color="#e07b1f", linestyle="--",
    )
    _label_axes(ax, out_dir, param_name, "Convergence rate (%)")
    ax.set_ylim(0, 100)
    ax.grid(linestyle="--", alpha=0.35, axis="y")
    fig.tight_layout()
    fig.savefig(figure_dir / f"{_figure_stem('convergence_rate', out_dir)}.png")
    fig.savefig(figure_dir / f"{_figure_stem('convergence_rate', out_dir)}.pdf")
    plt.close(fig)


def plot_combined_boxplots(
    alpha_rows: list[tuple[float, float | None, float]],
    beta_rows: list[tuple[float, float | None, float]],
    out_dir: Path,
    param1_name: str = "alpha",
    param2_name: str = "beta",
    fixed_value: float = 0.5,
) -> None:
    """Combined convergence_time box plot: the param1 sweep (param2 held at
    fixed_value) in blue side by side with the param2 sweep (param1 held at
    fixed_value) in orange, sharing one x-axis of parameter value — valid
    because both sweeps use the same parameter-value grid and only differ
    in which of param1/param2 is varied vs. held fixed. Only converged runs
    contribute to either box."""
    figure_dir = out_dir / "figure"
    figure_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "real_controller_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    def _by_param(rows: list[tuple[float, float | None, float]]) -> dict[float, list[tuple[float, float]]]:
        d: dict[float, list[tuple[float, float]]] = defaultdict(list)
        for pv, conv, over in rows:
            if conv is not None:
                d[pv].append((conv, over))
        return d

    alpha_by_param = _by_param(alpha_rows)
    beta_by_param = _by_param(beta_rows)
    values = sorted(set(alpha_by_param) | set(beta_by_param))
    if not values:
        return
    labels = [f"{v:g}" for v in values]
    positions = np.arange(len(values), dtype=float)
    width = 0.28
    offset = 0.17

    alpha_style = dict(
        widths=width,
        patch_artist=True,
        boxprops=dict(facecolor="#cde2fb", edgecolor="#2a78d6", linewidth=1.2),
        medianprops=dict(color="#184f95", linewidth=1.5),
        whiskerprops=dict(color="#2a78d6"),
        capprops=dict(color="#2a78d6"),
        flierprops=dict(markeredgecolor="#2a78d6", markersize=4),
    )
    beta_style = dict(
        widths=width,
        patch_artist=True,
        boxprops=dict(facecolor="#fbdcb0", edgecolor="#e07b1f", linewidth=1.2),
        medianprops=dict(color="#a35a12", linewidth=1.5),
        whiskerprops=dict(color="#e07b1f"),
        capprops=dict(color="#e07b1f"),
        flierprops=dict(markeredgecolor="#e07b1f", markersize=4),
    )
    legend_handles = [
        Patch(facecolor="#cde2fb", edgecolor="#2a78d6", label=f"{param2_name}={fixed_value:g}"),
        Patch(facecolor="#fbdcb0", edgecolor="#e07b1f", label=f"{param1_name}={fixed_value:g}"),
    ]

    def _draw(metric_idx: int, ylabel: str, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(6.5, 5.0), dpi=300)
        for i, v in enumerate(values):
            if v in alpha_by_param:
                data = [t[metric_idx] for t in alpha_by_param[v]]
                ax.boxplot([data], positions=[positions[i] - offset], **alpha_style)
            if v in beta_by_param:
                data = [t[metric_idx] for t in beta_by_param[v]]
                ax.boxplot([data], positions=[positions[i] + offset], **beta_style)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_xlabel("parameter value")
        ax.set_ylabel(ylabel)
        ax.grid(linestyle="--", alpha=0.35, axis="y")
        ax.legend(handles=legend_handles)
        fig.tight_layout()
        fig.savefig(figure_dir / f"{filename}.png")
        fig.savefig(figure_dir / f"{filename}.pdf")
        plt.close(fig)

    _draw(0, "Convergence time (s)", f"convergence_time_combined_{param1_name}_{param2_name}")
    _draw(1, "Oversize time (s)", f"oversize_time_combined_{param1_name}_{param2_name}")


def infer_fixed_value(sweep_dir: Path, param1_name: str, param2_name: str) -> float:
    """The shared midpoint value: the combo folder where param1's value ==
    param2's value (e.g. alpha_0.5_beta_0.5, or fr_10.0_fd_10.0)."""
    for p in sorted(sweep_dir.iterdir()):
        if not p.is_dir():
            continue
        m = COMBINED_DIR_RE.fullmatch(p.name)
        if not m:
            continue
        names_values = {m.group(1): float(m.group(2)), m.group(3): float(m.group(4))}
        if param1_name in names_values and param2_name in names_values:
            v1, v2 = names_values[param1_name], names_values[param2_name]
            if v1 == v2:
                return v1
    raise FileNotFoundError(
        f"No shared-midpoint folder (where {param1_name} == {param2_name}) found under {sweep_dir}"
    )


def combine_two_param_sweeps(
    out_root: Path, param1_name: str, param2_name: str, fixed_value: float = 0.5,
) -> Path:
    """Combine the latest param1-sweep and param2-sweep analyses already
    written under out_root (via a prior plain run of this script on each
    sweep) into one combined box plot. Returns the combined category folder."""
    dir1 = _latest_dir(out_root, f"real_controller_sweep_{param1_name}_")
    dir2 = _latest_dir(out_root, f"real_controller_sweep_{param2_name}_")
    if dir1 is None or dir2 is None:
        missing = param1_name if dir1 is None else param2_name
        raise FileNotFoundError(
            f"No real_controller_sweep_{missing}_* output found under {out_root}; "
            f"analyze the {missing} sweep first (see module docstring)."
        )
    rows1 = read_per_run(dir1 / "result" / "per_run_convergence_oversize.csv", param1_name)
    rows2 = read_per_run(dir2 / "result" / "per_run_convergence_oversize.csv", param2_name)

    combined_dir = out_root / f"real_controller_combined_{param1_name}_{param2_name}"
    plot_combined_boxplots(rows1, rows2, combined_dir, param1_name, param2_name, fixed_value)
    print(f"Combined: {param1_name}={dir1.name}  {param2_name}={dir2.name}")
    return combined_dir


def plot_all(
    per_run_rows: list[tuple[float, float | None, float]],
    summary_rows: list[SummaryRow],
    out_dir: Path,
    param_name: str,
    plot_range: tuple[float, float] | None = None,
) -> None:
    """Draw the three single-sweep figures. `plot_range` (lo, hi), inclusive,
    limits which parameter values are drawn; the CSVs always keep all of them."""
    if plot_range is not None:
        lo, hi = plot_range
        per_run_rows = [r for r in per_run_rows if lo <= r[0] <= hi]
        summary_rows = [r for r in summary_rows if lo <= r.param_value <= hi]
    plot_boxplots(per_run_rows, out_dir, param_name)
    plot_mean_line(summary_rows, out_dir, param_name)
    plot_success_rate(summary_rows, out_dir, param_name)


def write_outputs(results: list[RunResult], out_dir: Path, param_name: str,
                  plot_range: tuple[float, float] | None = None) -> None:
    """`out_dir` is the category folder; CSVs go under its result/ subfolder,
    plots (via plot_boxplots) under its figure/ subfolder."""
    result_dir = out_dir / "result"
    result_dir.mkdir(parents=True, exist_ok=True)

    per_run_path = result_dir / "per_run_convergence_oversize.csv"
    with open(per_run_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            param_name,
            f"{param_name}_dir",
            "run",
            "rows",
            "agents",
            "start_time_s",
            "end_time_s",
            "converged",
            "convergence_time_s",
            "oversize_time_s",
        ])
        for r in results:
            w.writerow([
                r.param_value,
                r.param_dir,
                r.run,
                r.rows,
                r.agents,
                _round_or_blank(r.start_time_s),
                _round_or_blank(r.end_time_s),
                r.convergence_time_s is not None,
                _round_or_blank(r.convergence_time_s),
                _round_or_blank(r.oversize_time_s),
            ])

    by_param: dict[float, list[RunResult]] = defaultdict(list)
    for result in results:
        by_param[result.param_value].append(result)

    summary_rows = []
    for param_value in sorted(by_param):
        group = by_param[param_value]
        converged_group = [r for r in group if r.convergence_time_s is not None]
        conv = [r.convergence_time_s for r in converged_group]
        over = [r.oversize_time_s for r in converged_group]
        summary_rows.append(SummaryRow(
            param_value=param_value,
            runs=len(group),
            converged_runs=len(conv),
            convergence_rate=len(conv) / len(group),
            mean_convergence_time_s=mean(conv) if conv else None,
            median_convergence_time_s=median(conv) if conv else None,
            std_convergence_time_s=pstdev(conv) if len(conv) > 1 else 0.0 if conv else None,
            mean_oversize_time_s=mean(over) if over else None,
            median_oversize_time_s=median(over) if over else None,
            std_oversize_time_s=pstdev(over) if len(over) > 1 else 0.0 if over else None,
        ))

    summary_path = result_dir / "summary_by_param.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            param_name,
            "runs",
            "converged_runs",
            "convergence_rate",
            "mean_convergence_time_s",
            "median_convergence_time_s",
            "std_convergence_time_s",
            "mean_oversize_time_s",
            "median_oversize_time_s",
            "std_oversize_time_s",
        ])
        for r in summary_rows:
            w.writerow([
                r.param_value,
                r.runs,
                r.converged_runs,
                round(r.convergence_rate, 6),
                _round_or_blank(r.mean_convergence_time_s),
                _round_or_blank(r.median_convergence_time_s),
                _round_or_blank(r.std_convergence_time_s),
                _round_or_blank(r.mean_oversize_time_s),
                _round_or_blank(r.median_oversize_time_s),
                _round_or_blank(r.std_oversize_time_s),
            ])
    print_summary(summary_rows, param_name)
    plot_all(
        [(r.param_value, r.convergence_time_s, r.oversize_time_s) for r in results],
        summary_rows, out_dir, param_name, plot_range,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect convergence time and oversize time from real-controller parameter-sweep runs, "
                    "using position-clustering of STAY-mode agents to determine group size."
    )
    parser.add_argument(
        "sweep_dir",
        nargs="?",
        type=Path,
        help="Sweep directory, e.g. data/sweep_alpha_202608211548 or data/sweep_beta_*. Defaults to newest data/sweep_alpha_*.",
    )
    parser.add_argument("--data-root", type=Path, default=_HERE / ".." / "data")
    parser.add_argument("--out-root", type=Path, default=_HERE / ".." / "output")
    parser.add_argument("--out-name", default=None, help="Output subdirectory name under --out-root.")
    parser.add_argument(
        "--combine", action="store_true",
        help="Skip single-sweep analysis; instead combine the latest already-analyzed "
             "real_controller_sweep_<param1>_* and real_controller_sweep_<param2>_* outputs "
             "under --out-root into one combined box plot (convergence_time_combined) under "
             "real_controller_combined_<param1>_<param2>/. Param names/fixed value come from "
             "--combine-param1/--combine-param2/--combine-fixed-value.",
    )
    parser.add_argument("--combine-param1", default="alpha", help="First param name for --combine.")
    parser.add_argument("--combine-param2", default="beta", help="Second param name for --combine.")
    parser.add_argument("--combine-fixed-value", type=float, default=0.5, help="Fixed value for --combine.")
    parser.add_argument("--force", action="store_true", help="Re-read raw CSVs even if output CSVs already exist.")
    parser.add_argument("--param-name", default=None, help="Swept parameter folder prefix, e.g. alpha or beta.")
    parser.add_argument("--target-size", type=float, default=14.0)
    parser.add_argument("--plot-range", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                        help="Only draw parameter values in [LO, HI]; CSVs keep every value.")
    parser.add_argument("--num-agents", type=int, default=None, help="Expected total agents. Defaults to discovered ids.")
    parser.add_argument("--run-limit", type=int, default=None, help="Analyze only the first N runs per parameter value.")
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help="Number of worker processes. Defaults to min(cpu_count, number of runs). Use 1 for serial.",
    )
    parser.add_argument("--id-col", type=int, default=0, help="Zero-based id column index.")
    parser.add_argument("--time-col", type=int, default=1, help="Zero-based time column index.")
    parser.add_argument("--x-col", type=int, default=5, help="Zero-based x-position column index.")
    parser.add_argument("--y-col", type=int, default=6, help="Zero-based y-position column index.")
    parser.add_argument("--mode-col", type=int, default=10, help="Zero-based mode column index.")
    parser.add_argument("--stay-mode", default="STAY", help="Mode value that marks an agent as parked/clusterable.")
    parser.add_argument(
        "--cluster-radius",
        type=float,
        default=1.0,
        help="Position-clustering radius: STAY agents within this Euclidean distance are one cluster/group.",
    )
    return parser.parse_args()


def run_one_analysis(
    runs: list[tuple[float, Path, Path]],
    param_name: str,
    out_dir: Path,
    args: argparse.Namespace,
    sweep_dir: Path,
) -> None:
    """Shared body for one param_name's worth of analysis: given already-
    discovered runs, either reuse cached CSV outputs under out_dir or
    re-read the raw logs, then write CSVs + box plots + console summary."""
    result_dir = out_dir / "result"
    figure_dir = out_dir / "figure"
    per_run_path = result_dir / "per_run_convergence_oversize.csv"
    summary_path = result_dir / "summary_by_param.csv"

    if not args.force and per_run_path.exists() and summary_path.exists():
        summary_rows = read_summary(summary_path, param_name)
        print_summary(summary_rows, param_name)
        plot_all(read_per_run(per_run_path, param_name), summary_rows, out_dir, param_name,
                 args.plot_range)
        print(f"Input : {sweep_dir}")
        print(f"Output: {out_dir}")
        print("Cache : existing CSV outputs found; skipped raw data reading")
        print(f"CSV   : {per_run_path}")
        print(f"CSV   : {summary_path}")
        print(f"Plot  : {figure_dir / 'convergence_time.png'}")
        print(f"Plot  : {figure_dir / 'convergence_time.pdf'}")
        print(f"Plot  : {figure_dir / 'mean_convergence_time.png'}")
        print(f"Plot  : {figure_dir / 'mean_convergence_time.pdf'}")
        print(f"Plot  : {figure_dir / 'convergence_rate.png'}")
        print(f"Plot  : {figure_dir / 'convergence_rate.pdf'}")
        return

    runs = limit_runs_per_param(runs, args.run_limit)
    jobs = [
        (
            param_value,
            param_dir,
            run_path,
            args.target_size,
            args.num_agents,
            args.id_col,
            args.time_col,
            args.x_col,
            args.y_col,
            args.mode_col,
            args.cluster_radius,
            args.stay_mode,
        )
        for param_value, param_dir, run_path in runs
    ]
    labels = [
        f"{param_name}={param_value:g} run={run_path.name}"
        for param_value, _param_dir, run_path in runs
    ]
    processes = args.processes or min(os.cpu_count() or 1, len(jobs))

    print(f"Analyzing {len(jobs)} runs with {processes} process(es)  "
          f"(size = position clustering of {args.stay_mode} agents, radius={args.cluster_radius})", flush=True)
    if processes == 1:
        results = []
        for i, (job, label) in enumerate(zip(jobs, labels), start=1):
            print(f"[{i}/{len(jobs)}] {label}", flush=True)
            results.append(analyze_run_job(job))
    else:
        results = []
        with futures.ProcessPoolExecutor(max_workers=processes) as pool:
            future_to_label = {
                pool.submit(analyze_run_job, job): label
                for job, label in zip(jobs, labels)
            }
            for i, fut in enumerate(futures.as_completed(future_to_label), start=1):
                label = future_to_label[fut]
                result = fut.result()
                results.append(result)
                print(
                    f"[{i}/{len(jobs)}] done {label} "
                    f"rows={result.rows} converged={result.convergence_time_s is not None}",
                    flush=True,
                )

    results.sort(key=lambda r: (r.param_value, r.run))
    write_outputs(results, out_dir, param_name, args.plot_range)

    converged = sum(1 for r in results if r.convergence_time_s is not None)
    empty = sum(1 for r in results if r.rows == 0)
    print(f"Input : {sweep_dir}")
    print(f"Output: {out_dir}")
    print(f"Runs  : {len(results)} ({converged} converged, {empty} with no parsed rows)")
    print(f"CSV   : {per_run_path}")
    print(f"CSV   : {summary_path}")
    print(f"Plot  : {figure_dir / 'convergence_time.png'}")
    print(f"Plot  : {figure_dir / 'convergence_time.pdf'}")
    print(f"Plot  : {figure_dir / 'mean_convergence_time.png'}")
    print(f"Plot  : {figure_dir / 'mean_convergence_time.pdf'}")
    print(f"Plot  : {figure_dir / 'convergence_rate.png'}")
    print(f"Plot  : {figure_dir / 'convergence_rate.pdf'}")


def main() -> None:
    args = parse_args()

    if args.combine:
        combined_dir = combine_two_param_sweeps(
            args.out_root, args.combine_param1, args.combine_param2, args.combine_fixed_value,
        )
        figure_dir = combined_dir / "figure"
        conv_name = f"convergence_time_combined_{args.combine_param1}_{args.combine_param2}"
        over_name = f"oversize_time_combined_{args.combine_param1}_{args.combine_param2}"
        print(f"Output: {combined_dir}")
        print(f"Plot  : {figure_dir / f'{conv_name}.png'}")
        print(f"Plot  : {figure_dir / f'{conv_name}.pdf'}")
        print(f"Plot  : {figure_dir / f'{over_name}.png'}")
        print(f"Plot  : {figure_dir / f'{over_name}.pdf'}")
        return

    sweep_dir = args.sweep_dir or _latest_dir(args.data_root, "sweep_alpha_")
    if sweep_dir is None:
        raise FileNotFoundError(f"No sweep_alpha_* directory found under {args.data_root}")
    sweep_dir = sweep_dir.resolve()

    if is_combined_layout(sweep_dir):
        # Merged two-parameter sweep layout: <p1>_<v1>_<p2>_<v2>/run_*/bot*.csv,
        # combos sharing a midpoint where both params equal the same fixed
        # value. Slice it into a param1 view (param2 held fixed) and a
        # param2 view (param1 held fixed), analyze each independently, then
        # overlay them automatically.
        param1_name, param2_name = infer_combined_param_names(sweep_dir)
        fixed_value = infer_fixed_value(sweep_dir, param1_name, param2_name)
        print(f"Detected combined {param1_name}/{param2_name} layout under {sweep_dir} "
              f"(shared midpoint {param1_name}={param2_name}={fixed_value:g})")

        runs1 = discover_combined_runs(sweep_dir, param1_name, param2_name, fixed_value)
        out1 = args.out_root / (args.out_name or f"real_controller_sweep_{param1_name}_{sweep_dir.name}")
        print(f"\n=== {param1_name} view ({param2_name}={fixed_value:g} fixed): {len(runs1)} runs ===")
        run_one_analysis(runs1, param1_name, out1, args, sweep_dir)

        runs2 = discover_combined_runs(sweep_dir, param2_name, param1_name, fixed_value)
        out2 = args.out_root / (args.out_name or f"real_controller_sweep_{param2_name}_{sweep_dir.name}")
        print(f"\n=== {param2_name} view ({param1_name}={fixed_value:g} fixed): {len(runs2)} runs ===")
        run_one_analysis(runs2, param2_name, out2, args, sweep_dir)

        print(f"\n=== combined {param1_name}/{param2_name} overlay ===")
        combined_dir = combine_two_param_sweeps(args.out_root, param1_name, param2_name, fixed_value)
        figure_dir = combined_dir / "figure"
        conv_name = f"convergence_time_combined_{param1_name}_{param2_name}"
        over_name = f"oversize_time_combined_{param1_name}_{param2_name}"
        print(f"Output: {combined_dir}")
        print(f"Plot  : {figure_dir / f'{conv_name}.png'}")
        print(f"Plot  : {figure_dir / f'{conv_name}.pdf'}")
        print(f"Plot  : {figure_dir / f'{over_name}.png'}")
        print(f"Plot  : {figure_dir / f'{over_name}.pdf'}")
        return

    param_name = infer_param_name(sweep_dir, args.param_name)
    out_name = args.out_name or f"real_controller_{sweep_dir.name}"
    out_dir = args.out_root / out_name
    runs = discover_runs(sweep_dir, param_name)
    run_one_analysis(runs, param_name, out_dir, args, sweep_dir)


if __name__ == "__main__":
    main()

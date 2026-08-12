#!/usr/bin/env python3
"""
simulate_convergence_physical.py
---------------------------------
Estimates decisions-per-agent and wall-clock convergence time.

Travel model (no 2D positions):
  FUSION  : t ~ Exp(t_fusion_mean)
  FISSION : t ~ Exp(t_fiss_mean)
  STAY    : T = T_max * (1 - alpha * max(0, (n*-n)/n*))
  p_split : min(1, beta * max(0, (n-n*)/n*))

Key behaviours from C++ controller:
  - Once FUSION/FISSION travel starts, stay-timers are ignored (in_travel)
  - Traveling agents stay in group_members → selectable as join-targets
  - On arrival: join unconditionally, no size check (the target may already
    be full or overfull by the time we get there — that's exactly the
    overshoot the sim is meant to measure)
  - Group-size change resets all members' stay-timers

Decision-vs-perception timing: target selection references size exactly
once — at decision time, an idle agent filters candidate groups by their
current *perceived* size (perceived_n, itself a delayed/lagged belief) and
picks a random member of a random group that still looks like it has room.
That's the only lookup: no size check happens again during travel or on
arrival, the agent just walks in and joins whatever it finds there (which is
exactly how overshoot can still occur — perceived_n can lag behind the true
size).

Event tags: D=decide  A=arrive  F=fission_done  S=stay_done
Version counters: ver_stay (S events) / ver_travel (D,A,F events)
  → timer resets never accidentally cancel travel events
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import heapq, csv
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PhysSimParams:
    num_agents:    int   = 30
    n_star:        int   = 15
    t_fusion_mean: float = 10.0
    t_fiss_mean:   float = 6.0
    T_max:         float = 30.0
    alpha_ctrl:    float = 0.5
    beta_ctrl:     float = 0.5
    t_percep_mean:    float = 0.05  # mean delay per new member before size is perceived [s]
    t_overfull_check: float = 0.5   # re-check interval when n > n* [s]
    max_dec:       int   = 200
    max_time:      float = 7200.0
    num_trials:    int   = 50
    out_dir:       str   = "../output/convergence_physical"


def _T(n, p):
    return max(0.0, p.T_max * (1.0 - p.alpha_ctrl * max(0.0, (p.n_star - n) / p.n_star)))

def _ps(n, p):
    return min(1.0, p.beta_ctrl * max(0.0, (n - p.n_star) / p.n_star))


def simulate_one_trial(p: PhysSimParams, rng: np.random.Generator) -> dict:
    N, tie = p.num_agents, 0
    agent_gid = np.arange(N, dtype=int)
    grp: dict[int, set] = {i: {i} for i in range(N)}
    nxt = N

    dec   = np.zeros(N, dtype=int)
    done  = np.zeros(N, dtype=bool)
    vs    = np.zeros(N, dtype=int)   # ver_stay
    vt    = np.zeros(N, dtype=int)   # ver_travel
    going = np.zeros(N, dtype=bool)  # in_travel (FUSION or FISSION)
    ftgt  = np.full(N, -1, dtype=int)  # fusion target agent_id (-1 if not fusing)

    # ── Perception state (per group) ──────────────────────────────────────
    # perceived_n[gid] : what group members currently believe the size to be.
    # Every membership change (join OR leave) schedules a 'P' event that moves
    # perceived_n one step towards the true size after a random delay;
    # updates are sequential (queue-based), not parallel, and symmetric in
    # both directions (noticing a departure is just as delayed as noticing a
    # new arrival).
    perceived_n: dict[int, int]   = {i: 1 for i in range(N)}
    percep_q_t:  dict[int, float] = {i: 0.0 for i in range(N)}  # end of perception queue
    percep_ver:  dict[int, int]   = {i: 0 for i in range(N)}    # stale-check version

    overshoot_entry:    dict[int, float] = {}  # gid → time when actual size first exceeded n*
    overshoot_durations: list            = []  # duration of each completed overshoot event

    pq: list = []

    def _push(t, a, tag, ex, v):
        nonlocal tie
        heapq.heappush(pq, (t, tie, a, tag, ex, v))
        tie += 1

    def decide(t, a):
        going[a] = False
        vt[a] += 1
        _push(t, a, 'D', 0, vt[a])

    def arrive(t, a, g):
        going[a] = True
        ftgt[a] = g
        vt[a] += 1
        _push(t, a, 'A', g, vt[a])

    def fission(t, a):
        going[a] = True
        ftgt[a] = -1
        vt[a] += 1
        _push(t, a, 'F', 0, vt[a])

    def stay(t, a):
        vs[a] += 1
        _push(t, a, 'S', 0, vs[a])

    def stale(a, tag, v):
        return v != (vs[a] if tag == 'S' else vt[a])

    def free_groups(a):
        """Candidate fusion targets: other groups whose *perceived* size
        still looks like it has room — checked once, here, at decision
        time. Never re-checked again during travel or on arrival."""
        my = agent_gid[a]
        return [g for g, _ in grp.items() if g != my and 0 < perceived_n.get(g, 0) < p.n_star]

    def singleton(a):
        nonlocal nxt
        g = nxt
        nxt += 1
        grp[g] = {a}
        agent_gid[a] = g
        perceived_n[g] = 1
        percep_q_t[g]  = 0.0
        percep_ver[g]  = 0

    def _check_overshoot_exit(g, t):
        if g in overshoot_entry and len(grp.get(g, set())) <= p.n_star:
            overshoot_durations.append(t - overshoot_entry.pop(g))

    def leave(a, t):
        old = agent_gid[a]
        grp[old].discard(a)
        if not grp[old]:
            del grp[old]
            perceived_n.pop(old, None)
            percep_q_t.pop(old, None)
            percep_ver.pop(old, None)
            overshoot_entry.pop(old, None)
        else:
            percep_ver[old] = percep_ver.get(old, 0) + 1
            _check_overshoot_exit(old, t)
            schedule_perception(old, t)
        return old

    def join(a, g, t):
        if agent_gid[a] == g:
            return
        old = agent_gid[a]
        grp[old].discard(a)
        if not grp[old]:
            del grp[old]
            perceived_n.pop(old, None)
            percep_q_t.pop(old, None)
            percep_ver.pop(old, None)
            overshoot_entry.pop(old, None)
        else:
            percep_ver[old] = percep_ver.get(old, 0) + 1
            _check_overshoot_exit(old, t)
            schedule_perception(old, t)
        grp[g].add(a)
        agent_gid[a] = g
        if len(grp[g]) > p.n_star and g not in overshoot_entry:
            overshoot_entry[g] = t

    def schedule_perception(g, t):
        """Queue one sequential ±1 perception step for group g (moves
        perceived_n one step closer to the true size, whichever direction)."""
        fire_t = max(percep_q_t.get(g, 0.0), t) + rng.exponential(p.t_percep_mean)
        percep_q_t[g] = fire_t
        _push(fire_t, g, 'P', g, 0)

    def pn(a):
        """Perceived group size for agent a."""
        return perceived_n.get(agent_gid[a], 1)

    def reset_timers(g, t, n_new):
        for m in list(grp.get(g, set())):
            if done[m]:
                continue
            if n_new == p.n_star:
                stay(t + p.T_max * 2, m)
            elif n_new > p.n_star:
                stay(t + p.t_overfull_check, m)
            else:
                stay(t + _T(n_new, p), m)

    snaps, wall = [], 0.0
    for i in range(N):
        decide(0.0, i)

    for _ in range(N * p.max_dec * 8):
        if not pq:
            break
        t, _, a, tag, ex, v = heapq.heappop(pq)
        if t > p.max_time:
            break
        wall = t

        # ── P: group perception update (not per-agent) ────────────────────
        if tag == 'P':
            g = ex
            if g not in perceived_n:
                continue                        # group dissolved
            true_n = len(grp.get(g, set()))
            if perceived_n[g] < true_n:
                perceived_n[g] += 1
                reset_timers(g, t, perceived_n[g])
            elif perceived_n[g] > true_n:
                perceived_n[g] -= 1
                reset_timers(g, t, perceived_n[g])
            continue

        if done[a] or stale(a, tag, v):
            continue

        if tag == 'F':                          # fission done → re-decide
            decide(t, a)

        elif tag == 'S':                        # stay timer fired
            if going[a]:
                continue                        # committed to travel, ignore
            n = pn(a)
            if n == 1:
                decide(t, a)                    # group dissolved → RANDOM_WALK
                continue
            dec[a] += 1
            snaps.append((t, sorted([len(m) for m in grp.values()], reverse=True)))
            if n == p.n_star:
                done[a] = True
                if done.all():
                    wall = t
                    break
            elif n > p.n_star:
                # Overfull: probabilistic fission
                if rng.random() < _ps(n, p):
                    leave(a, t)
                    singleton(a)
                    fission(t + rng.exponential(p.t_fiss_mean), a)
                else:
                    stay(t + p.t_overfull_check, a)
            else:
                # n < n*: wait time already elapsed (S fired after T(n) delay)
                leave(a, t)
                singleton(a)
                fission(t + rng.exponential(p.t_fiss_mean), a)

        elif tag == 'A':                        # fusion travel complete → join target agent's group, no size check
            going[a] = False
            ftgt[a] = -1
            target_agent = ex                   # target is an individual, always exists
            g = agent_gid[target_agent]
            dec[a] += 1
            snaps.append((t, sorted([len(m) for m in grp.values()], reverse=True)))
            join(a, g, t)
            schedule_perception(g, t)

        elif tag == 'D':                        # RANDOM_WALK: select a target individual
            n = pn(a)
            dec[a] += 1
            snaps.append((t, sorted([len(m) for m in grp.values()], reverse=True)))
            if n == p.n_star:
                done[a] = True
                if done.all():
                    wall = t
                    break
            elif n > p.n_star:
                stay(t, a)
            elif n > 1:
                # Joined by someone during fission travel → wait in this group
                stay(t + _T(n, p), a)
            else:
                # n == 1: true loner → pick a target individual
                tgts = free_groups(a)
                if tgts:
                    chosen_grp = int(rng.choice(tgts))
                    target_agent = int(rng.choice(list(grp[chosen_grp])))
                    # Chain: if target is itself fusing, follow to their target agent
                    if going[target_agent] and ftgt[target_agent] != -1:
                        target_agent = ftgt[target_agent]
                    ftgt[a] = target_agent
                    arrive(t + rng.exponential(p.t_fusion_mean), a, target_agent)
                else:
                    decide(t + _T(1, p), a)

    return {
        'decisions': dec, 'total_time': wall,
        'converged': bool(done.all()), 'snaps': snaps,
        'overshoot_count':    len(overshoot_durations),
        'overshoot_dur_mean': float(np.mean(overshoot_durations)) if overshoot_durations else 0.0,
        'overshoot_time':     float(sum(overshoot_durations)),
    }


def run_trials(p: PhysSimParams, seed: int = 0) -> dict:
    rng0 = np.random.default_rng(seed)
    decs, times, n_ok = [], [], 0
    print(f"N={p.num_agents} n*={p.n_star} α={p.alpha_ctrl} β={p.beta_ctrl} "
          f"t_fuse~Exp({p.t_fusion_mean}) T_max={p.T_max}  [{p.num_trials} trials]")
    for i in range(p.num_trials):
        r = simulate_one_trial(p, np.random.default_rng(int(rng0.integers(0, 2**31))))
        if r['converged']:
            n_ok += 1
            decs.append(float(r['decisions'].mean()))
            times.append(r['total_time'])
        if (i + 1) % 10 == 0:
            print(f"  {i+1:3d}  dec={decs[-1]:.1f}  t={times[-1]:.0f}s  ok={r['converged']}" if decs else
                  f"  {i+1:3d}  (no converged trials yet)  ok={r['converged']}")
    return {
        'mean_dec': float(np.mean(decs)) if decs else float('nan'),
        'std_dec':  float(np.std(decs))  if decs else float('nan'),
        'mean_t':   float(np.mean(times)) if times else float('nan'),
        'std_t':    float(np.std(times))  if times else float('nan'),
        'conv_rate': n_ok / p.num_trials, 'all_dec': decs, 'all_t': times,
    }


def save_and_plot(s: dict, p: PhysSimParams):
    out = Path(p.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        for k, v in s.items():
            if not isinstance(v, list):
                w.writerow([k, round(v, 3)])

    with open(out / "per_trial.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial", "decisions", "time_s"])
        for i, (d, t) in enumerate(zip(s['all_dec'], s['all_t'])):
            w.writerow([i + 1, round(d, 2), round(t, 1)])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), dpi=120)
    for ax, data, xlabel, color in [
        (ax1, s['all_dec'], "Decisions / agent", 'C0'),
        (ax2, s['all_t'],   "Convergence time (s)", 'C1'),
    ]:
        mn, md = float(np.mean(data)), float(np.median(data))
        ax.hist(data, bins=20, color=color, edgecolor='white', alpha=0.85)
        ax.axvline(mn, color='red',    linestyle='--', label=f"mean {mn:.1f}")
        ax.axvline(md, color='orange', linestyle=':',  label=f"median {md:.1f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        ax.legend()
        ax.grid(linestyle='--', alpha=0.4)
    ax1.set_title(f"N={p.num_agents}, n*={p.n_star}, α={p.alpha_ctrl}, β={p.beta_ctrl} "
                  f"({s['conv_rate']*100:.0f}% converged)")
    ax2.set_title(f"t_fuse~Exp({p.t_fusion_mean}s), T_max={p.T_max}s")
    fig.tight_layout()
    fig.savefig(out / "convergence_stats.png")
    plt.close()

    print(f"\n→ {out}")
    print(f"  decisions : {s['mean_dec']:.1f} ± {s['std_dec']:.1f}")
    print(f"  time      : {s['mean_t']:.0f} s ± {s['std_t']:.0f} s")
    print(f"  converged : {s['conv_rate']*100:.0f}%")


def sweep_alpha_beta(base: PhysSimParams, values=None, trials_per_cell: int = 20, seed: int = 0):
    """
    Sweep alpha ∈ values and beta ∈ values, plot two heatmaps:
      left  — mean decisions per agent
      right — mean convergence time (s)
    """
    from copy import deepcopy
    if values is None:
        values = [round(v * 0.1, 1) for v in range(1, 11)]   # 0.1 … 1.0

    n = len(values)
    mat_dec       = np.full((n, n), np.nan)
    mat_time      = np.full((n, n), np.nan)
    mat_conv      = np.full((n, n), np.nan)
    mat_os_count  = np.full((n, n), np.nan)
    mat_os_dur    = np.full((n, n), np.nan)
    mat_os_total  = np.full((n, n), np.nan)

    rng0 = np.random.default_rng(seed)
    total = n * n
    done  = 0

    print(f"Sweeping α×β  {n}×{n} grid  ({trials_per_cell} trials/cell) …")
    for i, alpha in enumerate(values):
        for j, beta in enumerate(values):
            p2 = deepcopy(base)
            p2.alpha_ctrl = alpha
            p2.beta_ctrl  = beta
            p2.num_trials = trials_per_cell

            decs, times, n_ok = [], [], 0
            os_counts, os_durs, os_totals = [], [], []
            for _ in range(trials_per_cell):
                r = simulate_one_trial(p2, np.random.default_rng(int(rng0.integers(0, 2**31))))
                if r['converged']:
                    n_ok += 1
                    decs.append(float(r['decisions'].mean()))
                    times.append(r['total_time'])
                os_counts.append(r['overshoot_count'])
                os_durs.append(r['overshoot_dur_mean'])
                os_totals.append(r['overshoot_time'])

            mat_dec[i, j]      = float(np.mean(decs))      if decs      else np.nan
            mat_time[i, j]     = float(np.mean(times))     if times     else np.nan
            mat_conv[i, j]     = n_ok / trials_per_cell
            mat_os_count[i, j] = float(np.mean(os_counts))
            mat_os_dur[i, j]   = float(np.mean([d for d in os_durs if d > 0])) if any(d > 0 for d in os_durs) else 0.0
            mat_os_total[i, j] = float(np.mean(os_totals))
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{total}  α={alpha:.1f} β={beta:.1f}  "
                      f"t={mat_time[i,j]:.0f}s  os_cnt={mat_os_count[i,j]:.1f}  "
                      f"os_dur={mat_os_dur[i,j]:.2f}s  conv={mat_conv[i,j]*100:.0f}%")

    # ── save CSVs ──────────────────────────────────────────────────────────
    out = Path(base.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, mat in [
        ("sweep_decisions",       mat_dec),
        ("sweep_time",            mat_time),
        ("sweep_conv_rate",       mat_conv),
        ("sweep_overshoot_count", mat_os_count),
        ("sweep_overshoot_dur",   mat_os_dur),
        ("sweep_overshoot_total", mat_os_total),
    ]:
        with open(out / f"{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["alpha\\beta"] + values)
            for i, alpha in enumerate(values):
                w.writerow([alpha] + [round(mat[i, j], 3) for j in range(n)])

    suptitle = (f"N={base.num_agents}, n*={base.n_star}, "
                f"t_fuse~Exp({base.t_fusion_mean}s), T_max={base.T_max}s  "
                f"({trials_per_cell} trials/cell)")
    tick_labels = [f"{v:.1f}" for v in values]

    def _heatmap(specs, show_title=True):
        for ax, mat, title, cmap, fmt in specs:
            im = ax.imshow(mat, origin="lower", cmap=cmap, aspect="auto")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xticks(range(n))
            ax.set_xticklabels(tick_labels)
            ax.set_yticks(range(n))
            ax.set_yticklabels(tick_labels)
            ax.set_xlabel("β  (split probability scale)")
            ax.set_ylabel("α  (waiting-time scale)")
            if show_title:
                ax.set_title(title)
            vmax = np.nanmax(mat)
            for ii in range(n):
                for jj in range(n):
                    val = mat[ii, jj]
                    if not np.isnan(val):
                        ax.text(jj, ii, fmt.format(val), ha="center", va="center",
                                fontsize=6.5, color="black" if val < vmax * 0.7 else "white")

    # figure 1: convergence metrics
    fig1, axes1 = plt.subplots(1, 2, figsize=(13, 5), dpi=130)
    _heatmap([
        (axes1[0], mat_dec,  "Mean decisions / agent",    "YlOrRd", "{:.0f}"),
        (axes1[1], mat_time, "Mean convergence time (s)", "YlOrRd", "{:.0f}"),
    ])
    fig1.suptitle(suptitle, fontsize=11)
    fig1.tight_layout()
    fig1.savefig(out / "sweep_alpha_beta.png")
    plt.close(fig1)

    # figure 1b: standalone convergence-time heatmap (paper-ready, vector PDF, no title)
    fig1b, ax1b = plt.subplots(figsize=(6.5, 5.5), dpi=150)
    _heatmap([(ax1b, mat_time, "Mean convergence time (s)", "YlOrRd", "{:.0f}")], show_title=False)
    fig1b.tight_layout()
    fig1b.savefig(out / "convergence_time.pdf")
    plt.close(fig1b)

    # figure 2: overshoot metrics
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5), dpi=130)
    _heatmap([
        (axes2[0], mat_os_count, "Mean overshoot count / trial",      "Blues",  "{:.1f}"),
        (axes2[1], mat_os_dur,   "Mean overshoot duration (s)",        "Oranges","{:.2f}"),
        (axes2[2], mat_os_total, "Mean total overshoot time / trial (s)", "Reds", "{:.1f}"),
    ])
    fig2.suptitle(suptitle, fontsize=11)
    fig2.tight_layout()
    fig2.savefig(out / "sweep_overshoot.png")
    plt.close(fig2)

    # figure 2b: standalone overshoot-time heatmap (paper-ready, vector PDF, no title)
    fig2b, ax2b = plt.subplots(figsize=(6.5, 5.5), dpi=150)
    _heatmap([(ax2b, mat_os_total, "Mean total overshoot time / trial (s)", "Reds", "{:.1f}")], show_title=False)
    fig2b.tight_layout()
    fig2b.savefig(out / "overshoot_time.pdf")
    plt.close(fig2b)

    print(f"\n[Sweep] → {out / 'sweep_alpha_beta.png'}")
    print(f"[Sweep] → {out / 'convergence_time.pdf'}")
    print(f"[Sweep] → {out / 'sweep_overshoot.png'}")
    print(f"[Sweep] → {out / 'overshoot_time.pdf'}")


def sweep_fusion_fission_time(base: PhysSimParams, fusion_values=None, fiss_values=None,
                               trials_per_cell: int = 100, seed: int = 0):
    """
    Sweep t_fusion_mean (FUSION travel time) x t_fiss_mean (FISSION travel
    time) — both Exp(mean), same form, same default grid — holding
    alpha_ctrl / beta_ctrl fixed at `base`'s values. These two are physical
    travel-time assumptions (not controller-policy knobs like alpha/beta),
    so they get their own combined heatmap rather than joining the alpha x
    beta grid. Plots mean convergence time as a standalone, paper-ready
    vector PDF (no title, matching convergence_time.pdf / overshoot_time.pdf).
    """
    from copy import deepcopy
    if fusion_values is None:
        fusion_values = [3.0, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0]
    if fiss_values is None:
        fiss_values = [3.0, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0]

    n_rows, n_cols = len(fusion_values), len(fiss_values)
    mat_time = np.full((n_rows, n_cols), np.nan)
    mat_dec  = np.full((n_rows, n_cols), np.nan)
    mat_conv = np.full((n_rows, n_cols), np.nan)

    rng0 = np.random.default_rng(seed)
    total = n_rows * n_cols
    done = 0

    print(f"Sweeping t_fusion_mean x t_fiss_mean  {n_rows}x{n_cols} grid  "
          f"({trials_per_cell} trials/cell) …")
    for i, tf in enumerate(fusion_values):
        for j, tfi in enumerate(fiss_values):
            p2 = deepcopy(base)
            p2.t_fusion_mean = tf
            p2.t_fiss_mean = tfi
            p2.num_trials = trials_per_cell

            decs, times, n_ok = [], [], 0
            for _ in range(trials_per_cell):
                r = simulate_one_trial(p2, np.random.default_rng(int(rng0.integers(0, 2**31))))
                if r['converged']:
                    n_ok += 1
                    decs.append(float(r['decisions'].mean()))
                    times.append(r['total_time'])

            mat_time[i, j] = float(np.mean(times)) if times else np.nan
            mat_dec[i, j]  = float(np.mean(decs))  if decs  else np.nan
            mat_conv[i, j] = n_ok / trials_per_cell
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{total}  t_fusion_mean={tf:.1f}s  t_fiss_mean={tfi:.1f}s  "
                      f"t={mat_time[i, j]:.0f}s  conv={mat_conv[i, j]*100:.0f}%")

    out = Path(base.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, mat in [
        ("sweep_fusion_fission_time", mat_time),
        ("sweep_fusion_fission_dec",  mat_dec),
        ("sweep_fusion_fission_conv", mat_conv),
    ]:
        with open(out / f"{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_fusion_mean\\t_fiss_mean"] + fiss_values)
            for i, tf in enumerate(fusion_values):
                w.writerow([tf] + [round(mat[i, j], 3) for j in range(n_cols)])

    x_labels = [f"{tfi:.1f}" for tfi in fiss_values]
    y_labels = [f"{tf:.1f}" for tf in fusion_values]

    # Values span ~30x (hundreds to ~13000s) — a linear color scale crushes
    # almost the whole grid into the same pale color, so use a log norm to
    # keep the gradient visible across the full range. vmin/vmax are pinned
    # to the data's actual range (not e.g. 10^0) so the color gradient isn't
    # wasted on decades nothing in the grid ever reaches.
    from matplotlib.colors import LogNorm
    norm = LogNorm(vmin=np.nanmin(mat_time), vmax=np.nanmax(mat_time))

    fig, ax = plt.subplots(figsize=(7.0, 5.5), dpi=150)
    im = ax.imshow(mat_time, origin="lower", cmap="YlOrRd", aspect="auto", norm=norm)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # Bare powers of ten (10^3, 10^4) that fall within the data's actual
    # range — built and labeled manually (rather than via
    # set_major_formatter) because LogFormatterSciNotation silently drops
    # labels when attached to an axis in some cases.
    from matplotlib.ticker import LogFormatterSciNotation, LogLocator
    fmt = LogFormatterSciNotation(base=10, labelOnlyBase=False)
    raw_ticks = LogLocator(base=10, subs=(1.0,)).tick_values(norm.vmin, norm.vmax)
    cbar_ticks = [t for t in raw_ticks if norm.vmin <= t <= norm.vmax]
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([fmt(t) for t in cbar_ticks])
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("t_fiss_mean  (s)")
    ax.set_ylabel("t_fusion_mean  (s)")
    for ii in range(n_rows):
        for jj in range(n_cols):
            val = mat_time[ii, jj]
            if not np.isnan(val):
                # threshold on the log-normalized position, not the raw value
                ax.text(jj, ii, f"{val:.0f}", ha="center", va="center",
                         fontsize=6.5, color="black" if norm(val) < 0.7 else "white")
    fig.tight_layout()
    fig.savefig(out / "fusion_fission_time.pdf")
    plt.close(fig)

    print(f"\n[Sweep] → {out / 'fusion_fission_time.pdf'}")
    return mat_time, mat_dec, mat_conv


if __name__ == "__main__":
    p = PhysSimParams(
        num_agents=42,
        n_star=14,
        t_fusion_mean=15.0,
        t_fiss_mean=3.0,     # shortened from 6.0 (the old Uniform(4,8) midpoint) —
                             # sensitivity analysis showed this (and not
                             # t_fusion_mean/T_max) is the clean lever for lowering
                             # convergence time without destabilizing the sim
        T_max=30.0,
        alpha_ctrl=0.5,
        beta_ctrl=0.5,
        num_trials=500,
        max_dec=200,
        max_time=100000.0,
        out_dir="../output/convergence_physical",
    )
    save_and_plot(run_trials(p, seed=0), p)
    sweep_alpha_beta(p, trials_per_cell=100, seed=0)

    # Combined t_fusion_mean x t_fiss_mean sensitivity figure, swept around
    # the *original* t_fiss_mean=6.0 physical baseline — independent of the
    # shortened t_fiss_mean now used as the default above.
    p_ref = PhysSimParams(
        num_agents=42,
        n_star=14,
        t_fusion_mean=15.0,
        t_fiss_mean=6.0,
        T_max=30.0,
        alpha_ctrl=0.5,
        beta_ctrl=0.5,
        max_time=100000.0,
        out_dir="../output/convergence_physical",
    )
    sweep_fusion_fission_time(p_ref, trials_per_cell=100, seed=0)

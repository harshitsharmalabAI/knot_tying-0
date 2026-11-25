"""Figures, tables and RESULTS.md from everything in results/."""
import csv
import glob
import json
import math
import os
import zipfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
RUNS = os.path.join(RES, "runs")
FIG = os.path.join(RES, "figures")
TAB = os.path.join(RES, "tables")
STAGE_NAMES = ["settle", "bring", "wrap 1", "wrap 2", "B to tail", "B grips", "draw-through", "cinch", "hold"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
REPORT = []


def say(text=""):
    print(text)
    REPORT.append(text)


def load_run(name):
    path = os.path.join(RUNS, name, "run_summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        s = json.load(f)
    log = os.path.join(RUNS, name, "tension_log.csv")
    s["log"] = np.genfromtxt(log, delimiter=",", names=True) if os.path.exists(log) else None
    s["name"] = name
    return s


def success(s, min_tension=8.0):
    """Tied and held: stable, clean strands, wound together, both strands >= 8 N, A-B gap >= 3 cm."""
    if s is None or not s["stable"]:
        return False
    released = bool(s["args"].get("release"))
    if s.get("last_stage") != (9 if released else 8):
        return False
    clean = all(v["det"] == 1 for v in s["strands"].values())
    if released:
        return bool(clean and abs(s["final_winding"]) >= 1.0)
    return bool(clean and abs(s["final_winding"]) >= 1.0 and s["final_a_tension"] >= min_tension
                and s["final_tail_tension"] >= min_tension and s["min_clearance_m"] >= 0.03)


def mean_ci(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    if len(x) == 1:
        return x[0], np.nan, np.nan
    t95 = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26, 11: 2.23,
           12: 2.20, 13: 2.18, 14: 2.16, 15: 2.14, 16: 2.13, 20: 2.09, 30: 2.05}.get(len(x), 2.0)
    sd = x.std(ddof=1)
    return x.mean(), sd, t95 * sd / math.sqrt(len(x))


def wilson(k, n):
    if n == 0:
        return np.nan, np.nan
    z, p = 1.96, k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def write_table(name, rows):
    if not rows:
        return
    os.makedirs(TAB, exist_ok=True)
    with open(os.path.join(TAB, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    say(f"\n**{name}**\n")
    keys = list(rows[0])
    say("| " + " | ".join(keys) + " |")
    say("|" + "---|" * len(keys))
    for r in rows:
        say("| " + " | ".join(str(r[k]) for k in keys) + " |")


def shade(ax, log, alpha=0.06):
    if log is None:
        return
    t, st = log["t"], log["stage"].astype(int)
    for k in np.unique(st):
        sel = t[st == k]
        ax.axvspan(sel.min(), sel.max(), color="C0" if k % 2 else "C1", alpha=alpha, lw=0)
        ax.text((sel.min() + sel.max()) / 2, 1.0, STAGE_NAMES[k], transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=7, color="0.4")


def hold_metrics(s, target):
    log = s["log"]
    if log is None or len(log) == 0:
        return {}
    out = {}
    cinch = log["stage"] >= 7
    hold = log["stage"] == 8
    t0 = log["t"][cinch].min() if cinch.any() else np.nan
    for strand, col in (("A", "a_tension"), ("tail", "tail_tension")):
        T = log[col]
        if not hold.any():
            continue
        m, sd = T[hold].mean(), T[hold].std(ddof=1)
        outside = cinch & (np.abs(T - target) > 0.10 * target)
        idx = np.where(outside)[0]
        if len(idx) == 0:
            settle = 0.0
        elif idx[-1] == np.where(cinch)[0][-1]:
            settle = np.nan
        else:
            settle = log["t"][idx[-1] + 1] - t0
        out[strand] = dict(mean=m, std=sd, err_pct=100 * (m - target) / target,
                           overshoot_pct=max(0.0, 100 * (T[cinch].max() - target) / target), settle_s=settle)
    return out


def fmt(v, nd=2):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "never" if isinstance(v, float) and np.isnan(v) else "-"
    return f"{v:.{nd}f}"


def baseline():
    s = load_run("baseline")
    if s is None:
        say("baseline: not run yet")
        return None
    say("## Baseline (published method: deadband control, T* = 15 N)")
    say(f"stable {s['stable']}, final A {s['final_a_tension']:.2f} N, tail {s['final_tail_tension']:.2f} N, "
        f"closest A-B {100*s['min_clearance_m']:.1f} cm, winding {s['final_winding']:+.2f}, strands "
        f"{ {k: v['det'] for k, v in s['strands'].items()} }, success {success(s)}, wall {s['wall_s']/60:.1f} min")
    if not success(s):
        say("WARNING: the baseline does not meet the success rule - check the rule before using the batch results")
    write_table("baseline_stages.csv", [{k: v for k, v in r.items()} for r in s["stages"]])
    log = s["log"]
    if log is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
        ax1.plot(log["t"], log["a_tension"], label="strand at A")
        ax1.plot(log["t"], log["tail_tension"], label="tail at B")
        ax1.axhline(12, ls="--", c="0.5", lw=0.8)
        ax1.set_ylabel("tension (N)")
        ax1.legend(loc="upper left", frameon=False)
        shade(ax1, log)
        ax2.plot(log["t"], 100 * log["servo_a"], label="A servo travel")
        ax2.plot(log["t"], 100 * log["servo_b"], label="B servo travel")
        ax2.plot(log["t"], 100 * log["a_b_gap"], label="A-B gap", c="0.4")
        ax2.axhline(3, ls=":", c="r", lw=0.8)
        ax2.set_ylabel("cm")
        ax2.set_xlabel("time (s)")
        ax2.legend(loc="upper left", frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "baseline_tension.png"))
        plt.close(fig)
    return s


def controllers():
    rows, runs = [], sorted(glob.glob(os.path.join(RUNS, "ctrl_*")))
    if not runs:
        say("E3: no controller runs yet")
        return
    say("\n## E3 Tension control in the full simulation (from the after-throw checkpoint)")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for i, folder in enumerate(runs):
        s = load_run(os.path.basename(folder))
        if s is None:
            continue
        target = s["target_tension"]
        m = hold_metrics(s, target)
        for strand, ax, col in (("A", axes[0], "a_tension"), ("tail", axes[1], "tail_tension")):
            if s["log"] is not None and len(s["log"]):
                ax.plot(s["log"]["t"], s["log"][col], lw=1.2, label=s["name"].replace("ctrl_", ""),
                        ls="-" if "pi" in s["name"] else ("--" if "_p_" in s["name"] else ":"))
            if strand in m:
                rows.append(dict(run=s["name"].replace("ctrl_", ""), target_N=target, strand=strand, stable=s["stable"],
                                 hold_mean_N=fmt(m[strand]["mean"]), hold_std_N=fmt(m[strand]["std"]),
                                 steady_error_pct=fmt(m[strand]["err_pct"], 1),
                                 overshoot_pct=fmt(m[strand]["overshoot_pct"], 1),
                                 settle_10pct_s=fmt(m[strand]["settle_s"]),
                                 min_gap_cm=fmt(100 * s["min_clearance_m"], 1), success=success(s)))
    for ax, title in zip(axes, ("strand at A", "tail at B")):
        ax.axhline(12, c="k", lw=0.8, ls="--")
        ax.set_title(title)
        ax.set_xlabel("time (s)")
    axes[0].set_ylabel("tension (N)")
    axes[1].legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "e3_controllers_sim.png"))
    plt.close(fig)
    write_table("e3_controllers_sim.csv", rows)


def controller_model():
    path = os.path.join(RES, "controller_model", "controller_series.npz")
    if not os.path.exists(path):
        return
    d = np.load(path)
    fig, ax = plt.subplots(figsize=(7, 4))
    for k in d.files:
        if k in ("t", "K_s", "loss"):
            continue
        ax.plot(d["t"], d[k], label=k, lw=1.3, ls="-" if k.startswith("PI") else ("--" if k.startswith("P ") else ":"))
    ax.axhline(12, c="k", lw=0.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("tension (N)")
    ax.set_title(f"Spring model: K_s = {float(d['K_s']):.0f} N/m, loss {float(d['loss']):.1f} N/s")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "e3_controller_model.png"))
    plt.close(fig)
    with open(os.path.join(RES, "controller_model", "controller_model.csv")) as f:
        write_table("e3_controller_model.csv", list(csv.DictReader(f)))


def robustness():
    groups = {"deadband": [], "pi": [], "pi2": []}
    for ctrl in groups:
        for folder in sorted(glob.glob(os.path.join(RUNS, f"rob_{ctrl}_s*"))):
            s = load_run(os.path.basename(folder))
            if s is not None:
                groups[ctrl].append(s)
    if not any(groups.values()):
        say("E4: no robustness runs yet")
        return
    say("\n## E4 Robustness (10 random variations of friction, wrap radius and thread placement)")
    rows, summary = [], []
    for ctrl, runs in groups.items():
        for s in runs:
            hm = hold_metrics(s, 12.0)
            a = s["args"]
            rows.append(dict(controller=ctrl, seed=a["seed"], mu=a["mu_thread"], orbit_cm=100 * a["orbit_radius"],
                             stable=s["stable"], success=success(s),
                             final_A_N=fmt(s["final_a_tension"]), final_tail_N=fmt(s["final_tail_tension"]),
                             hold_err_A_pct=fmt(hm.get("A", {}).get("err_pct", np.nan), 1),
                             hold_err_tail_pct=fmt(hm.get("tail", {}).get("err_pct", np.nan), 1),
                             min_gap_cm=fmt(100 * s["min_clearance_m"], 1), winding=fmt(s["final_winding"]),
                             strands="/".join(str(v["det"]) for v in s["strands"].values())))
        k, n = sum(success(s) for s in runs), len(runs)
        lo, hi = wilson(k, n)
        if n == 0:
            continue
        line = dict(controller={"deadband": "deadband (T*=15)", "pi": "PI kp=ki=0.01", "pi2": "PI kp=ki=0.02"}[ctrl],
                    runs=n, success=f"{k}/{n}", success_95CI=f"{100*lo:.0f}-{100*hi:.0f}%")
        for key, label in (("final_a_tension", "final_A_N"), ("final_tail_tension", "final_tail_N")):
            m, sd, ci = mean_ci([s[key] for s in runs if s["stable"]])
            line[label] = f"{m:.2f} ± {sd:.2f} (95% CI ± {ci:.2f})"
        m, sd, ci = mean_ci([100 * s["min_clearance_m"] for s in runs])
        line["min_gap_cm"] = f"{m:.1f} ± {sd:.1f}"
        for strand in ("A", "tail"):
            errs = [abs(hold_metrics(s, 12.0).get(strand, {}).get("err_pct", np.nan)) for s in runs if s["stable"]]
            m, sd, ci = mean_ci(errs)
            line[f"abs_hold_err_{strand}_pct_vs_12N"] = f"{m:.1f} ± {sd:.1f} (95% CI ± {ci:.1f})"
        own = [abs(hold_metrics(s, s["target_tension"]).get("A", {}).get("err_pct", np.nan)) for s in runs if s["stable"]]
        line["abs_err_A_vs_own_command_pct"] = f"{np.nanmean(own):.1f}"
        summary.append(line)
    write_table("e4_robustness_runs.csv", rows)
    write_table("e4_robustness_summary.csv", summary)
    pairs_for_plot = {}
    for ctrl in ("pi", "pi2"):
        for strand in ("A", "tail"):
            paired = []
            for s_d in groups["deadband"]:
                s_p = next((s for s in groups[ctrl] if s["args"]["seed"] == s_d["args"]["seed"]), None)
                if s_p and s_d["stable"] and s_p["stable"]:
                    e_d = abs(hold_metrics(s_d, 12.0).get(strand, {}).get("err_pct", np.nan))
                    e_p = abs(hold_metrics(s_p, 12.0).get(strand, {}).get("err_pct", np.nan))
                    paired.append((e_d, e_p))
            if len(paired) < 3:
                continue
            pairs_for_plot[(ctrl, strand)] = paired
            diff = np.array([p - d for d, p in paired])
            m, sd, ci = mean_ci(diff)
            text = (f"paired |hold error| at {strand} ({ctrl} - deadband): {m:+.2f} ± {sd:.2f} % points, "
                    f"95% CI [{m-ci:+.2f}, {m+ci:+.2f}], n = {len(diff)}")
            try:
                from scipy import stats
                text += f"; paired t-test p = {stats.ttest_rel([p for _, p in paired], [d for d, _ in paired]).pvalue:.3g}"
                text += f"; Wilcoxon p = {stats.wilcoxon(diff).pvalue:.3g}"
            except Exception:
                pass
            say(text)
    if pairs_for_plot:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=False)
        for ax, strand in zip(axes, ("A", "tail")):
            for ctrl, x1, col in (("pi", 1, "C1"), ("pi2", 2, "C2")):
                for d, p in pairs_for_plot.get((ctrl, strand), []):
                    ax.plot([0, x1], [d, p], "-o", c=col, ms=4, alpha=0.6)
            ax.set_xticks([0, 1, 2], ["deadband\n(T*=15)", "PI\n0.01/0.01", "PI\n0.02/0.02"])
            ax.set_ylabel(f"|hold error| at {strand} vs 12 N (%)")
            ax.set_title(f"strand {strand}: same 10 variations")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "e4_paired.png"))
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    names = [c for c in groups if groups[c]]
    for i, ctrl in enumerate(names):
        runs = groups[ctrl]
        ok = [s for s in runs if s["stable"]]
        jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(ok))
        axes[0].scatter(i + jitter, [s["final_a_tension"] for s in ok], c=["C2" if success(s) else "C3" for s in ok])
        axes[1].scatter(i + jitter, [s["final_tail_tension"] for s in ok], c=["C2" if success(s) else "C3" for s in ok])
        axes[2].scatter(np.full(len(runs), i), [100 * s["min_clearance_m"] for s in runs], c="C0")
    labels = {"deadband": "deadband\n(T*=15)", "pi": "PI\n0.01/0.01", "pi2": "PI\n0.02/0.02"}
    for ax, yl in zip(axes, ("final tension at A (N)", "final tension at tail (N)", "closest A-B approach (cm)")):
        ax.set_xticks(range(len(names)), [labels[c] for c in names])
        ax.set_ylabel(yl)
    axes[1].axhline(12, c="k", ls="--", lw=0.8)
    axes[0].axhline(12, c="k", ls="--", lw=0.8)
    axes[2].axhline(3, c="r", ls=":", lw=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "e4_robustness.png"))
    plt.close(fig)


def full_run_table(prefix, title):
    rows = []
    base = load_run("baseline")
    for folder in ([os.path.join(RUNS, "baseline")] if base else []) + sorted(glob.glob(os.path.join(RUNS, prefix + "*"))):
        s = load_run(os.path.basename(folder))
        if s is None:
            continue
        last = s["stages"][-1] if s["stages"] else {}
        rows.append(dict(run=s["name"], stable=s["stable"], reached_stage=s.get("last_stage"),
                         unstable_at_s=fmt(s["unstable_t"]) if s["unstable_t"] else "-",
                         winding=fmt(s["final_winding"]), wraps_on_B=last.get("wraps_on_b", "-"),
                         strands="/".join(str(v["det"]) for v in s["strands"].values()),
                         final_A_N=fmt(s["final_a_tension"]), final_tail_N=fmt(s["final_tail_tension"]),
                         min_gap_cm=fmt(100 * s["min_clearance_m"], 1), success=success(s)))
    if len(rows) > 1:
        say(f"\n## {title}")
        write_table(f"{prefix.strip('_')}_runs.csv", rows)


def draw_knot(ax, p, view="side", zoom=(-0.13, 0.22, -0.012, 0.135)):
    i, j = (0, 1) if view == "side" else (0, 2)
    ax.plot(p[i, :31], p[j, :31], c="darkorange", lw=2.2, label="tail")
    ax.plot(p[i, 30:38], p[j, 30:38], c="k", lw=3.5, label="stitch")
    ax.plot(p[i, 37:], p[j, 37:], c="C0", lw=1.6, label="long strand")
    if view == "side":
        ax.axhspan(zoom[2], -0.005, color="0.88", zorder=0)
    ax.set_xlim(zoom[0], zoom[1])
    ax.set_ylim(zoom[2], zoom[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def knot_stages():
    path = os.path.join(RUNS, "baseline", "tension_knot_stages.npy")
    if not os.path.exists(path):
        return
    shapes = np.load(path)
    fig, axes = plt.subplots(3, 3, figsize=(11, 6.5))
    for k, ax in enumerate(axes.ravel()):
        if k >= len(shapes):
            ax.axis("off")
            continue
        draw_knot(ax, shapes[k])
        ax.set_title(f"{k}  {STAGE_NAMES[k]}", fontsize=10)
    axes[0][0].legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle("Baseline knot at the end of each stage (side view, zoomed; grey = tissue)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "knot_stages_baseline.png"))
    plt.close(fig)


def outcome_gallery():
    names = ["baseline"] + [os.path.basename(f) for f in sorted(glob.glob(os.path.join(RUNS, "release_*")))
                             + sorted(glob.glob(os.path.join(RUNS, "break_*")))
                             + sorted(glob.glob(os.path.join(RUNS, "fail_*")))]
    have = [n for n in names if os.path.exists(os.path.join(RUNS, n, "tension_knot_final.npy"))]
    if len(have) < 2:
        return
    cols = 4
    rows = int(np.ceil(len(have) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.5 * rows), squeeze=False)
    for ax, n in zip(axes.ravel(), have):
        p = np.load(os.path.join(RUNS, n, "tension_knot_final.npy"))
        s = load_run(n)
        draw_knot(ax, p, zoom=(-0.16, 0.25, -0.012, 0.11))
        ok = success(s)
        state = "unstable" if not s["stable"] else ("tied" if ok else "FAILED")
        ax.set_title(f"{n}\n{state}; winding {s['final_winding']:+.2f}", fontsize=9, color="C2" if ok else "C3")
    for ax in axes.ravel()[len(have):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "outcomes_gallery.png"))
    plt.close(fig)


def release_and_breaks():
    rows = []
    for folder in sorted(glob.glob(os.path.join(RUNS, "release_*"))) + sorted(glob.glob(os.path.join(RUNS, "break_*"))):
        s = load_run(os.path.basename(folder))
        if s is None:
            continue
        st = {r["stage"]: r for r in s["stages"]}
        held = st.get(8, {})
        rows.append(dict(run=s["name"], mu=s["args"]["mu_thread"], stable=s["stable"], reached_stage=s.get("last_stage"),
                         unstable_at_s=fmt(s["unstable_t"]) if s["unstable_t"] else "-",
                         winding_held=fmt(held.get("winding", np.nan)), winding_final=fmt(s["final_winding"]),
                         wraps_on_B=s["stages"][-1].get("wraps_on_b", "-") if s["stages"] else "-",
                         strands="/".join(str(v["det"]) for v in s["strands"].values()),
                         final_A_N=fmt(s["final_a_tension"]), final_tail_N=fmt(s["final_tail_tension"]),
                         success=success(s)))
    if rows:
        say("\n## Batch 2: release tests and breaking points")
        write_table("batch2_release_breaks.csv", rows)
    for name in ("current", "best"):
        path = os.path.join(RES, "base_placement", f"verify_{name}_bases.txt")
        if os.path.exists(path):
            say(f"\n### Planner check, {name} bases\n```\n{open(path).read().strip()}\n```")


def ik():
    path = os.path.join(RES, "ik_compare", "ik_compare.csv")
    if not os.path.exists(path):
        return
    say("\n## E1 IK methods on the recorded knot (all frames, warm start)")
    with open(path) as f:
        write_table("e1_ik_compare.csv", list(csv.DictReader(f)))
    if not os.path.exists(os.path.join(RES, "ik_compare", "ik_series.npz")):
        return
    d = np.load(os.path.join(RES, "ik_compare", "ik_series.npz"))
    t = d["t"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    for ax, arm in zip(axes, ("a_", "b_")):
        for k in d.files:
            if k.startswith(arm) and k.endswith("_iters"):
                ax.plot(t, np.maximum(d[k], 0.5), lw=1, label=k[2:-6])
        ax.set_yscale("log")
        ax.set_ylabel(f"arm {arm[0].upper()}: iterations")
    axes[0].legend(fontsize=7, frameon=False, ncol=3)
    axes[1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "e1_ik_iterations.png"))
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.5), sharex=True)
    for ax, key, yl in ((axes[0], "manip", "|det J|"), (axes[1], "sigma_min", "smallest singular value")):
        for arm in ("a_", "b_"):
            ax.plot(t, d[arm + key], label=f"arm {arm[0].upper()}")
        ax.set_ylabel(yl)
    axes[0].legend(frameon=False)
    axes[1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "e1_manipulability.png"))
    plt.close(fig)
    rows = [dict(arm=arm[0].upper(), min_det_J=f"{d[arm+'manip'].min():.4f}", mean_det_J=f"{d[arm+'manip'].mean():.4f}",
                 min_sigma=f"{d[arm+'sigma_min'].min():.4f}",
                 at_t_s=f"{t[np.argmin(d[arm+'manip'])]:.2f}") for arm in ("a_", "b_")]
    write_table("e1_manipulability.csv", rows)
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    for ax, arm in zip(axes, ("a_", "b_")):
        q = np.degrees(np.unwrap(d[arm + "q"], axis=0))
        for j in range(q.shape[1]):
            ax.plot(t, q[:, j], lw=1, label=f"q{j+1}")
        ax.set_ylabel(f"arm {arm[0].upper()} joints (deg)")
    axes[0].legend(fontsize=7, ncol=6, frameon=False)
    axes[1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "joint_angles.png"))
    plt.close(fig)
    sp = os.path.join(RES, "ik_compare", "ik_stress.npz")
    if os.path.exists(sp):
        s = np.load(sp)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        r = s["flange_dist_m"]
        for k in s.files:
            if k.endswith("_err"):
                axes[0].plot(r, 1e3 * s[k], label=k[:-4])
                axes[1].plot(r, s[k.replace("_err", "_jump")], label=k[:-4])
        for ax in axes:
            ax.axvline(0.85, c="r", ls=":", lw=0.8)
            ax.set_xlabel("distance shoulder to flange target (m)")
        axes[0].set_ylabel("position error (mm)")
        axes[1].set_ylabel("largest joint change per step (deg)")
        axes[0].legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "e1_singularity_stress.png"))
        plt.close(fig)


def profiles():
    path = os.path.join(RES, "traj_profiles", "traj_profiles.csv")
    if not os.path.exists(path):
        return
    say("\n## E2 Time scaling of the instrument moves (nominal plan, DLS IK)")
    with open(path) as f:
        write_table("e2_traj_profiles.csv", list(csv.DictReader(f)))
    if not os.path.exists(os.path.join(RES, "traj_profiles", "traj_series.npz")):
        return
    d = np.load(os.path.join(RES, "traj_profiles", "traj_series.npz"))
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    for label in ("linear", "cosine (ours)", "quintic", "cosine + eased wraps"):
        if f"{label}|t" not in d.files:
            continue
        t = d[f"{label}|t"]
        axes[0].plot(t, d[f"{label}|a_speed"], lw=1, label=label)
        axes[1].plot(t, np.degrees(np.abs(d[f"{label}|a_qdd"]).max(1)), lw=1, label=label)
    axes[0].set_ylabel("A tool speed (m/s)")
    axes[1].set_ylabel("A max |joint acc| (deg/s²)")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("time (s)")
    axes[0].legend(fontsize=7, frameon=False)
    for ax in axes:
        for e in d["stage_ends"]:
            ax.axvline(e, c="0.85", lw=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "e2_profiles.png"))
    plt.close(fig)


def bases():
    path = os.path.join(RES, "base_placement", "base_grid.csv")
    if not os.path.exists(path):
        return
    say("\n## E6 Base placement (grid search, score = smallest |det J| along the knot)")
    with open(path) as f:
        rows = list(csv.DictReader(f))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, arm in zip(axes, ("A", "B")):
        r = [x for x in rows if x["arm"] == arm]
        xs, ys = sorted({float(x["x"]) for x in r}), sorted({float(x["y"]) for x in r})
        Z = np.full((len(ys), len(xs)), np.nan)
        for x in r:
            if x["feasible"] == "True":
                Z[ys.index(float(x["y"])), xs.index(float(x["x"]))] = float(x["min_manip"])
        im = ax.imshow(Z, origin="lower", cmap="viridis", aspect="auto",
                       extent=(xs[0] - 0.025, xs[-1] + 0.025, ys[0] - 0.05, ys[-1] + 0.05))
        for x in r:
            if x["current"] == "True":
                ax.plot(float(x["x"]), float(x["y"]), "r*", ms=12, label="current")
        ax.set_title(f"arm {arm} (blank = infeasible)")
        ax.set_xlabel("base x (m)")
        ax.set_ylabel("base y (m)")
        fig.colorbar(im, ax=ax, label="min |det J|")
        ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "e6_base_placement.png"))
    plt.close(fig)
    cur = [dict(arm=x["arm"], x=x["x"], y=x["y"], min_det_J=x["min_manip"], feasible=x["feasible"]) for x in rows if x["current"] == "True"]
    write_table("e6_current_bases.csv", cur)
    bp = os.path.join(RES, "base_placement", "best_pairs.csv")
    if os.path.exists(bp):
        with open(bp) as f:
            write_table("e6_best_pairs.csv", list(csv.DictReader(f))[:5])


def instrument_paths():
    path = os.path.join(RUNS, "baseline", "tension_knot_frames.npz")
    if not os.path.exists(path):
        return
    d = np.load(path)
    import sys
    sys.path.insert(0, os.path.join(HERE, "knot_ros"))
    import scene
    a = np.array([scene.to_world(p) for p in d["a_pos"]])
    b = np.array([scene.to_world(p) for p in d["b_tip"]])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (i, j, xl, yl) in zip(axes, ((0, 2, "x (m)", "z (m)"), (1, 2, "y (m)", "z (m)"))):
        sc = ax.scatter(a[:, i], a[:, j], c=d["t"], s=4, cmap="viridis", label="A (hinge)")
        ax.scatter(b[:, i], b[:, j], c=d["t"], s=4, cmap="magma", marker="s", label="B (tip)")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_aspect("equal")
    axes[0].legend(fontsize=7, frameon=False)
    fig.colorbar(sc, ax=axes[1], label="time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "instrument_paths.png"))
    plt.close(fig)


def pack():
    keep = []
    for root, _, files in os.walk(RES):
        for f in files:
            full = os.path.join(root, f)
            if f.endswith((".png", ".csv", ".json", ".md", ".txt", ".npy")):
                if os.path.getsize(full) < 3e6:
                    keep.append(full)
    with zipfile.ZipFile(os.path.join(HERE, "results_summary.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for full in keep:
            z.write(full, os.path.relpath(full, HERE))
    print(f"packed {len(keep)} files into results_summary.zip "
          f"({os.path.getsize(os.path.join(HERE, 'results_summary.zip'))/1e6:.1f} MB)")


def main():
    os.makedirs(FIG, exist_ok=True)
    say("# Course experiment results\n")
    baseline()
    controller_model()
    controllers()
    robustness()
    full_run_table("fail_", "E5 Failure cases (full runs)")
    full_run_table("prof_", "E2b Full runs with other time scalings")
    release_and_breaks()
    knot_stages()
    outcome_gallery()
    ik()
    profiles()
    bases()
    instrument_paths()
    with open(os.path.join(RES, "RESULTS.md"), "w") as f:
        f.write("\n".join(REPORT) + "\n")
    print("wrote results/RESULTS.md, results/figures/, results/tables/")
    pack()


if __name__ == "__main__":
    main()

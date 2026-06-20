#!/usr/bin/env python
"""
plot_depth_accuracy.py -- depth vs. pedigree-inference accuracy figure.

Walks the sweep tree

    <sweep_root>/depth<D>/seed<S>/results/
        ground_truth_pedigree.csv
        pedigree_inference_discovered.csv

and, for every (depth, seed) combo, recomputes the two metrics the pipeline
reports (it prints them to the log but does not save them), using the SAME logic
as pedigree_sim_pipeline.py:

    * Generation accuracy  = fraction of samples whose inferred Generation
                             matches the truth.  Measured over F2+F3
                             descendants by default (--gen-scope, below) --
                             the SAME scope as parentage -- so the trivial
                             default-F1 label (assigned to any sample with no
                             inferred parent) cannot inflate it.  At depths
                             where no parent links survive the consistency
                             cutoff, every sample collapses to the default F1
                             and F2+F3 generation accuracy is correctly 0%
                             (vs the F1 base rate the F1+F2+F3 scope reports).
    * Parentage accuracy   = fraction of F2+F3 samples whose inferred parent
      (F2+F3)                SET matches the truth (an F1 -- whose true parents
                             are Founders -- counts as correct iff it was given
                             no parents).

It then averages across seeds per depth, with a t-based confidence interval, and
draws a publication-style figure (mean line + markers, shaded CI band, faint
per-seed points) saved as PNG (300 dpi) + vector PDF, plus a tidy summary CSV.

Usage (from the sweep project dir, with bio-env active):

    python plot_depth_accuracy.py
    python plot_depth_accuracy.py --sweep-root /path/to/pedigree_depth_sweep
    python plot_depth_accuracy.py --out figures/depth_accuracy --xscale linear
    python plot_depth_accuracy.py --ci 0.90 --no-points
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")              # headless: render to file, no display needed
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

# Columns we rely on in both pedigree CSVs.
_COLS = ["Sample", "Generation", "Parent1", "Parent2"]

# Colourblind-safe (Wong) palette + markers, one per metric.
_SERIES = [
    ("parentage",  "Parentage (F2+F3)", "#0072B2", "o"),
    ("generation", "Generation",        "#D55E00", "s"),
]


def _metric_label(key, gen_scope):
    """Legend/console label; generation label reflects the chosen scope."""
    if key == "parentage":
        return "Parentage (F2+F3)"
    return "Generation (F2+F3)" if gen_scope == "descendants" else "Generation"


# ---------------------------------------------------------------------------
# Per-combo accuracy -- mirrors pedigree_sim_pipeline.py exactly
# ---------------------------------------------------------------------------
def _parents_match(row):
    """True iff the inferred parent set matches truth (pipeline's rule)."""
    true_p = {row["Parent1_True"], row["Parent2_True"]}
    true_p = {x for x in true_p if pd.notna(x)}
    inf_p = {row["Parent1_Inf"], row["Parent2_Inf"]}
    inf_p = {x for x in inf_p if pd.notna(x)}
    # F1: truth parents are Founders -> correct iff inferred gave it no parents.
    if any("Founder" in str(x) for x in true_p):
        return len(inf_p) == 0
    return true_p == inf_p


def combo_accuracy(truth_csv, inf_csv, gen_scope="descendants"):
    """Return (parentage_acc%, generation_acc%, n_matched_samples).

    gen_scope selects which samples the generation accuracy is measured over:
      "descendants" (default) -- F2+F3 only, matching the parentage scope.  F1
          is excluded because the inference labels any sample with no inferred
          parent as the default 'F1'; counting F1s credits that trivial
          default, so at depths where no real parent links are found the
          metric reads the true-F1 base rate instead of 0.
      "all" -- F1+F2+F3, the definition pedigree_sim_pipeline.py logs.
    """
    truth = pd.read_csv(truth_csv)
    inf = pd.read_csv(inf_csv)
    missing = [c for c in _COLS if c not in truth.columns or c not in inf.columns]
    if missing:
        raise ValueError("missing column(s) %s" % missing)

    v = pd.merge(truth[_COLS], inf[_COLS], on="Sample", suffixes=("_True", "_Inf"))
    if len(v) == 0:
        return np.nan, np.nan, 0

    descendants = v["Generation_True"].isin(["F2", "F3"])
    n_desc = int(descendants.sum())

    gen_match = (v["Generation_True"] == v["Generation_Inf"])
    if gen_scope == "descendants":
        gen_acc = float(gen_match[descendants].mean()) * 100.0 if n_desc else np.nan
    else:
        gen_acc = float(gen_match.mean()) * 100.0

    par_match = v.apply(_parents_match, axis=1)
    parent_acc = float(par_match[descendants].mean()) * 100.0 if n_desc else np.nan

    return parent_acc, gen_acc, int(len(v))


# ---------------------------------------------------------------------------
# Sweep traversal
# ---------------------------------------------------------------------------
def collect(sweep_root, gen_scope="descendants"):
    """Build a per-(depth, seed) DataFrame of accuracies from the sweep tree."""
    rows = []
    depth_dirs = sorted(glob.glob(os.path.join(sweep_root, "depth*")))
    if not depth_dirs:
        raise SystemExit("[plot_depth_accuracy] no depth*/ dirs under %s" % sweep_root)

    for ddir in depth_dirs:
        base = os.path.basename(ddir)
        try:
            depth = float(base[len("depth"):])      # "depth0.2" -> 0.2, "depth5" -> 5.0
        except ValueError:
            print("[skip] unparseable depth dir: %s" % base, file=sys.stderr)
            continue
        for sdir in sorted(glob.glob(os.path.join(ddir, "seed*"))):
            sbase = os.path.basename(sdir)
            try:
                seed = int(sbase[len("seed"):])
            except ValueError:
                continue
            res = os.path.join(sdir, "results")
            truth_csv = os.path.join(res, "ground_truth_pedigree.csv")
            inf_csv = os.path.join(res, "pedigree_inference_discovered.csv")
            if not (os.path.isfile(truth_csv) and os.path.isfile(inf_csv)):
                print("[skip] depth %g seed %d: results CSVs not found (incomplete?)"
                      % (depth, seed), file=sys.stderr)
                continue
            try:
                p, g, n = combo_accuracy(truth_csv, inf_csv, gen_scope)
            except Exception as exc:                 # noqa: BLE001 - report + skip
                print("[skip] depth %g seed %d: %s" % (depth, seed, exc),
                      file=sys.stderr)
                continue
            rows.append({"depth": depth, "seed": seed,
                         "parentage": p, "generation": g, "n_samples": n})

    if not rows:
        raise SystemExit("[plot_depth_accuracy] found depth dirs but no readable "
                         "results CSVs -- have any combos finished?")
    return pd.DataFrame(rows).sort_values(["depth", "seed"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Aggregation (mean + t-based CI across seeds)
# ---------------------------------------------------------------------------
def _t_crit(ci, df):
    """Two-sided t critical value; scipy if present, else normal approx."""
    try:
        from scipy.stats import t
        return float(t.ppf((1.0 + ci) / 2.0, df))
    except Exception:                                # noqa: BLE001
        return 1.96


def aggregate(raw, ci_level):
    out = []
    for metric, _, _, _ in _SERIES:
        for depth, grp in raw.groupby("depth"):
            vals = grp[metric].to_numpy(float)
            vals = vals[np.isfinite(vals)]
            n = int(vals.size)
            if n == 0:
                continue
            mean = float(vals.mean())
            if n > 1:
                sem = float(vals.std(ddof=1) / np.sqrt(n))
                half = _t_crit(ci_level, n - 1) * sem
            else:
                sem, half = 0.0, 0.0                 # single seed -> point only
            # Accuracy is bounded [0, 100]; clip the interval to the feasible
            # range so a wide small-n CI never plots a band above 100% / below 0.
            out.append({"depth": float(depth), "metric": metric, "mean": mean,
                        "ci_low": max(0.0, mean - half),
                        "ci_high": min(100.0, mean + half),
                        "sem": sem, "n_seeds": n,
                        "seeds": ",".join(str(int(s)) for s in sorted(grp["seed"]))})
    return pd.DataFrame(out).sort_values(["metric", "depth"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def make_figure(agg, raw, out_prefix, ci_level, xscale, show_points, title, gen_scope):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.labelsize": 12.5,
        "axes.titlesize": 13.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10.5,
        "axes.linewidth": 0.9,
        "xtick.direction": "out",
        "ytick.direction": "out",
    })

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color="0.88", lw=0.6)

    rng = np.random.default_rng(0)
    depths_all = np.sort(agg["depth"].unique())
    ymins = []

    for key, label, color, marker in _SERIES:
        sub = agg[agg["metric"] == key].sort_values("depth")
        if sub.empty:
            continue
        d = sub["depth"].to_numpy(float)
        m = sub["mean"].to_numpy(float)
        lo = sub["ci_low"].to_numpy(float)
        hi = sub["ci_high"].to_numpy(float)

        ax.fill_between(d, lo, hi, color=color, alpha=0.16, linewidth=0, zorder=1)
        ax.plot(d, m, color=color, lw=1.9, marker=marker, ms=6.5,
                mfc=color, mec="white", mew=0.9,
                label=_metric_label(key, gen_scope), zorder=3)
        ymins.append(np.nanmin(lo))

        if show_points:
            r = raw.dropna(subset=[key])
            xs = r["depth"].to_numpy(float)
            ys = r[key].to_numpy(float)
            if xscale == "log":
                xs = xs * np.exp(rng.uniform(-0.018, 0.018, size=xs.size))
            else:
                span = (depths_all.max() - depths_all.min()) or 1.0
                xs = xs + rng.uniform(-0.012, 0.012, size=xs.size) * span
            ax.scatter(xs, ys, s=13, color=color, alpha=0.30, linewidth=0, zorder=2)
            if ys.size:
                ymins.append(float(np.nanmin(ys)))

    if xscale == "log":
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(depths_all))
        ax.xaxis.set_major_formatter(FixedFormatter(["%g" % x for x in depths_all]))
        ax.xaxis.set_minor_locator(NullLocator())
        pad = 1.12
        ax.set_xlim(depths_all.min() / pad, depths_all.max() * pad)
    else:
        span = (depths_all.max() - depths_all.min()) or 1.0
        ax.set_xlim(depths_all.min() - 0.05 * span, depths_all.max() + 0.05 * span)

    ymin = max(0.0, (min(ymins) if ymins else 0.0) - 2.0)
    ax.set_ylim(ymin, 100.6)

    ax.set_xlabel("Sequencing depth (\u00d7)")
    ax.set_ylabel("Accuracy (%)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower right", handlelength=1.6,
              borderaxespad=0.6)
    if title:
        ax.set_title(title, pad=10)

    ax.text(0.0, -0.155,
            "Markers: mean across seeds.  Bands: %d%% CI (t-distribution).  "
            "Faint points: individual seeds." % round(ci_level * 100),
            transform=ax.transAxes, fontsize=8.3, color="0.4")

    fig.tight_layout()
    png, pdf = out_prefix + ".png", out_prefix + ".pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")          # vector, for the manuscript
    plt.close(fig)
    return png, pdf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-root", default=os.path.join(here, "pedigree_depth_sweep"),
                    help="root holding depth*/seed*/results (default: "
                         "./pedigree_depth_sweep next to this script)")
    ap.add_argument("--out", default=None,
                    help="output path PREFIX for .png/.pdf/_summary.csv "
                         "(default: <sweep-root>/depth_accuracy)")
    ap.add_argument("--ci", type=float, default=0.95, help="CI level (default 0.95)")
    ap.add_argument("--xscale", choices=["log", "linear"], default="log",
                    help="depth-axis scale (default log)")
    ap.add_argument("--no-points", action="store_true",
                    help="hide the per-seed scatter, show only mean + CI band")
    ap.add_argument("--gen-scope", choices=["descendants", "all"],
                    default="descendants",
                    help="samples the generation accuracy is measured over: "
                         "'descendants' = F2+F3 (default, matches parentage; "
                         "low-depth all-default-F1 collapse reads 0%%), "
                         "'all' = F1+F2+F3 (pedigree_sim_pipeline.py's logged "
                         "definition; inflated by the trivial F1 base rate)")
    ap.add_argument("--title", default="Pedigree-inference accuracy vs. sequencing depth",
                    help="figure title ('' for none)")
    args = ap.parse_args()

    sweep_root = os.path.abspath(args.sweep_root)
    out_prefix = args.out or os.path.join(sweep_root, "depth_accuracy")
    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)

    raw = collect(sweep_root, args.gen_scope)
    agg = aggregate(raw, args.ci)

    # tidy summary CSV (one row per depth x metric)
    summary_csv = out_prefix + "_summary.csv"
    agg.to_csv(summary_csv, index=False)

    # console summary
    print("\nDepth-accuracy summary (mean across seeds; %d%% CI):" % round(args.ci * 100))
    for metric, _label, _, _ in _SERIES:
        print("  %s:" % _metric_label(metric, args.gen_scope))
        for _, r in agg[agg["metric"] == metric].iterrows():
            print("    depth %-5g  %6.2f%%  [%6.2f, %6.2f]  (n=%d seeds: %s)"
                  % (r["depth"], r["mean"], r["ci_low"], r["ci_high"],
                     r["n_seeds"], r["seeds"]))

    png, pdf = make_figure(agg, raw, out_prefix, args.ci, args.xscale,
                           not args.no_points, args.title or None, args.gen_scope)
    print("\nWrote:\n  %s\n  %s\n  %s" % (png, pdf, summary_csv))


if __name__ == "__main__":
    main()
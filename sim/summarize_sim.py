#!/usr/bin/env python3
"""
summarize_sim.py

Read the JSON summary produced by `aggregate_results.py` and create
human-readable displays + diagnostic plots (headless/SLURM-safe).

Usage:
  python summarize_sim.py --summary results/summary.json --outdir results/plots

Notes:
- The attached `aggregate_results.py` does not store the time grid explicitly.
  By default we plot survival curves against an index-based grid in [0, 1].
  If you know the actual t_max used in the simulation, pass --t_max to label
  the x-axis in time units.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _time_axis(m: int, t_max: Optional[float]) -> np.ndarray:
    if m <= 1:
        return np.zeros((m,), dtype=float)
    if t_max is None:
        return np.linspace(0.0, 1.0, m)
    return np.linspace(0.0, float(t_max), m)


def add_metric_block(metrics: Dict[str, Any], name: str,
                     lines: List[str], test_subjects: List[Dict[str, Any]]) -> List[str]:
    blk = metrics.get(name, {})
    if not blk:
        return lines
    lines.append(f"{name.upper()}:")
    means = blk.get("mean", [])
    sds = blk.get("sd", [])
    for i, mu in enumerate(means):
        sd = sds[i] if i < len(sds) else None
        label = test_subjects[i].get("label") if i < len(test_subjects) else str(i)
        lines.append(f"  - {label}: mean={mu:.6g}" + (f", sd={sd:.6g}" if sd is not None else ""))
    lines.append("")
    return lines


def write_text_summary(summary: Dict[str, Any], out_path: Path) -> None:
    design = summary.get("design", {})
    lines: List[str] = []
    lines.append("Simulation summary (aggregate_results.py output)")
    lines.append("=" * 60)
    if design:
        lines.append("Design:")
        for k, v in design.items():
            lines.append(f"  - {k}: {v}")
        lines.append("")

    n_rep = summary.get("n_rep")
    if n_rep is not None:
        lines.append(f"Replicates aggregated: {n_rep}")
        lines.append("")

    test_subjects = summary.get("test_subjects", [])
    if test_subjects:
        lines.append("Fixed test subjects:")
        for s in test_subjects:
            lines.append(f"  - i={s.get('i')} label={s.get('label')}")
        lines.append("")

    metrics = summary.get("metrics", {})
    for nm in metrics.keys():
        lines = add_metric_block(metrics, nm, lines, test_subjects)

    vs = summary.get("variable_selection", {})
    if vs:
        lines.append("Variable selection (if VS used):")
        lines.append(f"  - prop_signal_selected_mean: {_safe_float(vs.get('prop_signal_selected_mean'))}")
        lines.append(f"  - n_selected_mean: {_safe_float(vs.get('n_selected_mean'))}")
        lines.append("")

    out_path.write_text("\n".join(lines))


def plot_metrics_bar(summary: Dict[str, Any], outdir: Path) -> None:
    """Bar plots of IMSE, bias, RMSE means with SD error bars."""
    test_subjects = summary.get("test_subjects", [])
    labels = [s.get("label", str(i)) for i, s in enumerate(test_subjects)]
    x = np.arange(len(labels))

    for metric_name in ["imse", "bias", "rmse"]:
        blk = summary.get(metric_name, {})
        if not blk:
            continue
        means = np.asarray(blk.get("mean_per_subject", []), dtype=float)
        sds = np.asarray(blk.get("sd_per_subject", []), dtype=float) if blk.get("sd_per_subject") else None

        fig = plt.figure()
        ax = fig.add_subplot(111)
        if sds is not None and len(sds) == len(means):
            ax.bar(x, means, yerr=sds, capsize=4)
        else:
            ax.bar(x, means)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title(f"{metric_name.upper()} (mean ± SD across replicates)")
        ax.set_xlabel("Fixed test subject")
        ax.set_ylabel(metric_name.upper())
        fig.tight_layout()
        fig.savefig(outdir / f"{metric_name}_bar.png", dpi=200)
        plt.close(fig)


def plot_survival_bands(summary: Dict[str, Any], outdir: Path, t_max: Optional[float]) -> None:
    """Plot mean S_hat and 90% bands per subject; overlay mean true curve if present."""
    bands = summary.get("bands_90", {})
    if not bands:
        return

    labels = bands.get("subject_labels") or [s.get("label", str(i)) for i, s in enumerate(summary.get("test_subjects", []))]
    lo = np.asarray(bands.get("lo", []), dtype=float)
    hi = np.asarray(bands.get("hi", []), dtype=float)
    mean_hat = np.asarray(bands.get("mean_hat", []), dtype=float)
    mean_true = np.asarray(bands.get("mean_true", []), dtype=float) if bands.get("mean_true") is not None else None

    if mean_hat.ndim != 2:
        # unexpected shape
        return

    m = mean_hat.shape[1]
    t = _time_axis(m, t_max)

    for i in range(mean_hat.shape[0]):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(t, mean_hat[i, :], label="Average SCENIC")
        ax.fill_between(t, lo[i, :], hi[i, :], alpha=0.3, label="Empirical Bound: SCENIC")
        if mean_true is not None and mean_true.shape == mean_hat.shape:
            ax.plot(t, mean_true[i, :], linestyle="--", label="True Survival Function")

        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"Survival Function with 90% Empirical Bound — Subject {labels[i] if i < len(labels) else i}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Survival Probability")
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / f"survival_bands_subject_{i}_{labels[i] if i < len(labels) else i}.png", dpi=200)
        plt.close(fig)


def plot_qq_bands(summary: Dict[str, Any], outdir: Path) -> None:
    """
    Optional: if your aggregator includes qq_90 (from other versions),
    plot mean + 90% bands for generated and true QQ curves per subject.
    """
    qq90 = summary.get("qq_90") or None
    if not qq90:
        return

    subjects = qq90.get("subjects", [])
    for subj in subjects:
        if not subj or not subj.get("available", True):
            continue
        label = subj.get("label", str(subj.get("i", "")))
        q = np.asarray(subj.get("q", []), dtype=float)
        gen = subj.get("gen", {})
        tru = subj.get("true", {})

        def arr(key: str, block: Dict[str, Any]) -> np.ndarray:
            return np.asarray(block.get(key, []), dtype=float)

        gen_mean, gen_lo, gen_hi = arr("mean", gen), arr("lo", gen), arr("hi", gen)
        true_mean, true_lo, true_hi = arr("mean", tru), arr("lo", tru), arr("hi", tru)

        if q.size == 0 or gen_mean.size == 0:
            continue

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(true_mean, gen_mean, label="Median QQ Line")
        if gen_lo.size == gen_mean.size and gen_hi.size == gen_mean.size:
            ax.fill_between(q, gen_lo, gen_hi, alpha=0.3, label="Empirical Bound")

        if true_mean.size == gen_mean.size:
            ax.plot(true_mean, true_mean, linestyle="--", label="True QQ Line")

        ax.set_title(f"Aggregate QQ Plot with 90% Empirical Bound — Subject {label}")
        ax.set_xlabel("Quantiles from True")
        ax.set_ylabel("Quantiles from Generator")
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / f"qq_bands_subject_{subj.get('i', 0)}_{label}.png", dpi=200)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=str, required=True, help="Path to JSON produced by aggregate_results.py")
    p.add_argument("--outdir", type=str, default=None, help="Directory to write plots (default: <summary_dir>/plots)")
    p.add_argument("--t_max", type=float, default=None, help="Optional: true t_max used in simulation for x-axis labeling")
    return p.parse_args()


def summarize_sim_input(summary_path: Path, outdir: Path, t_max: Optional[float]) -> None:
    summary_path = Path(summary_path)
    summary = json.loads(summary_path.read_text())

    outdir = Path(outdir) if outdir else (summary_path.parent / "plots")
    outdir.mkdir(parents=True, exist_ok=True)

    # Write a readable text summary
    write_text_summary(summary, outdir / "summary.txt")

    # Core plots present in the attached aggregate_results.py
    plot_metrics_bar(summary, outdir)
    plot_survival_bands(summary, outdir, t_max=t_max)

    # Optional QQ plots for other aggregator variants
    plot_qq_bands(summary, outdir)

    # Also save a lightweight index file (paths) for convenience
    produced = sorted([p.name for p in outdir.glob("*.png")])
    (outdir / "index.json").write_text(json.dumps({"plots": produced, "outdir": str(outdir)}, indent=2))

    print(f"Wrote plots to: {outdir}")
    print(f"Text summary: {outdir/'summary.txt'}")
    if produced:
        print("PNG files:")
        for nm in produced:
            print(f"  - {nm}")


def main() -> None:
    args = parse_args()
    summarize_sim_input(args.summary, args.outdir, args.t_max)


if __name__ == "__main__":
    main()

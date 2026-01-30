"""aggregate_results.py

Aggregate replicate JSON outputs produced by `run_simulation.py`.

Computes across-replicate summaries:
  - IMSE / bias / RMSE (per fixed test subject)
  - mean S_hat(t|x) and 90% empirical bounds (5th/95th percentiles) over replicates
  - QQ diagnostics: mean and 90% empirical bounds for each QQ curve, per fixed test subject

Example:
  python aggregate_results.py --glob 'results/scenic_PH_cens-mixed_mon-low_p5_pu5_rep*.json' \
    --out results/summary_PH_mixed_low_p5_pu5.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _load_jsons(glob_pat: str) -> List[Dict[str, Any]]:
    paths = sorted(glob.glob(glob_pat))
    if not paths:
        raise FileNotFoundError(f"No files matched glob: {glob_pat}")
    reps = []
    for p in paths:
        with open(p, "r") as f:
            reps.append(json.load(f))
    return reps


def _subject_labels(n_subj: int) -> List[str]:
    base = ["random", "0.25", "0.50", "0.75"]
    labels = []
    for i in range(n_subj):
        labels.append(base[i] if i < len(base) else f"extra_{i}")
    return labels


def _summarize_array(x: np.ndarray) -> Dict[str, Any]:
    return {
        "mean": np.asarray(x).mean(axis=0).tolist(),
        "lo": np.quantile(x, 0.05, axis=0).tolist(),
        "hi": np.quantile(x, 0.95, axis=0).tolist(),
    }


def _aggregate_qq(reps: List[Dict[str, Any]], n_subj: int) -> Dict[str, Any]:
    """Aggregate QQ diagnostics across replicates.

    Expects each replicate JSON to contain:
      qq: list[{i,label,q,true,gen}]  (preferred)
    Falls back to:
      qq: {q,true,gen}  (assumed subject 0)
    """
    labels = _subject_labels(n_subj)
    out: Dict[str, Any] = {"subjects": []}

    # Build per-rep per-subject qq dicts
    per_rep: List[Dict[int, Dict[str, Any]]] = []
    for rep in reps:
        qq = rep.get("qq", None)
        d: Dict[int, Dict[str, Any]] = {}
        if isinstance(qq, list):
            for item in qq:
                try:
                    i = int(item["i"])
                except Exception:
                    continue
                d[i] = item
        elif isinstance(qq, dict) and "q" in qq and "gen" in qq and "true" in qq:
            d[0] = {"i": 0,
                    "label": labels[0],
                    "q": qq["q"],
                    "true": qq["true"],
                    "gen": qq["gen"]}
        per_rep.append(d)

    # Aggregate for each subject i
    for i in range(n_subj):
        q_ref = None
        gen_stack = []
        true_stack = []
        for d in per_rep:
            if i not in d:
                continue
            item = d[i]
            q = np.asarray(item["q"], dtype=float)
            g = np.asarray(item["gen"], dtype=float)
            t = np.asarray(item["true"], dtype=float)

            if q_ref is None:
                q_ref = q
            else:
                # Require same q grid; if not, skip this rep for this subject
                if len(q) != len(q_ref) or np.max(np.abs(q - q_ref)) > 1e-12:
                    continue
            gen_stack.append(g)
            true_stack.append(t)

        if q_ref is None or len(gen_stack) == 0:
            # No QQ info for this subject
            out["subjects"].append({"i": i, "label": labels[i], "available": False})
            continue

        gen_arr = np.stack(gen_stack, axis=0)
        true_arr = np.stack(true_stack, axis=0)

        out["subjects"].append(
            {
                "i": i,
                "label": labels[i],
                "available": True,
                "n_rep_used": int(gen_arr.shape[0]),
                "q": q_ref.tolist(),
                "gen": _summarize_array(gen_arr),
                "true": _summarize_array(true_arr),
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--glob", required=True, help="Glob pattern for replicate JSON files.")
    p.add_argument("--out", required=True, help="Output path for aggregated JSON summary.")
    return p.parse_args()


def aggregate_results_input(glob_path, out_path) -> None:
    reps = _load_jsons(glob_path)

    n_rep = len(reps)
    subj0 = reps[0].get("subj_metrics", [])
    if not subj0:
        raise ValueError("Replicate JSON files missing 'subj_metrics'.")

    n_subj = len(subj0)
    labels = _subject_labels(n_subj)

    # Assume each subject metric includes s_hat/s_true vectors on same t_grid across reps
    t_grid = reps[0].get("t_grid", None)
    if t_grid is None:
        raise ValueError("Replicate JSON files missing 't_grid'.")
    t_grid = np.asarray(t_grid, dtype=float)

    # Collect arrays
    imse = np.zeros((n_rep, n_subj), dtype=float)
    bias = np.zeros((n_rep, n_subj), dtype=float)
    rmse = np.zeros((n_rep, n_subj), dtype=float)

    s_hat = np.zeros((n_rep, n_subj, len(t_grid)), dtype=float)
    s_true = np.zeros((n_rep, n_subj, len(t_grid)), dtype=float)

    for r_i, rep in enumerate(reps):
        sm_list = rep["subj_metrics"]
        if len(sm_list) != n_subj:
            raise ValueError(f"Inconsistent number of subjects in replicate {r_i}.")
        for s_i, sm in enumerate(sm_list):
            imse[r_i, s_i] = float(sm["imse"])
            bias[r_i, s_i] = float(sm["bias"])
            rmse[r_i, s_i] = float(sm["rmse"])
            s_hat[r_i, s_i, :] = np.asarray(sm["s_hat"], dtype=float)
            s_true[r_i, s_i, :] = np.asarray(sm["s_true"], dtype=float)

    # Bands for survival curves
    lo = np.quantile(s_hat, 0.05, axis=0)
    hi = np.quantile(s_hat, 0.95, axis=0)
    mean_hat = s_hat.mean(axis=0)
    mean_true = s_true.mean(axis=0)

    # Variable selection (if present)
    vs_items = [rep.get("vs", {}) for rep in reps if rep.get("vs")]
    vs_summary = {}
    if vs_items:
        # Only average numeric scalars
        keys = sorted(set().union(*[v.keys() for v in vs_items]))
        for k in keys:
            vals = [v.get(k, None) for v in vs_items]
            vals = [float(x) for x in vals if isinstance(x, (int, float)) and not np.isnan(x)]
            if vals:
                vs_summary[k] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}

    qq_summary = _aggregate_qq(reps, n_subj)

    out = {
        "n_rep": n_rep,
        "design": {k: reps[0].get(k) for k in ["sim_model", "scenario", "monitor", "p_cov", "p_aux", "n_aux", "epochs"] if k in reps[0]},
        "t_grid": t_grid.tolist(),
        "test_subjects": [{"i": i, "label": labels[i]} for i in range(n_subj)],
        "metrics": {
            "imse": {"mean": imse.mean(axis=0).tolist(), "sd": imse.std(axis=0, ddof=1).tolist()},
            "bias": {"mean": bias.mean(axis=0).tolist(), "sd": bias.std(axis=0, ddof=1).tolist()},
            "rmse": {"mean": rmse.mean(axis=0).tolist(), "sd": rmse.std(axis=0, ddof=1).tolist()},
        },
        "bands_90": {
            "subject_labels": labels,
            "mean_hat": mean_hat.tolist(),
            "lo": lo.tolist(),
            "hi": hi.tolist(),
            "mean_true": mean_true.tolist(),
        },
        "qq_90": qq_summary,
        "vs": vs_summary,
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote summary to: {out_path}")


def main() -> None:
    args = parse_args()
    aggregate_results_input(args.glob, args.out)


if __name__ == "__main__":
    main()

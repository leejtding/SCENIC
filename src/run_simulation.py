"""run_simulation.py

Run a single simulation replicate.

This script is designed for SLURM array jobs:
  - each task runs one replicate (dataset generation, training, evaluation)
  - outputs a compact JSON result file (and optionally model checkpoints)
  - supports CPU or GPU via `--device auto|cpu|cuda`

Example (local):
  python run_simulation.py --scenario mixed --monitor low --sim_model PH \
    --p_cov 5 --p_aux 5 --rep 0 --outdir results

Example (SLURM array, GPU):
  srun python run_simulation.py ... --rep $SLURM_ARRAY_TASK_ID --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from models import SCENIC
from train import train
from sim_data import sim_ic_cond
from metrics import survival_metrics_for_subject, qq_data, variable_selection_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one simulation iteration.")

    # Core design factors
    p.add_argument("--sim_model", type=str, choices=["PH", "PO", "AFT"],
                   default="PH")
    p.add_argument("--censor", type=str, choices=["mixed", "pure_interval"],
                   default="mixed", help="Censoring scenario.")
    p.add_argument("--monitor", type=str, choices=["low", "high"],
                   default="low", help="Monitoring frequency.")
    p.add_argument("--p_cov", type=int, choices=[5, 100], default=5)
    p.add_argument("--p_aux", type=int, choices=[1, 5, 50], default=5)

    # Replication and output
    p.add_argument("--rep", type=int, default=0,
                   help="Replicate index (use SLURM_ARRAY_TASK_ID).")
    p.add_argument("--base_seed", type=int, default=3)
    p.add_argument("--outdir", type=str, default="../results")
    p.add_argument("--save_model", action="store_true",
                   help="Save trained model state_dict.")

    # Training hyperparameters
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--after_vs_epochs", type=int, default=20)
    p.add_argument("--vs", type=str, choices=["T", "F"], default="F")
    p.add_argument("--phi_steps", type=int, default=1)
    p.add_argument("--gen_steps", type=int, default=1)

    p.add_argument("--n_aux", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=5)
    p.add_argument("--train_size", type=int, default=100)
    p.add_argument("--temp", type=float, default=0.1)
    p.add_argument("--temp_decay", type=float, default=1.0)
    p.add_argument("--gen_hidden", type=int, default=1000)
    p.add_argument("--phi_hidden", type=int, default=1000)
    p.add_argument("--gen_lr", type=float, default=1e-4)
    p.add_argument("--phi_lr", type=float, default=1e-5)

    # Device
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda"],
                   help="Use 'auto' to select cuda if available.")

    # Evaluation parameters
    p.add_argument("--m_time", type=int, default=100,
                   help="Number of evaluation time points.")
    p.add_argument("--t_max", type=float, default=50.0,
                   help="Upper bound for evaluation time grid.")
    p.add_argument("--n_gen_eval", type=int, default=1000,
                   help="Generator draws for evaluation.")

    return p.parse_args()


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def censor_params(censor: str) -> Dict[str, float]:
    """Map scenario label to n_rc/n_ic/n_uc and censoring settings.

    We consider the following scenarios:
      1) Mixed interval censoring: n_R=2000, n_IC=2000, tau_R=30
      2) Pure interval censoring: text shows n_IC=4000, tau_R=30
    """
    if censor == "mixed":
        return {"n_rc": 2000, "n_ic": 2000, "n_uc": 0, "tau_rc": 30.0}
    if censor == "pure_interval":
        return {"n_rc": 0, "n_ic": 100, "n_uc": 0, "tau_rc": 30.0}
    raise ValueError("Unknown censoring scenario")


def monitor_params(monitor: str) -> Dict[str, float]:
    """Monitoring frequencies considered."""
    if monitor == "low":
        return {"c_ic": 1.0, "d_ic": 10.0, "k_ic": 30, "p_ic": 0.5}
    if monitor == "high":
        return {"c_ic": 0.5, "d_ic": 5.0, "k_ic": 60, "p_ic": 1.0}
    raise ValueError("Unknown monitoring frequency")


def fixed_test_covariates(p_cov: int) -> np.ndarray:
    """Generate fixed test individuals used across replicates.

    The plan (page 8) refers to fixed test individuals from SCENE.
    Since those exact covariates are not provided in the codebase,
    we create deterministic test covariates via a fixed RNG seed.
    """
    rng = np.random.default_rng(369)
    indv1 = rng.uniform(-1, 1, size = (p_cov,)).astype(np.float32)
    indv2 = np.repeat(0.25, p_cov).astype(np.float32)
    indv3 = np.repeat(0.5, p_cov).astype(np.float32)
    indv4 = np.repeat(0.75, p_cov).astype(np.float32)
    return np.vstack([indv1, indv2, indv3, indv4])


def to_jsonable(x):
    """Recursively convert numpy / torch objects to JSON-serializable Python types."""
    # numpy arrays
    if isinstance(x, np.ndarray):
        return x.tolist()
    # numpy scalars (np.float32, np.int64, etc.)
    if isinstance(x, (np.generic,)):
        return x.item()
    # torch tensors (optional, only if torch is used)
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().tolist()
    except Exception:
        pass

    # containers
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]

    # base python types (int/float/str/bool/None) pass through
    return x


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    rep_seed = int(args.base_seed + args.rep)
    np.random.seed(rep_seed)
    torch.manual_seed(rep_seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Data generation
    sp = censor_params(args.censor)
    mp = monitor_params(args.monitor)

    ic_df = sim_ic_cond(
        n_lc=0,
        n_rc=sp["n_rc"],
        n_ic=sp["n_ic"],
        n_uc=sp["n_uc"],
        p_cov=args.p_cov,
        r_cov={"PH": 0.7, "PO": 0.5, "AFT": 0.6}[args.sim_model],
        tau_lc=5.0,
        tau_rc=sp["tau_rc"],
        c_ic=mp["c_ic"],
        d_ic=mp["d_ic"],
        k_ic=int(mp["k_ic"]),
        p_ic=float(mp["p_ic"]),
        sim_model=args.sim_model,
        lambda_=0.5,
        seed=rep_seed,
    )

    l = torch.tensor(ic_df["l"].values, dtype=torch.float)
    r = torch.tensor(ic_df["r"].values, dtype=torch.float)
    X = torch.tensor(ic_df.iloc[:, 4:(4 + args.p_cov)].values, dtype=torch.float)

    # Initialize and train model
    scenic = SCENIC(
        l=l,
        r=r,
        X=X,
        n_aux=args.n_aux,
        p_aux=args.p_aux,
        batch_size=args.batch_size,
        train_size=args.train_size,
        device=device,
        temp=args.temp,
        temp_decay=args.temp_decay,
        gen_hidden=args.gen_hidden,
        phi_hidden=args.phi_hidden,
        gen_lr=args.gen_lr,
        phi_lr=args.phi_lr,
    )

    arg_vs = True if args.vs == "T" else False
    train(
        scenic,
        vs=arg_vs,
        total_epochs=args.epochs,
        after_vs_epochs=args.after_vs_epochs,
        phi_steps=args.phi_steps,
        gen_steps=args.gen_steps,
        sim_model=args.sim_model,
        n_plot=0, # Disable plotting during large-scale simulations
    )

    # Evaluate bias and variability of estimates
    x_test = fixed_test_covariates(args.p_cov)
    t_grid = np.linspace(0.0, args.t_max, args.m_time).astype(np.float32)

    subj_metrics = []
    for i in range(4):  # Four fixed test individuals
        m = survival_metrics_for_subject(
            scenic,
            sim_model=args.sim_model,
            x_test=x_test[i, :],
            t_grid=t_grid,
            n_gen=args.n_gen_eval,
        )
        subj_metrics.append({"i": i, **asdict(m)})

    # QQ diagnostics
    subj_labels = ["random", "0.25", "0.50", "0.75"]
    qq_metrics = []
    for i in range(min(4, x_test.shape[0])):
        qq = qq_data(
            scenic,
            sim_model=args.sim_model,
            x_test=x_test[i],
            n_gen=args.n_gen_eval,
        )
        qq_metrics.append(
            {
                "i": i,
                "label": subj_labels[i] if i < len(subj_labels) else f"extra_{i}",
                "q": qq["q"].tolist(),
                "true": qq["true"].tolist(),
                "gen": qq["gen"].tolist(),
            }
        )

    vs_metrics = {}
    if arg_vs:
        vs_metrics = variable_selection_metrics(np.asarray(scenic.vs_mask),
                                                p_cov=args.p_cov)

    result = {
        "rep": args.rep,
        "seed": rep_seed,
        "device": device,
        "sim_model": args.sim_model,
        "censor": args.censor,
        "monitor": args.monitor,
        "p_cov": args.p_cov,
        "p_aux": args.p_aux,
        "n_aux": args.n_aux,
        "epochs": args.epochs,
        "t_grid": t_grid.tolist(),
        "subj_metrics": subj_metrics,
        "qq": qq_metrics,
        "vs": vs_metrics,
    }

    stem = f"scenic_{args.sim_model}_cens-{args.censor}_mon-{args.monitor}_p{args.p_cov}_pu{args.p_aux}_rep{args.rep:03d}"
    out_json = outdir / f"{stem}.json"
    out_json.write_text(json.dumps(to_jsonable(result), indent=2))

    if args.save_model:
        out_pt = outdir / f"{stem}.pt"
        torch.save({"state_dict": scenic.state_dict(), "args": vars(args)}, out_pt)


if __name__ == "__main__":
    main()

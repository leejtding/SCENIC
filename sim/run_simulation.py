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
from typing import Dict, Any

import numpy as np
import torch
from scipy.special import erf

from scenic.models import SCENIC
from scenic.train import train
from scenic.sim_data import sim_ic_cond
from scenic.metrics import survival_metrics_for_subject, qq_data, variable_selection_metrics, true_survival


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one simulation iteration.")

    # Core design factors
    p.add_argument("--method", type=str, choices=["scenic", "spinet"], default="scenic",
               help="Which method to fit: 'scenic' (default) or 'spinet' (LassoNetIntervalRegressor).")
    p.add_argument("--sim_model", type=str, choices=["PH", "PO", "AFT"],
                   default="PH")
    p.add_argument("--censor", type=str, choices=["mixed", "pure_interval"],
                   default="mixed", help="Censoring scenario.")
    p.add_argument("--monitor", type=str, choices=["low", "high"],
                   default="low", help="Monitoring frequency.")
    p.add_argument("--p_cov", type=int, choices=[5, 100], default=5)
    p.add_argument("--p_aux", type=int, choices=[1, 5, 50], default=5)

    # SPINet (LassoNetIntervalRegressor) hyperparameters (only used if --method spinet)
    p.add_argument("--spinet_hidden", type=str, default="10,10",
               help="Comma-separated hidden layer sizes for SPINet, e.g. '10,10'.")
    p.add_argument("--spinet_dense_only", action="store_true",
               help="If set, fit SPINet dense-only (no sparsity path).")

    # Replication and output
    p.add_argument("--rep", type=int, default=0,
                   help="Replicate index (use SLURM_ARRAY_TASK_ID).")
    p.add_argument("--base_seed", type=int, default=3)
    p.add_argument("--outdir", type=str, default="../results")
    p.add_argument("--save_model", action="store_true",
                   help="Save trained model state_dict.")

    # Training hyperparameters
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--after_vs_epochs", type=int, default=20)
    p.add_argument("--vs", type=str, choices=["T", "F"], default="F")
    p.add_argument("--phi_steps", type=int, default=1)
    p.add_argument("--gen_steps", type=int, default=1)

    p.add_argument("--n_aux", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=10)
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


def _parse_hidden_dims(s: str) -> tuple[int, ...]:
    s = s.strip()
    if not s:
        return ()
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def _icdf_to_spinet_y(l: np.ndarray, r: np.ndarray, cens: np.ndarray) -> np.ndarray:
    """Convert (l, r, cens) into the (u, v, delta1, delta2, delta3) format used by
    the interval-regression interface in the attached SPINet example code.

    Conventions (matching your attached generate.py / main.py):
      - delta1: left-censored (not used here; your sim sets n_lc=0)
      - delta2: interval / observed within (u, v]
      - delta3: right-censored
      - right-censored uses v=9999 (a large finite number)
      - exact events are represented as a zero-width interval (u==v) with delta2=1
    """
    l = np.asarray(l, dtype=float)
    r = np.asarray(r, dtype=float)
    cens = np.asarray(cens, dtype=bool)

    u = l.copy()
    v = r.copy()

    delta1 = np.zeros_like(u)
    delta3 = (cens & ~np.isfinite(r)).astype(float)
    delta2 = (1.0 - delta1 - delta3).astype(float)

    # Avoid log(0) issues: push u away from 0
    eps_t = 1e-5
    u = np.maximum(u, eps_t)

    # Replace +inf right endpoint for right-censored with large finite value
    v[delta3.astype(bool)] = 9999.0

    # (Optional) guard: ensure v >= u
    swap = v < u
    if np.any(swap):
        tmp = u[swap].copy()
        u[swap] = v[swap]
        v[swap] = tmp

    return np.column_stack([u, v, delta1, delta2, delta3]).astype(np.float32)


def _spinet_predict_survival(model: Any, x: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Survival prediction for SPINet/LassoNet interval regressor.
      - model has methods: model._cast_input, model.model (a torch module)
      - calling model.model(X_tensor) returns (mo_tensor, sigma_tensor)
      - mo is the location on log-time scale; sigma positive scale
      - Survival: S(t) = 1 - Phi((log t - mo) / sigma)

    Returns
    -------
    s_hat : np.ndarray shape (m_time,)
    """
    X = np.asarray(x, dtype=float).reshape(1, -1)
    t_grid = np.asarray(t_grid, dtype=float).ravel()
    
    # _cast_input returns device-aware tensor
    X_t = model._cast_input(X)
    model.model.eval()
    with torch.no_grad():
        out = model.model(X_t)
    # out may be (ans, sigma)
    if isinstance(out, (tuple, list)) and len(out) >= 2:
        mo_t, sigma_t = out[0], out[1]
    else:
        raise RuntimeError("Unexpected model.model output format")
    mo = torch.squeeze(mo_t).cpu().numpy().astype(float)
    sigma = torch.squeeze(sigma_t).cpu().numpy().astype(float)

    # Normalise shapes to 1-D arrays
    if getattr(mo, "ndim", 0) == 0:
        mo = np.array([float(mo)])
    if getattr(sigma, "ndim", 0) == 0:
        sigma = np.array([float(sigma)])

    # compute survival for single subject
    eps = 1e-8
    t_grid_clipped = np.maximum(t_grid, eps)
    z = (np.log(t_grid_clipped)[:, None] - mo[None, :]) / (sigma[None, :] + eps)
    cdf = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    s = 1.0 - cdf[:, 0]
    s = np.clip(s, 0.0, 1.0)
    return s.reshape(-1)


def _survival_metrics_from_curve(sim_model: str, x_test: np.ndarray, t_grid: np.ndarray, s_hat: np.ndarray) -> Dict[str, Any]:
    s_true = true_survival(sim_model, t_grid, x_test)
    err = np.asarray(s_hat) - np.asarray(s_true)
    imse = float(np.mean(err ** 2))
    bias = float(np.mean(err))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return {
        "imse": imse,
        "bias": bias,
        "rmse": rmse,
        "s_hat": np.asarray(s_hat, dtype=float).tolist(),
        "s_true": np.asarray(s_true, dtype=float).tolist(),
    }


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

    # Fit chosen method
    x_test = fixed_test_covariates(args.p_cov)
    t_grid = np.linspace(0.0, args.t_max, args.m_time).astype(np.float32)

    subj_metrics = []
    qq_metrics = []
    vs_metrics: Dict[str, Any] = {}

    if args.method == "scenic":
        # Initialize and train SCENIC
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

        # Evaluate bias and variability for fixed test individuals
        for i in range(4):
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

        # Variable selection
        if arg_vs:
            vs_metrics = variable_selection_metrics(
                np.asarray(scenic.vs_mask),
                p_cov=args.p_cov,
            )

    elif args.method == "spinet":
        # SPINet (based on attached LassoNetIntervalRegressor example)
        try:
            from lassonet.interfaces import LassoNetIntervalRegressor
        except Exception as e:
            raise ImportError(
                "SPINet requires the 'lassonet' Python package (and its deps). "
                "Install it in your environment, e.g. `pip install lassonet`."
            ) from e

        X_np = ic_df.iloc[:, 4:(4 + args.p_cov)].to_numpy(dtype=np.float32)
        y_np = _icdf_to_spinet_y(
            ic_df["l"].to_numpy(),
            ic_df["r"].to_numpy(),
            ic_df["cens"].to_numpy(),
        )

        hidden_dims = _parse_hidden_dims(args.spinet_hidden)
        model = LassoNetIntervalRegressor(hidden_dims=hidden_dims, device=device)

        # Fit the interval regressor. Some versions support dense_only=True.
        try:
            if args.spinet_dense_only:
                model.fit(X_np, y_np, dense_only=True)
            else:
                model.fit(X_np, y_np)
        except TypeError:
            # fallback for versions that don't support dense_only kwarg
            model.fit(X_np, y_np)

        # Evaluate by predicting survival curves on t_grid for fixed x_test
        for i in range(4):
            s_hat = _spinet_predict_survival(model, x_test[i, :], t_grid)
            subj_metrics.append({"i": i, **_survival_metrics_from_curve(args.sim_model, x_test[i, :], t_grid, s_hat)})

    else:
        raise ValueError(f"Unknown method: {args.method}")

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

    stem = f"{args.method}_{args.sim_model}_cens-{args.censor}_mon-{args.monitor}_p{args.p_cov}_pu{args.p_aux}_rep{args.rep:03d}"
    out_json = outdir / f"{stem}.json"
    out_json.write_text(json.dumps(to_jsonable(result), indent=2))

    if args.save_model:
        out_pt = outdir / f"{stem}.pt"
        torch.save({"state_dict": scenic.state_dict(), "args": vars(args)}, out_pt)


if __name__ == "__main__":
    main()

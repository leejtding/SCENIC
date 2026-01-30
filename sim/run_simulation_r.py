"""
run_simulation_r.py

A lightweight Python wrapper to fit/predict R-based methods inside the
existing simulation framework.

This module is designed to mirror the "save predictions + metadata" pattern
used by the SCENIC Python simulations: it takes a training DataFrame,
a time grid, and an xtest design (e.g., 4 x p_cov covariate points),
calls an R script (r_fit_predict*.R), and returns the survival matrix.

Supported R methods (via --method):
  - survreg_weibull
  - icenreg_ph_po_aft
  - dnn_ic
  - icrf

The R side writes JSON with:
  {"method": "...", "s_hat": [[...]], "extra": {...}}
where s_hat is n_test x n_time.

Usage example (inside your simulator):
  from run_simulation_rmethod import r_fit_predict

  out = r_fit_predict(
      method="icrf",
      sim_model="PH",
      p_cov=p,
      train_df=train_df,
      tgrid=tgrid,
      xtest=xtest,  # numpy array shape (4, p)
      r_script="r_fit_predict_icrf.R",
      r_extra=dict(icrf_ntree=300, icrf_split_rule="Wilcoxon")
  )
  s_hat = out["s_hat"]   # numpy array

Notes
-----
- This wrapper intentionally does NOT depend on your SCENIC model code.
- It is safe to import from other scripts; the CLI is optional.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from run_simulation import resolve_device, censor_params, monitor_params, fixed_test_covariates, to_jsonable
from scenic.sim_data import sim_ic_cond
from scenic.metrics import true_survival


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one replicate with an R-based method.")
    p.add_argument("--method", type=str, choices=["icenreg", "dnn_ic", "icrf"], required=True)

    p.add_argument("--sim_model", type=str, choices=["PH", "PO", "AFT"], default="PH")
    p.add_argument("--censor", type=str, choices=["mixed", "pure_interval"], default="mixed")
    p.add_argument("--monitor", type=str, choices=["low", "high"], default="low")
    p.add_argument("--p_cov", type=int, choices=[5, 100], default=5)
    p.add_argument("--p_aux", type=int, choices=[1, 5, 50], default=5)

    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--base_seed", type=int, default=3)
    p.add_argument("--outdir", type=str, default="../results")

    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda"])

    p.add_argument("--m_time", type=int, default=100)
    p.add_argument("--t_max", type=float, default=50.0)
    p.add_argument("--n_gen_eval", type=int, default=1000)

    p.add_argument("--r_script", type=str, default="r_fit_predict.R",
                   help="Path to the R script that fits/predicts.")
    
    # ---- icenReg hyperparameters ----
    p.add_argument("--icenreg_model", type=str, choices=["PH", "PO", "AFT"], default="PH")
    
    # ---- DNN-IC hyperparameters (defaults from analysis_DNN-IC.R) ----
    p.add_argument("--dnn_epoch", type=int, default=100)
    p.add_argument("--dnn_batch_size", type=int, default=10)
    p.add_argument("--dnn_num_nodes", type=int, default=1000)
    p.add_argument("--dnn_activation", type=str, default="selu")
    p.add_argument("--dnn_l1", type=float, default=0.1)
    p.add_argument("--dnn_dropout", type=float, default=0.0)
    p.add_argument("--dnn_lr", type=float, default=0.0002)
    p.add_argument("--dnn_num_layer", type=int, choices=[1, 2], default=2)
    p.add_argument("--dnn_m", type=int, default=3)

    # Path to fun_DNN-IC.R (so we can copy/source it inside temp dir)
    p.add_argument("--dnn_fun_path", type=str, default="fun_DNN-IC.R")

    # ---- ICRF hyperparameters (defaults from icrf) ----
    p.add_argument("--icrf_ntree", type=int, default=300)
    p.add_argument("--icrf_nodesize", type=int, default=6)
    p.add_argument("--icrf_mtry", type=int, default=0)
    p.add_argument("--icrf_nfold", type=int, default=10)
    p.add_argument("--icrf_split_rule", type=str, default="Wilcoxon")
    p.add_argument("--icrf_quasihonesty", type=str, default="TRUE")
    p.add_argument("--icrf_ert", type=str, default="TRUE")
    p.add_argument("--icrf_uniform_ert", type=str, default="TRUE")
    p.add_argument("--icrf_replace", type=str, default="FALSE")
    p.add_argument("--icrf_sampsize_frac", type=float, default=0.95)
    p.add_argument("--icrf_bandwidth", type=str, default="NA")

    return p.parse_args()


# def true_quantiles(sim_model: str, q: np.ndarray, x: np.ndarray) -> np.ndarray:
#     x_rep = np.repeat(x[None, :], len(q), axis=0)
#     if sim_model == "PH":
#         return np.asarray(sinv_ph_cond(q, x_rep))
#     if sim_model == "PO":
#         return np.asarray(sinv_po_cond(q, x_rep))
#     if sim_model == "AFT":
#         return np.asarray(sinv_aft_cond(q, x_rep))
#     raise ValueError("sim_model must be PH/PO/AFT")


# def sample_times_from_survival_grid(s_hat: np.ndarray, t_grid: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
#     S = np.clip(np.asarray(s_hat), 1e-10, 1.0)
#     F = 1.0 - S
#     F = np.maximum.accumulate(F)
#     eps = 1e-12
#     for i in range(1, len(F)):
#         if F[i] <= F[i-1]:
#             F[i] = min(1.0, F[i-1] + eps)
#     u = rng.uniform(low=0.0, high=min(1.0, F[-1]), size=n)
#     return np.interp(u, F, t_grid)

def make_dnn_ic_training_df(ic_df: pd.DataFrame, p_cov: int) -> pd.DataFrame:
    """
    Convert your sim output (t,l,r,cens,x1..xp) into the DNN-IC format:
      Left, Right, status, x1..xp

    DNN-IC convention (matching analysis_DNN-IC.R):
      - status = 0 for right-censored => Right should be finite large value
      - status = 1 for interval-censored (incl. left-censored)
      - exact (uncensored) isn't explicitly handled in that script, so we
        approximate by treating it as a tiny interval [t, t+eps] with status=1.
    """
    df = ic_df.copy()

    L = df["l"].to_numpy(dtype=float)
    R = df["r"].to_numpy(dtype=float)
    cens = df["cens"].to_numpy(dtype=bool)

    eps = 1e-3

    # Identify true right-censored: interval flag TRUE AND R is +inf
    is_rc = cens & np.isinf(R)

    # status: 0 for right-censored; 1 otherwise
    status = np.ones_like(L, dtype=int)
    status[is_rc] = 0

    # For exact observations (cens==False), make them small-width intervals
    # so likelihood doesn't degenerate.
    is_exact = ~cens
    R_adj = R.copy()
    R_adj[is_exact] = L[is_exact] + eps

    # For right-censored, replace inf with finite "max finite + 0.1"
    finite_R = R_adj[np.isfinite(R_adj)]
    if finite_R.size == 0:
        max_finite = float(np.nanmax(L)) + 1.0
    else:
        max_finite = float(np.max(finite_R))
    R_adj[is_rc] = max_finite + 0.1

    out = pd.DataFrame({"Left": L, "Right": R_adj, "status": status})
    X = df[[f"x{i+1}" for i in range(p_cov)]].reset_index(drop=True)
    out = pd.concat([out, X], axis=1)
    return out


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    rep_seed = int(args.base_seed + args.rep)
    rng = np.random.default_rng(rep_seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

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

    x_test = fixed_test_covariates(args.p_cov)
    t_grid = np.linspace(0.0, args.t_max, args.m_time).astype(float)

    # Prepare temp IO for R
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        # ---- training data ----
        train_csv = td / "train.csv"
        if args.method == "dnn_ic":
            train_df = make_dnn_ic_training_df(ic_df, args.p_cov)
            train_df.to_csv(train_csv, index=False)
        else:
            ic_df.to_csv(train_csv, index=False)

        # ---- inputs ----
        tgrid_json = td / "tgrid.json"
        xtest_json = td / "xtest.json"
        out_json = td / "r_out.json"

        tgrid_json.write_text(json.dumps(t_grid.tolist()))
        xtest_json.write_text(json.dumps(x_test.tolist()))

        # If DNN-IC: copy fun_DNN-IC.R into temp dir so R can source it reliably
        fun_copy_path = None
        if args.method == "dnn_ic":
            fun_src = Path(args.dnn_fun_path)
            if not fun_src.exists():
                raise FileNotFoundError(
                    f"--dnn_fun_path not found: {fun_src}. "
                    f"Provide the path to fun_DNN-IC.R."
                )
            fun_copy_path = td / "fun_DNN-IC.R"
            fun_copy_path.write_text(fun_src.read_text())

        cmd = [
            "Rscript",
            args.r_script,
            "--method", args.method,
            "--sim_model", args.sim_model,
            "--p_cov", str(args.p_cov),
            "--train_csv", str(train_csv),
            "--tgrid_json", str(tgrid_json),
            "--xtest_json", str(xtest_json),
            "--out_json", str(out_json),
        ]

        # icenReg hyperparameters passed to R
        if args.method == "icenreg":
            cmd += [
                "--icenreg_model", args.icenreg_model,
            ]
            args.method = "icenreg" + args.icenreg_model.lower()

        # DNN-IC hyperparameters passed to R
        if args.method == "dnn_ic":
            cmd += [
                "--dnn_epoch", str(args.dnn_epoch),
                "--dnn_batch_size", str(args.dnn_batch_size),
                "--dnn_num_nodes", str(args.dnn_num_nodes),
                "--dnn_activation", str(args.dnn_activation),
                "--dnn_l1", str(args.dnn_l1),
                "--dnn_dropout", str(args.dnn_dropout),
                "--dnn_lr", str(args.dnn_lr),
                "--dnn_num_layer", str(args.dnn_num_layer),
                "--dnn_m", str(args.dnn_m),
                "--dnn_fun", str(fun_copy_path),
            ]

        # ICRF hyperparameters passed to R
        if args.method == "icrf":
            cmd += [
                "--icrf_ntree", str(args.icrf_ntree),
                "--icrf_nodesize", str(args.icrf_nodesize),
                "--icrf_mtry", str(args.icrf_mtry),
                "--icrf_nfold", str(args.icrf_nfold),
                "--icrf_split_rule", str(args.icrf_split_rule),
                "--icrf_quasihonesty", str(args.icrf_quasihonesty),
                "--icrf_ert", str(args.icrf_ert),
                "--icrf_uniform_ert", str(args.icrf_uniform_ert),
                "--icrf_replace", str(args.icrf_replace),
                "--icrf_sampsize_frac", str(args.icrf_sampsize_frac),
                "--icrf_bandwidth", str(args.icrf_bandwidth),
            ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "R method failed.\n"
                f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}\n"
            )

        r_res = json.loads(out_json.read_text())

    s_hat_mat = np.asarray(r_res["s_hat"], dtype=float)  # shape (4, m_time)
    if s_hat_mat.shape != (4, len(t_grid)):
        raise ValueError(f"Expected s_hat shape (4,{len(t_grid)}), got {s_hat_mat.shape}")

    # Subject metrics
    subj_metrics = []
    for i in range(4):
        s_hat = s_hat_mat[i, :]
        s_true = true_survival(args.sim_model, t_grid, x_test[i, :])
        err = s_hat - s_true
        imse = float(np.mean(err**2))
        bias = float(np.mean(err))
        rmse = float(np.sqrt(np.mean(err**2)))
        subj_metrics.append({
            "i": i,
            "imse": imse,
            "bias": bias,
            "rmse": rmse,
            "s_hat": s_hat.tolist(),
            "s_true": s_true.tolist(),
        })

    # # QQ metrics (sample from S_hat)
    # q = np.linspace(0.01, 0.99, 99)
    # subj_labels = ["random", "0.25", "0.50", "0.75"]
    # qq_metrics = []
    # for i in range(4):
    #     samp = sample_times_from_survival_grid(s_hat_mat[i, :], t_grid, args.n_gen_eval, rng)
    #     gen_q = np.quantile(samp, q)
    #     tru_q = true_quantiles(args.sim_model, q, x_test[i, :])
    #     qq_metrics.append({
    #         "i": i,
    #         "label": subj_labels[i],
    #         "q": q.tolist(),
    #         "true": tru_q.tolist(),
    #         "gen": gen_q.tolist(),
    #     })

    result = {
        "rep": args.rep,
        "seed": rep_seed,
        "method": r_res.get("method", args.method),
        "sim_model": args.sim_model,
        "censor": args.censor,
        "monitor": args.monitor,
        "p_cov": args.p_cov,
        "p_aux": args.p_aux,
        "t_grid": t_grid.tolist(),
        "subj_metrics": subj_metrics,
        "qq": [],  # N/A
        "vs": {},  # N/A
        "fit": r_res.get("extra", {}),
    }


    stem = f"{args.method}_{args.sim_model}_cens-{args.censor}_mon-{args.monitor}_p{args.p_cov}_pu{args.p_aux}_rep{args.rep:03d}"
    out_path = outdir / f"{stem}.json"
    out_path.write_text(json.dumps(to_jsonable(result), indent=2))


if __name__ == "__main__":
    main()

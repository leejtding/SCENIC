"""metrics.py

Evaluation utilities for the simulation study, which involves:
1) Bias/variability of survival probability estimates and IMSE,
2) Distributional checks via QQ-plots with empirical bounds,
3) Variable selection accuracy.

This module implements numerically stable, CPU/GPU-friendly metrics
using PyTorch for generation and NumPy for aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch

from sim_data import surv_ph_cond, surv_po_cond, surv_aft_cond


def true_survival(sim_model: str, t: np.ndarray, x: np.ndarray) -> np.ndarray:
    """True conditional survival S(t|x)."""
    if sim_model == "PH":
        return surv_ph_cond(t, x)
    if sim_model == "PO":
        return surv_po_cond(t, x)
    if sim_model == "AFT":
        return surv_aft_cond(t, x)
    raise ValueError("sim_model must be one of 'PH', 'PO', or 'AFT'")


def scenic_survival_estimate(
    model,
    x: torch.Tensor,
    t_grid: torch.Tensor,
    n_gen: int = 1000,
) -> torch.Tensor:
    """Estimate survival curve via SCENIC generator samples.

    Returns S_hat(t|x) on t_grid, where S_hat(t) = P(T > t).

    Parameters
    ----------
    model : SCENIC
        Trained model.
    x : torch.Tensor
        Covariate row tensor of shape (p,) or (1,p).
    t_grid : torch.Tensor
        Time grid of shape (M,).
    n_gen : int
        Number of generator draws.
    """
    model.generator.eval()
    with torch.no_grad():
        # Generate samples from generator
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x_rep = x.to(model.device).repeat(n_gen, 1)
        u = 2 * torch.rand(n_gen, model.p_aux, device=model.device) - 1
        t_samp = model.generator(u, x_rep).view(-1)  # (n_gen,)

        # S(t) = P(T > t) = mean(1{T>t})
        # Compute in a vectorized way: (n_gen, M) comparison.
        # Keep memory modest by chunking if needed.
        t_grid = t_grid.to(model.device).view(1, -1)
        surv = (t_samp.view(-1, 1) > t_grid).float().mean(dim=0)
        return surv.detach().cpu()


@dataclass
class SurvivalMetricResult:
    """Container for survival-curve metrics."""
    imse: float
    bias: float
    rmse: float
    s_hat: np.ndarray
    s_true: np.ndarray


def survival_metrics_for_subject(
    model,
    sim_model: str,
    x_test: np.ndarray,
    t_grid: np.ndarray,
    n_gen: int = 1000,
) -> SurvivalMetricResult:
    """Compute IMSE, (mean) bias, and RMSE for a single test individual."""
    t_torch = torch.tensor(t_grid, dtype=torch.float)
    x_torch = torch.tensor(x_test, dtype=torch.float)
    s_hat = scenic_survival_estimate(model, x_torch, t_torch, n_gen=n_gen).numpy()
    s_true = true_survival(sim_model, t_grid, x_test)

    err = s_hat - s_true
    imse = float(np.mean(err ** 2))
    bias = float(np.mean(err))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return SurvivalMetricResult(imse=imse, bias=bias, rmse=rmse, s_hat=s_hat, s_true=s_true)


def qq_data(
    model,
    sim_model: str,
    x_test: np.ndarray,
    n_gen: int = 1000,
) -> Dict[str, np.ndarray]:
    """Return data needed for QQ diagnostics.

    We compare generated samples from SCENIC's generator against true
    conditional quantiles from the DGP.

    Output dict contains:
      - q: probability levels
      - true: true quantiles
      - gen: generator sample quantiles
      - samples: raw generator samples (useful for other plots)
    """
    model.generator.eval()
    with torch.no_grad():
        x = torch.tensor(x_test, dtype=torch.float, device=model.device).view(1, -1)
        x_rep = x.repeat(n_gen, 1)
        u = 2 * torch.rand(n_gen, model.p_aux, device=model.device) - 1
        samp = model.generator(u, x_rep).view(-1).detach().cpu().numpy()

    # True quantiles for the three models (based on the plan's inverse CDF)
    q = np.linspace(0.01, 0.99, 99)

    # Import locally to avoid circular imports.
    from sim_data import sinv_ph_cond, sinv_po_cond, sinv_aft_cond

    if sim_model == "PH":
        true_q = sinv_ph_cond(q, np.repeat(x_test[None, :], len(q), axis=0))
    elif sim_model == "PO":
        true_q = sinv_po_cond(q, np.repeat(x_test[None, :], len(q), axis=0))
    elif sim_model == "AFT":
        true_q = sinv_aft_cond(q, np.repeat(x_test[None, :], len(q), axis=0))
    else:
        raise ValueError("sim_model must be one of 'PH', 'PO', or 'AFT'")

    gen_q = np.quantile(samp, q)
    return {"q": q, "true": np.asarray(true_q).reshape(-1), "gen": gen_q, "samples": samp}


def variable_selection_metrics(vs_mask: np.ndarray, p_cov: int) -> Dict[str, float]:
    """Variable selection accuracy.

    We treat x1 and x2 as the true signal variables.

    Parameters
    ----------
    vs_mask : np.ndarray
        Boolean mask of length p_cov where True indicates *masked out*
        (i.e., not selected) in SCENIC.
    p_cov : int
        Number of covariates.
    """
    if p_cov < 2:
        return {"prop_signal_selected": float("nan"), "n_selected": float("nan")}

    selected = ~vs_mask
    true_signal = np.zeros(p_cov, dtype=bool)
    true_signal[:2] = True
    prop_signal_selected = float(selected[true_signal].mean())
    n_selected = float(selected.sum())
    return {"prop_signal_selected": prop_signal_selected, "n_selected": n_selected}

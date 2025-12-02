"""
sim_data.py

This module provides functions to simulate survival data under different
censoring mechanisms and model assumptions:
Proportional Hazards (PH), Proportional Odds (PO), and Accelerated Failure Time (AFT).
It supports generating marginal and conditional survival times, applying
left-, right-, interval-, and uncensored mechanisms, and returning data in
a tidy pandas DataFrame format.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from math import inf
from scipy.stats import norm

### Marginal models ###

# PH
def surv_ph_marg(t, lambda_=0.5):
    """
    Marginal proportional hazards (PH) survival function.

    Parameters
    ----------
    t : array-like
        Time points.
    lambda_ : float, optional
        Baseline hazard rate (default is 0.1).

    Returns
    -------
    ndarray
        Survival probabilities at each time point.
    """
    return np.exp(-lambda_ * np.asarray(t))

def sinv_ph_marg(u, lambda_=0.5):
    """
    Inverse CDF for the marginal PH model.

    Parameters
    ----------
    u : array-like
        Uniform(0,1) random draws.
    lambda_ : float, optional
        Baseline hazard rate.

    Returns
    -------
    ndarray
        Simulated survival times.
    """
    return -np.log(np.asarray(1 - u)) / lambda_


# PO
def surv_po_marg(t):
    """
    Marginal proportional odds (PO) survival function.

    Parameters
    ----------
    t : array-like
        Time points.

    Returns
    -------
    ndarray
        Survival probabilities at each time point.
    """
    return 1 / (1 + np.asarray(t))

def sinv_po_marg(u):
    """
    Inverse CDF for the marginal PO model.

    Parameters
    ----------
    u : array-like
        Uniform(0,1) random draws.

    Returns
    -------
    ndarray
        Simulated survival times.
    """
    return (1 / (1 - np.asarray(1 - u))) - 1


# AFT (log-normal)
def surv_aft_marg(t):
    """
    Marginal log-normal AFT survival function.

    Parameters
    ----------
    t : array-like
        Time points.

    Returns
    -------
    ndarray
        Survival probabilities at each time point.
    """
    return 1 - norm.cdf(np.log(np.asarray(t)))

def sinv_aft_marg(u):
    """
    Inverse CDF for the marginal AFT model.

    Parameters
    ----------
    u : array-like
        Uniform(0,1) random draws.

    Returns
    -------
    ndarray
        Simulated survival times.
    """
    return np.exp(norm.ppf(np.asarray(1 - u)))


### Conditional models ###

# Covariates function
def f_x(x, r):
    """
    Covariate function controlling effect strength.

    Parameters
    ----------
    x : ndarray
        Covariate matrix or vector.
    r : float
        Covariate effect scale.

    Returns
    -------
    ndarray
        Computed covariate effects.
    """
    if (x.ndim == 1):
        return -(x[0]**2 + x[1]**2) / (2 * r**2)
    else:
        return -(x[:,0]**2 + x[:,1]**2) / (2 * r**2)


# PH
def surv_ph_cond(t, x, r = 0.5, lambda_=0.5):
    """
    Conditional survival function for PH model.

    Parameters
    ----------
    t : array-like
        Time points.
    x : ndarray
        Covariate matrix or vector.
    r : float, optional
        Covariate effect scale (default 0.5).
    lambda_ : float, optional
        Baseline hazard rate (default 0.1).

    Returns
    -------
    ndarray
        Conditional survival probabilities.
    """
    return np.exp(-lambda_ * np.exp(f_x(x, r)) * np.asarray(t))

def sinv_ph_cond(u, x, r = 0.5, lambda_=0.5):
    """
    Inverse conditional CDF for PH model.

    Parameters
    ----------
    u : array-like
        Uniform(0,1) random draws.
    x : ndarray
        Covariate matrix or vector.
    r : float, optional
        Covariate effect scale.
    lambda_ : float, optional
        Baseline hazard rate.

    Returns
    -------
    ndarray
        Simulated conditional survival times.
    """
    return -np.log(np.asarray(1 - u)) / (lambda_ * np.exp(f_x(x, r)))


# PO
def surv_po_cond(t, x, r = 0.5):
    """
    Conditional survival function for PO model.

    Parameters
    ----------
    t : array-like
        Time points.
    x : ndarray
        Covariate matrix or vector.
    r : float, optional
        Covariate effect scale (default 0.5).

    Returns
    -------
    ndarray
        Conditional survival probabilities.
    """
    return 1 / (1 + np.asarray(t) * np.exp(f_x(x, r)))

def sinv_po_cond(u, x, r = 0.5):
    """
    Inverse conditional CDF for PO model.

    Parameters
    ----------
    u : array-like
        Uniform(0,1) random draws.
    x : ndarray
        Covariate matrix or vector.
    r : float, optional
        Covariate effect scale (default 0.5).

    Returns
    -------
    ndarray
        Simulated conditional survival times.
    """
    return ((1 / (1 - np.asarray(1 - u))) - 1) / np.exp(f_x(x, r))


# AFT (log-normal)
def surv_aft_cond(t, x, r = 0.5):
    """
    Conditional survival function for AFT model.

    Parameters
    ----------
    t : array-like
        Time points.
    x : ndarray
        Covariate matrix or vector.
    r : float, optional
        Covariate effect scale.

    Returns
    -------
    ndarray
        Conditional survival probabilities.
    """
    return 1 - norm.cdf(np.log(np.asarray(t)) + f_x(x, r))

def sinv_aft_cond(u, x, r = 0.5):
    """
    Inverse conditional CDF for AFT model.

    Parameters
    ----------
    u : array-like
        Uniform(0,1) random draws.
    x : ndarray
        Covariate matrix or vector.
    r : float, optional
        Covariate effect scale.

    Returns
    -------
    ndarray
        Simulated conditional survival times.
    """
    return np.exp(norm.ppf(np.asarray(1 - u)) - f_x(x, r))


### Censoring mechanisms ###

def gen_left_cens(n, tau):
    """
    Generate left-censoring intervals.

    Parameters
    ----------
    n : int
        Number of censored observations.
    tau : float
        Upper limit for censoring times.

    Returns
    -------
    ndarray
        Matrix with shape (2, n) for left- and right-censoring bounds.
    """
    U = np.random.uniform(0, tau, n)
    return np.vstack([np.zeros(n), U])


def gen_right_cens(n, tau):
    """
    Generate right-censoring intervals.

    Parameters
    ----------
    n : int
        Number of censored observations.
    tau : float
        Lower limit for right-censoring.

    Returns
    -------
    ndarray
        Matrix with shape (2, n) for left- and right-censoring bounds.
    """
    U = np.random.uniform(0, tau, n)
    return np.vstack([U, np.full(n, np.inf)])


def gen_int_cens(t, c, d, k):
    """
    Generate inspection intervals for interval censoring.

    Parameters
    ----------
    t : float
        True event time.
    c : float
        Inspection spacing.
    d : float
        Random offset range.
    k : int
        Number of inspection intervals.

    Returns
    -------
    tuple of float
        (left_bound, right_bound) interval around event time.
    """
    u1 = np.random.uniform(0, d)
    uh = [u1 + h * c + np.random.uniform(0, d) for h in range(1, k)]
    schedule = np.array([0] + [u1] + uh + [np.inf], dtype=float)
    idx = np.searchsorted(schedule, t, side="right") - 1
    return float(schedule[idx]), float(schedule[idx + 1])


### Convert to observed interval ###

@dataclass
class LRResult:
    """Container for left/right censoring results."""
    l: float
    r: float
    ind: bool  # True if interval-censored, False if exact


def convert_lr(t, cens):
    """
    Convert censoring interval to observed left/right times.

    Parameters
    ----------
    t : float
        True event time.
    cens : tuple of float
        (left, right) censoring bounds.

    Returns
    -------
    LRResult
        Object containing left, right, and censoring indicator.
    """
    l_c, r_c = float(cens[0]), float(cens[1])
    if (t < l_c) or (t >= r_c):
        return LRResult(l=t, r=t, ind=False)
    else:
        return LRResult(l=l_c, r=r_c, ind=True)


### Simulate data ###

def sim_ic_marg(n_lc=0, n_rc=0, n_ic=0, n_uc=0, tau_lc=1.0, tau_rc=50.0,
                c_ic=5.0, d_ic=5.0, k_ic=30, sim_model="PH", lambda_=0.5,
                seed=None):
    """
    Simulate marginally distributed survival data under censoring.

    Parameters
    ----------
    n_lc, n_rc, n_ic, n_uc : int
        Number of left-, right-, interval-, and uncensored observations.
    tau_lc, tau_rc : float
        Censoring bounds for left and right censoring.
    c_ic, d_ic, k_ic : float or int
        Parameters controlling interval-censoring.
    sim_model : {"PH", "PO", "AFT"}
        Model type for simulation.
    lambda_ : float, optional
        Baseline hazard rate for PH model.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        DataFrame with survival times, censoring bounds, and indicator.
    """
    if seed is not None:
        np.random.seed(seed)

    sim_cdf_ac = np.random.uniform(0, 1, n_lc + n_rc + n_ic)
    sim_cdf_uc = np.random.uniform(0, 1, n_uc)

    if sim_model == "PH":  # PH
        sim_t_ac = sinv_ph_marg(sim_cdf_ac, lambda_)
        sim_t_uc = sinv_ph_marg(sim_cdf_uc, lambda_)
    elif sim_model == "PO":  # PO
        sim_t_ac = sinv_po_marg(sim_cdf_ac)
        sim_t_uc = sinv_po_marg(sim_cdf_uc)
    elif sim_model == "AFT":  # AFT
        sim_t_ac = sinv_aft_marg(sim_cdf_ac)
        sim_t_uc = sinv_aft_marg(sim_cdf_uc)
    else:
        raise ValueError("sim_model must be one of 'PH', 'PO', or 'AFT'")

    sim_t, sim_l, sim_r, sim_ind = [], [], [], []

    # Interval-censored
    for i in range(n_ic):
        l, r = gen_int_cens(sim_t_ac[i], c_ic, d_ic, k_ic)
        res = convert_lr(sim_t_ac[i], (l, r))
        sim_t.append(sim_t_ac[i])
        sim_l.append(res.l)
        sim_r.append(res.r)
        sim_ind.append(res.ind)

    # Right-censored
    rc_mat = gen_right_cens(n_rc, tau_rc)
    for i in range(n_rc):
        tval = sim_t_ac[n_ic + i]
        res = convert_lr(tval, rc_mat[:, i])
        sim_t.append(tval)
        sim_l.append(res.l)
        sim_r.append(res.r)
        sim_ind.append(res.ind)

    # Left-censored
    lc_mat = gen_left_cens(n_lc, tau_lc)
    for i in range(n_lc):
        tval = sim_t_ac[n_ic + n_rc + i]
        res = convert_lr(tval, lc_mat[:, i])
        sim_t.append(tval)
        sim_l.append(res.l)
        sim_r.append(res.r)
        sim_ind.append(res.ind)

    # Uncensored
    sim_t.extend(sim_t_uc)
    sim_l.extend(sim_t_uc)
    sim_r.extend(sim_t_uc)
    sim_ind.extend([False] * n_uc)

    return pd.DataFrame({"t": sim_t, "l": sim_l, "r": sim_r, "cens": sim_ind})


def sim_ic_cond(n_lc=0, n_rc=0, n_ic=0, n_uc=0, p_cov=2, r_cov=0.5, tau_lc=5.0,
                tau_rc=50.0, c_ic=5.0, d_ic=5.0, k_ic=20, sim_model="PH",
                lambda_=0.5, seed=3):
    """
    Simulate conditional survival data given covariates.

    Parameters
    ----------
    n_lc, n_rc, n_ic, n_uc : int
        Number of left-, right-, interval-, and uncensored observations.
    p_cov : int
        Number of covariates.
    r_cov : float
        Covariate effect scale.
    tau_lc, tau_rc, c_ic, d_ic, k_ic : float or int
        Parameters controlling censoring schedule.
    sim_model : {"PH", "PO", "AFT"}
        Model type for simulation.
    lambda_ : float
        Baseline hazard rate (for PH model only).
    seed : int
        Random seed.

    Returns
    -------
    pandas.DataFrame
        DataFrame with simulated event times, censoring bounds, indicator, and covariates.
    """
    if seed is not None:
        np.random.seed(seed)

    sim_cdf_ac = np.random.uniform(0, 1, n_lc + n_rc + n_ic)
    sim_cdf_uc = np.random.uniform(0, 1, n_uc)
    sim_x_ac = np.random.uniform(-1, 1, (n_lc + n_rc + n_ic, p_cov))
    sim_x_uc = np.random.uniform(-1, 1, (n_uc, p_cov))
    sim_x = np.vstack([sim_x_ac, sim_x_uc])

    if sim_model == "PH":  # PH
        sim_t_ac = sinv_ph_cond(sim_cdf_ac, sim_x_ac, r_cov, lambda_)
        sim_t_uc = sinv_ph_cond(sim_cdf_uc, sim_x_uc, r_cov, lambda_)
    elif sim_model == "PO":  # PO
        sim_t_ac = sinv_po_cond(sim_cdf_ac, sim_x_ac, r_cov)
        sim_t_uc = sinv_po_cond(sim_cdf_uc, sim_x_uc, r_cov)
    elif sim_model == "AFT":  # AFT
        sim_t_ac = sinv_aft_cond(sim_cdf_ac, sim_x_ac, r_cov)
        sim_t_uc = sinv_aft_cond(sim_cdf_uc, sim_x_uc, r_cov)
    else:
        raise ValueError("sim_model must be one of 'PH', 'PO', or 'AFT'")

    sim_t, sim_l, sim_r, sim_ind = [], [], [], []

    # Interval-censored
    for i in range(n_ic):
        l, r = gen_int_cens(sim_t_ac[i], c_ic, d_ic, k_ic)
        res = convert_lr(sim_t_ac[i], (l, r))
        sim_t.append(sim_t_ac[i])
        sim_l.append(res.l)
        sim_r.append(res.r)
        sim_ind.append(res.ind)

    # Right-censored
    rc_mat = gen_right_cens(n_rc, tau_rc)
    for i in range(n_rc):
        tval = sim_t_ac[n_ic + i]
        res = convert_lr(tval, rc_mat[:, i])
        sim_t.append(tval)
        sim_l.append(res.l)
        sim_r.append(res.r)
        sim_ind.append(res.ind)

    # Left-censored
    lc_mat = gen_left_cens(n_lc, tau_lc)
    for i in range(n_lc):
        tval = sim_t_ac[n_ic + n_rc + i]
        res = convert_lr(tval, lc_mat[:, i])
        sim_t.append(tval)
        sim_l.append(res.l)
        sim_r.append(res.r)
        sim_ind.append(res.ind)

    # Uncensored
    sim_t.extend(sim_t_uc)
    sim_l.extend(sim_t_uc)
    sim_r.extend(sim_t_uc)
    sim_ind.extend([False] * n_uc)

    ret_df = pd.DataFrame({"t": sim_t, "l": sim_l, "r": sim_r, "cens": sim_ind})
    ret_df = pd.concat([ret_df,
                        pd.DataFrame(sim_x,
                                     columns=[f"x{i+1}" for i in range(p_cov)])], axis=1)
    return ret_df


### Example usage ###
if __name__ == "__main__":
    df = sim_ic_cond(n_lc=10, n_rc=0, n_ic=0, n_uc=0, model=1)
    print(df.head())
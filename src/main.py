"""
main.py

Entry point for running SCENIC simulations and model training.
This script handles command-line arguments, data simulation,
model initialization, training, and visualization.

Functions
---------
parse_args()
    Define and parse command-line arguments.
sim_scenic(...)
    Simulate data, initialize SCENIC, train for a specified number of epochs,
    and periodically visualize estimated vs. true survival curves.
main()
    Parse arguments and execute the SCENIC training pipeline.
"""

from __future__ import annotations
import argparse
import time
from tracemalloc import start
import torch

from models import *
from train import *
from sim_data import *
from print_results import *

def parse_args():
    """
    Define and parse command-line arguments for SCENIC training.

    Returns
    -------
    argparse.Namespace
        Parsed arguments including simulation, model, and training parameters.
    """
    p = argparse.ArgumentParser(description="Train SCENIC model on simulated interval-censored data.")

    # Data simulation parameters
    p.add_argument("--n_lc", type=float, default=0)
    p.add_argument("--n_rc", type=float, default=0)
    p.add_argument("--n_ic", type=float, default=0)
    p.add_argument("--n_uc", type=float, default=100)
    p.add_argument("--p_cov", type=int, default=2)
    p.add_argument("--r_cov", type=float, default=0.5)
    p.add_argument("--tau_lc", type=float, default=5.0)
    p.add_argument("--tau_rc", type=float, default=50.0)
    p.add_argument("--c_ic", type=float, default=5.0)
    p.add_argument("--d_ic", type=float, default=5.0)
    p.add_argument("--k_ic", type=int, default=20)
    p.add_argument("--sim_model", type=str, default="PH")
    p.add_argument("--lambda_", type=float, default=0.1)

    # Model and training parameters
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--phi_steps", type=int, default=1)
    p.add_argument("--gen_steps", type=int, default=1)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--n_aux", type=int, default=400)
    p.add_argument("--p_aux", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=5)
    p.add_argument("--train_size", type=int, default=100)
    p.add_argument("--temp", type=float, default=0.1)
    p.add_argument("--temp_decay", type=float, default=1.0)
    p.add_argument("--gen_hidden", type=int, default=100)
    p.add_argument("--phi_hidden", type=int, default=1000)
    p.add_argument("--gen_lr", type=float, default=1e-4)
    p.add_argument("--phi_lr", type=float, default=1e-5)

    # Plotting parameters
    p.add_argument("--n_plot", type=int, default=6)
    p.add_argument("--ub_plot", type=float, default=50)
    p.add_argument("--steps_plot", type=int, default=500)
    p.add_argument("--n_gen", type=int, default=2000)
    return p.parse_args()

def sim_scenic(n_lc=0, n_rc=0, n_ic=0, n_uc=0, p_cov=2, r_cov=0.5, tau_lc=5.0,
               tau_rc=50.0, c_ic=5.0, d_ic=5.0, k_ic=20, sim_model="PH",
               lambda_=0.1, epochs=100, phi_steps=1, gen_steps=1, device='cpu',
               seed=3, n_aux=400, p_aux=5, batch_size=5, train_size=100,
               temp=0.1, temp_decay=1.0, gen_hidden=100, phi_hidden=1000,
               gen_lr=1e-4, phi_lr=1e-5, n_plot=6, ub_plot=50, steps_plot=500,
               n_gen=2000):
    """
    Simulate data, train a SCENIC model, and visualize progress.

    Parameters
    ----------
    n_lc, n_rc, n_ic, n_uc : int
        Numbers of left-, right-, interval-, and uncensored samples.
    p_cov : int
        Number of covariates.
    r_cov : float
        Covariate effect scale.
    tau_lc, tau_rc : float
        Left/right censoring bounds.
    c_ic, d_ic : float
        Parameters controlling the spacing and variability of inspection times.
    k_ic : int
        Number of inspection times for interval censoring.
    sim_model : {"PH", "PO", "AFT"}
        Simulation model type.
    lambda_ : float
        Baseline hazard rate (PH only).
    epochs : int
        Number of training epochs.
    phi_steps, gen_steps : int
        Number of updates per mini-batch iteration for φ and generator networks.
    device : str
        Computation device.
    seed : int
        Random seed.
    n_aux, p_aux : int
        Auxiliary sample size and latent noise dimensionality.
    batch_size, train_size : int
        Mini-batch and subset sizes for training.
    temp, temp_decay : float
        Soft indicator temperature and decay factor.
    gen_hidden, phi_hidden : int
        Hidden layer sizes.
    gen_lr, phi_lr : float
        Learning rates.
    n_plot, ub_plot, steps_plot, n_gen : int
        Plotting parameters for visualizing training results.

    Notes
    -----
    - Data are generated using `sim_ic_cond` from `sim_data.py`.
    - The model alternates between φ- and generator-updates via `train`.
    - Plots are generated every 10 epochs using `print_results`.
    """
    # Simulate data
    np.random.seed(seed)
    ic_df = sim_ic_cond(n_lc=n_lc, n_rc=n_rc, n_ic=n_ic, n_uc=n_uc,
                        p_cov=p_cov, r_cov=r_cov, tau_lc=tau_lc,
                        tau_rc=tau_rc, c_ic=c_ic, d_ic=d_ic, k_ic=k_ic,
                        sim_model=sim_model, lambda_=lambda_, seed=seed)
    l = torch.tensor(ic_df["l"])
    r = torch.tensor(ic_df["r"])
    X = torch.tensor(ic_df.iloc[:, 4:(4 + p_cov)].values)
    
    # Initialize SCENIC model
    scenic = SCENIC(l=l, r=r, X=X, device=device, n_aux=n_aux,
                    p_aux=p_aux, batch_size=batch_size,
                    train_size=train_size, temp=temp,
                    temp_decay=temp_decay, gen_hidden=gen_hidden,
                    phi_hidden=phi_hidden, gen_lr=gen_lr, phi_lr=phi_lr)
    
    # Train SCENIC model
    start = time.time()
    for epoch in range(epochs):
        train(scenic, phi_steps=phi_steps, gen_steps=gen_steps)

        if (epoch + 1) % 10 == 0:
            print_results(scenic, epoch=epoch + 1, sim_model=sim_model,
                          n_plot=n_plot, ub_plot=ub_plot, steps_plot=steps_plot,
                          n_gen=n_gen)

    end = time.time()
    print(f"Training time: {end - start} seconds")


def main():
    """
    Main entry point for running SCENIC simulations and training.

    Steps
    -----
    1. Parse command-line arguments.
    2. Simulate interval-censored data using `sim_ic_cond`.
    3. Initialize and train SCENIC for the specified epochs.
    4. Periodically visualize results.
    """
    args = parse_args()
    sim_scenic(n_lc=args.n_lc, n_rc=args.n_rc, n_ic=args.n_ic, n_uc=args.n_uc,
               p_cov=args.p_cov, sim_model=args.model, epochs=args.epochs,
               phi_steps=args.phi_steps, gen_steps=args.gen_steps,
               device=args.device, seed=args.seed, n_aux=args.n_aux,
               p_aux=args.p_aux, batch_size=args.batch_size,
               train_size=args.train_size, temp=args.temp,
               temp_decay=args.temp_decay, gen_hidden=args.gen_hidden,
               phi_hidden=args.phi_hidden, gen_lr=args.gen_lr, phi_lr=args.phi_lr,
               n_plot=args.n_plot, ub_plot=args.ub_plot, steps_plot=args.steps_plot,
               n_gen=args.n_gen)
    
if __name__ == '__main__':
    main()
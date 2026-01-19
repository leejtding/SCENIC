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
import os
import json
import random
import platform
import numpy as np
from datetime import datetime

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
    p.add_argument("--n_lc", type=float, default=0,
                   help="Number of left-censored samples")
    p.add_argument("--n_rc", type=float, default=0,
                   help="Number of right-censored samples")
    p.add_argument("--n_ic", type=float, default=0,
                   help="Number of interval-censored samples")
    p.add_argument("--n_uc", type=float, default=100,
                   help="Number of uncensored samples")
    p.add_argument("--p_cov", type=int, default=2,
                   help="Number of covariates")
    p.add_argument("--r_cov", type=float, default=0.5,
                   help="Correlation of covariates")
    p.add_argument("--tau_lc", type=float, default=5.0,
                   help="Left-censoring threshold")
    p.add_argument("--tau_rc", type=float, default=50.0,
                   help="Right-censoring threshold")
    p.add_argument("--c_ic", type=float, default=5.0,
                   help="Left interval-censoring threshold")
    p.add_argument("--d_ic", type=float, default=5.0,
                   help="Right interval-censoring threshold")
    p.add_argument("--k_ic", type=int, default=20,
                   help="Number of interval-censored samples")
    p.add_argument("--p_ic", type=float, default=0.5,
                   help="Proportion of interval-censored samples")
    p.add_argument("--sim_model", type=str, default="PH",
                   help="Data generating process model")
    p.add_argument("--lambda_", type=float, default=0.1,
                   help="Baseline hazard parameter")

    # Model and training parameters
    p.add_argument("--vs", type=str, default="F",
                   help="Variable selection")
    p.add_argument("--epochs", type=int, default=100, 
                   help="Total training epochs")
    p.add_argument("--after_vs_epochs", type=int, default=20,
                   help="Epochs after which variable selection is performed")
    p.add_argument("--phi_steps", type=int, default=1,
                   help="Number of phi steps per iteration")
    p.add_argument("--gen_steps", type=int, default=1,
                   help="Number of generator steps per iteration")
    p.add_argument("--device", type=str, default="cpu",
                   help="Device to run the model on")
    p.add_argument("--seed", type=int, default=3,
                   help="Random seed for reproducibility")
    p.add_argument("--n_aux", type=int, default=500,
                   help="Number of auxiliary variables")
    p.add_argument("--p_aux", type=int, default=5,
                   help="Dimension of auxiliary variables")
    p.add_argument("--batch_size", type=int, default=5,
                   help="Batch size for training")
    p.add_argument("--train_size", type=int, default=100,
                   help="Training size for mini-batch")
    p.add_argument("--temp", type=float, default=0.1,
                   help="Temperature parameter for sampling")
    p.add_argument("--temp_decay", type=float, default=1.0,
                   help="Decay rate for temperature parameter")
    p.add_argument("--gen_hidden", type=int, default=100,
                   help="Number of hidden units in generator")
    p.add_argument("--phi_hidden", type=int, default=1000,
                   help="Number of hidden units in phi network")
    p.add_argument("--gen_lr", type=float, default=1e-4,
                   help="Learning rate for generator")
    p.add_argument("--phi_lr", type=float, default=1e-5,
                   help="Learning rate for phi network")

    # Plotting parameters
    p.add_argument("--n_plot", type=int, default=6,
                   help="Number of plots to generate")
    p.add_argument("--ub_plot", type=float, default=50,
                   help="Upper bound for plots")
    p.add_argument("--steps_plot", type=int, default=500,
                   help="Steps between plots")
    p.add_argument("--n_gen", type=int, default=2000,
                   help="Number of samples to generate")
    return p.parse_args()


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

    # Simulate data
    np.random.seed(args.seed)
    ic_df = sim_ic_cond(n_lc=args.n_lc, n_rc=args.n_rc, n_ic=args.n_ic,
                        n_uc=args.n_uc, p_cov=args.p_cov, r_cov=args.r_cov,
                        tau_lc=args.tau_lc, tau_rc=args.tau_rc, c_ic=args.c_ic,
                        d_ic=args.d_ic, k_ic=args.k_ic, p_ic = args.p_ic,
                        sim_model=args.sim_model, lambda_=args.lambda_,
                        seed=args.seed)
    l = torch.tensor(ic_df["l"])
    r = torch.tensor(ic_df["r"])
    X = torch.tensor(ic_df.iloc[:, 4:(4 + args.p_cov)].values)

    # Initialize model
    scenic = SCENIC(l=l, r=r, X=X, n_aux=args.n_aux, p_aux=args.p_aux,
                    train_size = args.train_size, batch_size=args.batch_size,
                    device=args.device, temp=args.temp, temp_decay=args.temp_decay,
                    gen_hidden=args.gen_hidden, phi_hidden=args.phi_hidden,
                    gen_lr=args.gen_lr, phi_lr=args.phi_lr)
    
    # Train model and visualize results
    arg_vs = True if args.vs == "T" else False
    train(scenic, vs=arg_vs, total_epochs=args.epochs,
          after_vs_epochs=args.after_vs_epochs, gen_steps=args.gen_steps,
          sim_model=args.sim_model, n_plot=args.n_plot, ub_plot=args.ub_plot,
          steps_plot=args.steps_plot, n_gen=args.n_gen)

if __name__ == '__main__':
    main()
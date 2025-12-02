"""
train.py

Training loop utilities for the SCENIC model. This module provides a
single function, `train`, which performs one epoch of alternating
optimization over the phi network and the generator using mini-batch SGD.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
import random

from models import SCENIC
from print_results import print_results

def train_epoch(model, phi_steps: int = 1, gen_steps: int = 1):
    """
    Run one training epoch for the SCENIC model.

    The procedure alternates between:
    1) Updating the phi network (`phi_steps` times) to maximize the loss gap,
    2) Updating the generator (`gen_steps` times) to minimize the loss,
    using mini-batch stochastic optimization. Inner time grids are formed
    by unique, finite censoring endpoints from a random training subset.

    Parameters
    ----------
    model : SCENIC
        An initialized SCENIC model instance with tensors and optimizers set.
    phi_steps : int, optional
        Number of phi network updates per mini-batch iteration (default is 1).
    gen_steps : int, optional
        Number of generator updates per mini-batch iteration (default is 1).

    Notes
    -----
    - The inner time grid is built from unique values in the union of `l` and
      `r` from a random subset (`model.train_size`) of indices, excluding 0
      and infinity.
    - Mini-batch indices are a random permutation of all `n` observations.
    - `model.iterations_epoch` controls the number of mini-batch iterations and
      is computed as `n // batch_size` at model construction.
    """
    # Initialize training
    model.generator.train()
    model.phi.train()

    # Get innermost time points for building conditional CDF
    train_inds = np.random.choice(range(model.n), model.train_size,
                                  replace=False)
    l_train = model.l[train_inds].to(model.device)
    r_train = model.r[train_inds].to(model.device)
    inner_train = torch.unique(torch.cat((l_train, r_train)), sorted=True)
    inner_train = inner_train[(inner_train != 0) &
                                (~torch.isinf(inner_train))]

    # Train with mini-batch SGD
    batch_inds = np.random.choice(range(model.n), model.n, replace=False)

    for iter in range(model.iterations_epoch):
        # 1. Draw mini batch and construct inner time grid & phi inputs
        b_i = batch_inds[iter*model.batch_size:
                        (iter*model.batch_size+model.batch_size)]

        X_batch = model.X[b_i, :].repeat(1, model.n_aux)
        X_batch = X_batch.reshape(model.batch_size*model.n_aux, model.p)
        l_batch = model.l[b_i].to(model.device)
        r_batch = model.r[b_i].to(model.device)

        inner_batch = torch.unique(torch.cat((l_batch, r_batch, inner_train)))
        inner_batch = inner_batch[(inner_batch != 0) &
                                  (~torch.isinf(inner_batch))]
        n_inner = inner_batch.shape[0]

        phi_X = model.X[b_i, :].repeat(1, n_inner)
        phi_X = phi_X.reshape(model.batch_size*n_inner, model.p)
        phi_t = inner_batch.repeat(model.batch_size, 1)
        phi_t = phi_t.reshape(model.batch_size*n_inner, 1)

        # 2. Update phi based on loss from generated conditional CDF
        for _ in range(phi_steps):
            model.step_phi(inner_batch, X_batch, l_batch, r_batch,
                           phi_X, phi_t)

        # 3. Update generator based on loss from generated conditional CDF
        for _ in range(gen_steps):
            model.step_generator(inner_batch, X_batch, l_batch, r_batch,
                                 phi_X, phi_t)

    # Decay temperature parameter if applicable
    model.temp = max(0.01, model.temp * model.temp_decay)


def train(model, vs=False, total_epochs=100, after_vs_epochs=20,
          phi_steps=1, gen_steps=1, sim_model="PH", n_plot=6, ub_plot=50,
          steps_plot=500, n_gen=2000):
    """
    Train model and visualize progress across epochs.

    Parameters
    ----------
    model : SCENIC
        An initialized SCENIC model instance with tensors and optimizers set.
    epochs : int
        Number of training epochs.
    phi_steps, gen_steps : int
        Number of updates per mini-batch iteration for phi and generator networks.
    sim_model : {"PH", "PO", "AFT"}
        Model type for visualizing training results.
    n_plot, ub_plot, steps_plot, n_gen : int
        Plotting parameters for visualizing training results.

    Notes
    -----
    - The model alternates between phi- and generator-updates via `train`.
    - Plots are generated every 10 epochs using `print_results`.
    """
    # Train model across epochs
    epoch = 0
    max_epoch = total_epochs
    n_vs = 0

    while epoch < max_epoch:
        epoch += 1

        # Check informative weights before variable selection
        if (vs and not model.informative_check and epoch > 5):
            model.informative_check = model.check_informative_weights()
        
        # Check epoch condition for variable selection
        if (vs and epoch >= total_epochs):
            model.epoch_check = True

        # Perform variable selection if conditions are met
        model.selected = model.epoch_check or model.informative_check

        if model.selected and n_vs == 0:
            model.variable_selection()
            max_epoch = epoch + after_vs_epochs
            n_vs += 1

        # Train for one epoch
        train_epoch(model, phi_steps=phi_steps, gen_steps=gen_steps)

        # Apply variable selection mask to generator weights
        if vs and model.selected:
            with torch.no_grad():
                model.generator.gen1.weight[:, model.p_aux:][:, model.vs_mask] = 0

        # Print results every 10 epochs
        if epoch % 10 == 0:
            # Print current training losses and plot results
            print_results(model, epoch=epoch, sim_model=sim_model,
                          n_plot=n_plot, ub_plot=ub_plot, steps_plot=steps_plot,
                          n_gen=n_gen)
            
            # Print variable selection diagnostics
            if vs:
                with torch.no_grad():
                    w1 = torch.matmul(torch.abs(model.generator.gen2.weight.data),
                                    torch.abs(model.generator.gen1.weight.data))
                    w2 = torch.matmul(torch.abs(model.generator.gen3.weight.data), w1)
                    w3 = torch.matmul(torch.abs(model.generator.gen4.weight.data), w2)
                    print("Auxiliary signal:",
                          torch.round(w3.view(-1)[:model.p_aux][:model.p],
                                      decimals=2))
                    print("X signal:",
                          torch.round(w3.view(-1)[model.p_aux:][:model.p],
                                      decimals=2))
                    print("VS start:", model.selected,
                          "Number of signals:", model.vs_Xidx.sum().item())
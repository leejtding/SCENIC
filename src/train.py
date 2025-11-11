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

def train(model, phi_steps: int = 1, gen_steps: int = 1):
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
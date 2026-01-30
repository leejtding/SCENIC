"""
loss.py

Defines loss-related utility functions for the SCENIC framework.
These functions compute left and right components of the conditional
CDF-based loss used to train the generator and phi networks on
interval-censored survival data.
"""

from __future__ import annotations
import torch
import torch.nn as nn

def get_cdf(F, inner, t):
    """
    Retrieve the CDF value at a specific time point.

    Parameters
    ----------
    F : torch.Tensor
        Conditional CDF values at grid points, shape (m,).
    inner : torch.Tensor
        Grid of time points corresponding to `F`, shape (m,).
    t : float
        Time value to retrieve the CDF for.

    Returns
    -------
    torch.Tensor
        Scalar tensor containing the CDF value at `t`.

    Notes
    -----
    Returns 0 if `t == 0`, 1 if `t == inf`, and the matching value otherwise.
    """
    # Get CDF values at time points t based on F_batch
    if (t == 0):
        F_t = 0
    elif torch.isinf(t):
        F_t = 1
    else:
        F_t = F[(inner == t).nonzero(as_tuple=True)[0].item()]
    return F_t

def C_right1(inner_batch, F_batch, l_batch, r_batch, b_i):
    """
    Compute the first right-hand term of the loss for batch index `b_i`.

    Parameters
    ----------
    inner_batch : torch.Tensor
        1D tensor of inner time grid points, shape (m,).
    F_batch : torch.Tensor
        Conditional CDFs, shape (batch_size, m).
    l_batch : torch.Tensor
        Left interval bounds, shape (batch_size,).
    r_batch : torch.Tensor
        Right interval bounds, shape (batch_size,).
    b_i : int
        Batch index.

    Returns
    -------
    torch.Tensor
        Tensor of shape (m,) representing the right-side contribution for index `b_i`.
    """
    # Get observation times and conditional CDF for batch index b_i
    F_batch_i = F_batch[b_i,:]
    l_batch_i = l_batch[b_i]
    r_batch_i = r_batch[b_i]

    # Compute C_right1 in loss function
    Fl = get_cdf(F_batch_i, inner_batch, l_batch_i)
    Fr = get_cdf(F_batch_i, inner_batch, r_batch_i)
    ind = ((inner_batch > l_batch_i) & (inner_batch < r_batch_i)).float()
    output = F_batch_i - Fl
    output = (output / (Fr - Fl + 1e-8)) * ind
    return output


def C_right2(inner_batch, r_batch, b_i):
    """
    Compute the second right-hand term of the loss for batch index `b_i`.

    Parameters
    ----------
    inner_batch : torch.Tensor
        1D tensor of inner time grid points, shape (m,).
    r_batch : torch.Tensor
        Right interval bounds, shape (batch_size,).
    b_i : int
        Batch index.

    Returns
    -------
    torch.Tensor
        Indicator tensor of shape (m,) with 1 where `r_batch[b_i] <= inner_batch`, else 0.
    """
    # Compute C_right2 in loss function
    ind = (r_batch[b_i] <= inner_batch).float()
    return ind


def C_loss(inner_batch, l_batch, r_batch, F_batch, phi_batch, batch_size):
    """
    Compute left and right loss components for interval-censored samples.

    Parameters
    ----------
    inner_batch : torch.Tensor
        1D tensor of sorted inner time points (m,).
    l_batch : torch.Tensor
        Left censoring times, shape (batch_size,).
    r_batch : torch.Tensor
        Right censoring times, shape (batch_size,).
    F_batch : torch.Tensor
        Conditional CDF values for each batch, shape (batch_size, m).
    phi_batch : torch.Tensor
        Phi-network outputs for each batch, shape (batch_size, m).
    batch_size : int
        Mini-batch size.

    Returns
    -------
    tuple of torch.Tensor
        (C_left, C_right) tensors, each of shape (m,), representing
        averaged contributions across the batch.

    Notes
    -----
    - C_left is the mean of F * phi.
    - C_right combines both right-hand components (C_right1, C_right2)
      weighted by phi outputs.
    """
    # Compute C_left and C_right for the batch
    C_left = (F_batch * phi_batch).mean(dim=0)
    Cr1 = map(lambda b_i: C_right1(inner_batch, F_batch, l_batch, r_batch,
                                        b_i),
                   range(batch_size))
    Cr1 = torch.vstack(list(Cr1))
    Cr2 = map(lambda b_i: C_right2(inner_batch, r_batch, b_i),
                   range(batch_size))
    Cr2 = torch.vstack(list(Cr2))
    C_right = ((Cr1 + Cr2) * phi_batch).mean(dim=0)
    return C_left, C_right
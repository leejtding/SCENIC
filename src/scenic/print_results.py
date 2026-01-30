"""
print_results.py

Utility for visualizing training progress in the SCENIC framework.
This module plots true and estimated conditional survival functions
for selected subjects at a given training epoch, saving the figure
to disk and displaying it.

The estimated survival curves are computed from the generator output,
and true curves are drawn from analytical model functions defined in
`sim_data.py` (e.g., PH or PO models).
"""

from __future__ import annotations
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from scenic.sim_data import *

def print_results(model, epoch, sim_model = "PH", n_plot = 6,
                  ub_plot = 50, steps_plot = 500, n_gen = 2000):
    """
    Plot and save true vs. estimated conditional survival functions.

    Parameters
    ----------
    model : SCENIC
        Trained SCENIC model containing generator, phi, and covariate data.
    epoch : int
        Current epoch index, used in plot filename.
    sim_model : {"PH", "PO"}
        Underlying simulation model used to compute true survival functions.
    n_plot : int, optional
        Number of subjects to visualize (default is 6).
    ub_plot : float, optional
        Upper bound of the time axis for plotting (default is 50).
    steps_plot : int, optional
        Number of time grid steps for the x-axis (default is 500).
    n_gen : int, optional
        Number of generated samples for estimating the survival curve (default is 2000).

    Notes
    -----
    - The function saves each plot to `../check_fig/plot_epoch_XXXX.png`.
    - Uses analytical functions `surv_ph_cond` and `surv_po_cond`
      from `sim_data.py` for true survival curves.
    - Estimated curves are derived from the generator outputs.
    """
    # Plot true and estimated survival functions
    with torch.no_grad():
        # Create time grids for plotting
        time_grid = torch.linspace(0, ub_plot, steps=steps_plot, dtype=torch.float).view(steps_plot, 1)
        print(f"Epoch: {epoch}, Generator Loss: {model.generator_loss:.4f}")

        if n_plot > 0:
            save_dir = "../check_fig/"
            os.makedirs(save_dir, exist_ok=True)

            n_plot = min(n_plot, model.X.size(0))
            fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(16, 4))
            axs[0].set_title('True conditional survival functions')
            for s in range(n_plot):
                if sim_model == "PH":
                    # For PH: True survival function
                    surv_time = surv_ph_cond(time_grid.view(-1).numpy(),
                                             model.X[s, :].cpu().numpy())
                    axs[0].plot(time_grid.view(-1).numpy(), surv_time, ls="--",
                                label=f'Subject {s}')
                elif sim_model == "PO":
                    # For PO: True survival function
                    surv_time = surv_po_cond(time_grid.view(-1).numpy(),
                                             model.X[s, :].cpu().numpy())
                    axs[0].plot(time_grid.view(-1).numpy(), surv_time, ls="--",
                                label=f'Subject {s}')

            axs[0].legend()

            axs[1].set_title('Estimated conditional survival functions from SCENIC')
            for s in range(n_plot):
                grid_size = time_grid.size(0)
                X_plot = model.X[s, :].unsqueeze(0).repeat(n_gen, 1).to(model.device)
                U_plot = 2 * torch.rand(n_gen, model.p_aux, device=model.device) - 1
                gen_plot = model.generator(U_plot, X_plot).view(-1).detach().cpu()
                time_plot = (gen_plot.repeat(grid_size, 1) > time_grid).sum(dim=1) / n_gen
                axs[1].plot(time_grid.numpy(), time_plot.numpy(), ls="--",
                            label=f'Subject {s}')
            axs[1].legend()

            filename = f"plot_epoch_{epoch:04d}.png"
            fig.savefig(os.path.join(save_dir, filename))
            plt.show()
            plt.clf()
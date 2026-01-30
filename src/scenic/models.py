"""
models.py

Neural components for the SCENIC framework (Semi-supervised Conditional
NEural model for Interval Censoring). This module defines:

- `generator`: maps latent noise and covariates to positive event times.
- `phi`: maps (time, covariates) to a value in (0, 1) used within the
  interval-censoring loss construction.
- `SCENIC`: orchestration module that owns networks, optimizers, and
  training steps used to estimate conditional CDFs under interval censoring.
"""

from __future__ import annotations
import torch
import torch.nn as nn

from scenic.loss import *

class generator(nn.Module):
    """
    Generator network producing (positive) event-time samples from
    latent noise and covariates.

    The network takes a concatenation of latent variables `u` and
    covariates `x` and outputs strictly positive times via an exponential
    activation on the last layer.

    Parameters
    ----------
    input_dim : int
        Dimension of concatenated inputs (p_aux + p_covariates).
    hidden_dim : int
        Hidden layer width.
    output_dim : int
        Output dimension (typically 1 for scalar time).

    Notes
    -----
    The final activation uses `torch.exp` to enforce positivity.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(generator, self).__init__()
        # Define activation functions
        self.relu = nn.ReLU()

        # Define layers
        self.gen1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.gen2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.gen3 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.gen4 = nn.Linear(hidden_dim, output_dim, bias=True)

    def forward(self, u, x):
        """
        Forward pass.

        Parameters
        ----------
        u : torch.Tensor
            Latent inputs of shape (N, p_aux).
        x : torch.Tensor
            Covariates of shape (N, p).

        Returns
        -------
        torch.Tensor
            Positive times of shape (N, output_dim).
        """
        layer1 = self.relu(self.gen1(torch.cat(tensors=(u, x), dim=1)))
        layer2 = self.relu(self.gen2(layer1))
        layer3 = self.relu(self.gen3(layer2))
        output = torch.exp(self.gen4(layer3))
        return output


class phi(nn.Module):
    """
    Phi-network that weighs (time, covariates) pairs in (0, 1).

    The φ network is used as part of the interval-censoring loss. It
    receives concatenated time `t` and covariates `x` and produces a
    sigmoid output.

    Parameters
    ----------
    input_dim : int
        Number of covariates p (time is provided separately).
    hidden_dim : int
        Hidden layer width.
    output_dim : int
        Output dimension (typically 1).

    Notes
    -----
    The final activation is a sigmoid to bound outputs in (0, 1).
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(phi, self).__init__()
        # Define activation functions
        self.relu = nn.ReLU()
        self.sigmoid  = nn.Sigmoid()

        # Define layers
        self.phi1 = nn.Linear(1 + input_dim, hidden_dim, bias=True)
        self.phi2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.phi3 = nn.Linear(hidden_dim, output_dim, bias=True)

    def forward(self, t, x):
        """
        Forward pass.

        Parameters
        ----------
        t : torch.Tensor
            Times of shape (N, 1) or broadcastable to it.
        x : torch.Tensor
            Covariates of shape (N, p).

        Returns
        -------
        torch.Tensor
            Weights in (0, 1) of shape (N, output_dim).
        """
        layer1 = self.relu(self.phi1(torch.cat((t,x),1)))
        layer2 = self.relu(self.phi2(layer1))
        output = self.sigmoid(self.phi3(layer2))
        return output
    

class SCENIC(nn.Module):
    """
    SCENIC model tying together generator and phi networks, optimizers, and
    training steps for interval-censored survival estimation.

    Parameters
    ----------
    l : torch.Tensor
        Left bounds of censoring intervals, shape (n,).
    r : torch.Tensor
        Right bounds of censoring intervals, shape (n,).
    X : torch.Tensor
        Covariates, shape (n, p).
    device : str, default='cpu'
        Device specifier ('cpu' or 'cuda').
    n_aux : int, default=400
        Number of auxiliary latent samples per observation.
    p_aux : int, default=5
        Dimensionality of latent input u.
    batch_size : int, default=5
        Mini-batch size.
    train_size : int, default=100
        Number of samples used to form inner time grids each epoch.
    temp : float, default=0.1
        Temperature for soft indicator approximation in CDF estimation.
    temp_decay : float, default=1.0
        Optional temperature decay (unused in current code).
    gen_hidden : int, default=100
        Hidden width of generator network.
    phi_hidden : int, default=1000
        Hidden width of phi network.
    gen_lr : float, default=1e-4
        Learning rate for generator optimizer (Adam).
    phi_lr : float, default=1e-5
        Learning rate for phi optimizer (SGD with momentum).

    Attributes
    ----------
    generator : generator
        Generator network instance.
    phi : phi
        Phi network instance.
    loss_func : nn.Module
        Loss function used to compare left/right components (MSELoss).
    generator_loss : float
        Latest generator loss value (for logging).
    phi_loss : float
        Latest phi loss value (for logging).
    iterations_epoch : int
        Number of training iterations per epoch (= n // batch_size).
    """

    def __init__(
        self,
        l : torch.Tensor,
        r : torch.Tensor,
        X : torch.Tensor,
        device : str = 'cpu',
        n_aux: int = 500,
        p_aux: int = 5,
        batch_size: int = 5,
        train_size: int = 100,
        temp: float = 0.1,
        temp_decay: float = 1.0,
        gen_hidden: int = 100,
        phi_hidden: int = 1000,
        gen_lr: float = 1e-4,
        phi_lr: float = 1e-5
    ):
        super(SCENIC, self).__init__()

        # Move tensors to the specified device
        self.device = torch.device(device)
        self.l = l.to(self.device).float()
        self.r = r.to(self.device).float()
        self.X = X.to(self.device).float()

        # Initialize training parameters
        self.n = X.shape[0]
        self.p = X.shape[1]
        self.n_aux = n_aux
        self.p_aux = p_aux

        self.train_size = train_size
        self.batch_size = batch_size
        self.iterations_epoch = int(self.n / self.batch_size)
        self.temp = temp
        self.temp_decay = temp_decay

        # Initialize variable selection components
        self.vs_Xidx = torch.ones(self.p) > 0
        self.vs_mask = ~self.vs_Xidx

        self.informative_check = False
        self.epoch_check = False
        self.selected = False

        # Initialize generator
        self.generator_loss = 0.0
        self.gen_lr = gen_lr
        self.generator = generator(self.p_aux + self.p, gen_hidden, 1).to(device)
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(),
                                                    lr = self.gen_lr,
                                                    betas = (0.9, 0.999))

        # Initialize phi function
        self.phi_loss = 0.0
        self.phi_lr = phi_lr
        self.phi = phi(self.p, phi_hidden, 1).to(device)
        # self.phi_optimizer = torch.optim.Adam(self.phi.parameters(),
        #                                       lr = self.phi_lr,
        #                                       betas = (0.9, 0.999))
        self.phi_optimizer = torch.optim.SGD(self.phi.parameters(),
                                             lr = self.phi_lr,
                                             momentum=0.9)
        
        # Set loss function
        self.loss_func = nn.MSELoss()

    def gen_cdf(self, inner, X):
        """
        Approximate conditional CDF at grid `inner` for covariates `X`.

        Uses the generator to sample times given latent draws and computes
        a soft indicator `I(T <= t)` approximation via a sigmoid with
        temperature `self.temp`, then averages over latent samples.

        Parameters
        ----------
        inner : torch.Tensor
            1D tensor of sorted time grid points, shape (m,).
        X : torch.Tensor
            Covariate block, shape (N, p). N should be batch_size * n_aux.

        Returns
        -------
        torch.Tensor
            Estimated CDF values of shape (N, m).
        """
        U_gen = 2 * torch.rand(self.batch_size * self.n_aux, self.p_aux,
                               device=self.device) - 1
        gen = self.generator(U_gen, X).view(self.batch_size, self.n_aux)
        diff = inner[None, None, :] - gen[:, :, None]
        F_gen = torch.sigmoid(diff / self.temp).mean(dim=1)
        return F_gen
    
    def step_phi(self, inner, X_batch, l_batch, r_batch, phi_X, phi_t):
        """
        One update step for the phi network maximizing the loss gap.

        Parameters
        ----------
        inner : torch.Tensor
            1D tensor of inner time grid points (m,).
        X_batch : torch.Tensor
            Covariates for the current batch, shape (batch_size * n_aux, p)
            for generator CDF computation.
        l_batch : torch.Tensor
            Left bounds for the batch, shape (batch_size,).
        r_batch : torch.Tensor
            Right bounds for the batch, shape (batch_size,).
        phi_X : torch.Tensor
            Covariates repeated to align with inner (batch_size * m, p).
        phi_t : torch.Tensor
            Time grid repeated to align with covariates (batch_size * m, 1).

        Returns
        -------
        None
        """
        F_gen = self.gen_cdf(inner, X_batch)
        phi_batch = self.phi(phi_t, phi_X).reshape(self.batch_size,
                                                   inner.shape[0])
        C_left, C_right = C_loss(inner, l_batch, r_batch, F_gen, phi_batch,
                                 self.batch_size)

        # 4. Update phi function weights to maximize loss
        self.phi_loss = -1 * self.loss_func(C_left, C_right)
        self.phi_optimizer.zero_grad()
        self.phi_loss.backward()
        self.phi_optimizer.step()

    def step_generator(self, inner, X_batch, l_batch, r_batch, phi_X, phi_t):
        """
        One update step for the generator minimizing the loss gap.

        Parameters
        ----------
        inner : torch.Tensor
            1D tensor of inner time grid points (m,).
        X_batch : torch.Tensor
            Covariates for the current batch, shape (batch_size * n_aux, p).
        l_batch : torch.Tensor
            Left bounds for the batch, shape (batch_size,).
        r_batch : torch.Tensor
            Right bounds for the batch, shape (batch_size,).
        phi_X : torch.Tensor
            Covariates repeated to align with inner (batch_size * m, p).
        phi_t : torch.Tensor
            Time grid repeated to align with covariates (batch_size * m, 1).

        Returns
        -------
        None
        """
        F_gen = self.gen_cdf(inner, X_batch)
        phi_batch = self.phi(phi_t, phi_X).reshape(self.batch_size,
                                                   inner.shape[0])
        C_left, C_right = C_loss(inner, l_batch, r_batch, F_gen, phi_batch,
                                 self.batch_size)

        # 7. Update generator weights to minimize loss
        self.generator_loss = self.loss_func(C_left, C_right)
        self.generator_optimizer.zero_grad()
        self.generator_loss.backward()
        self.generator_optimizer.step()

    def check_informative_weights(self):
        # Check for informative signals before variable selection
        w1 = torch.matmul(torch.abs(self.generator.gen2.weight.data),
                          torch.abs(self.generator.gen1.weight.data))
        w2 = torch.matmul(torch.abs(self.generator.gen3.weight.data), w1)
        w3 = torch.matmul(torch.abs(self.generator.gen4.weight.data), w2)

        aux_signal_mean = torch.mean(w3.view(-1)[:self.p_aux])
        X_signal_mean = torch.mean(w3.view(-1)[self.p_aux:])
        return (X_signal_mean > aux_signal_mean).item()

    def variable_selection(self):
        # Perform variable selection
        w1 = torch.matmul(torch.abs(self.generator.gen2.weight.data),
                          torch.abs(self.generator.gen1.weight.data))
        w2 = torch.matmul(torch.abs(self.generator.gen3.weight.data), w1)
        w3 = torch.matmul(torch.abs(self.generator.gen4.weight.data), w2)

        aux_threshold = torch.mean(w3.view(-1)[:self.p_aux])
        X_signal = w3.view(-1)[self.p_aux:]
        new_vs_Xidx = (X_signal > aux_threshold).detach().cpu().numpy()
        new_vs_mask = ~new_vs_Xidx

        with torch.no_grad():
            self.generator.gen1.weight[:, self.p_aux:][:, new_vs_mask] = 0
            self.generator_optimizer = torch.optim.Adam(self.generator.parameters(),
                                                        lr = self.gen_lr,
                                                        betas = (0.9, 0.999))

        self.vs_Xidx = new_vs_Xidx
        self.vs_mask = new_vs_mask

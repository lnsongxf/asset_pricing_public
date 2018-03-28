"""
Computes recursive utility, spectral radius for Bansal--Yaron model

    g_c = μ_c + z + σ η                 # consumption growth, μ_c is μ

    z' = ρ z + ϕ_z σ e'                 # z, ϕ_z here is x, ϕ_e in BY

    g_d = μ_d + α z + ϕ_d σ η           # div growth, α here is ϕ in BY

    (σ^2)' = v σ^2 + d + ϕ_σ w'         # v, d, ϕ_σ is v_1, σ^2(1-v_1), σ_w

Innovations are IID and N(0, 1). 

See table IV on page 1489 for parameter values.

"""

import numpy as np
from numpy import exp
from numpy.random import randn
from numba import jit, prange

from interp2d import lininterp_2d


default_params = (0.998,     # β
                 10.0,       # γ
                 1.5,        # ψ
                 0.0015,     # μ_c
                 0.979,      # ρ
                 0.044,      # ϕ_z
                 0.987,      # v
                 7.9092e-7,  # d
                 2.3e-6,     # ϕ_σ
                 0.0015,     # μ_d
                 3.0,        # α
                 4.5)        # ϕ_d


@jit(nopython=True)
def quantile(x, q):
    """
    Return, roughly, the q-th quantile of univariate data set x.

    Not exact, skips linear interpolation.  Works fine for large
    samples.
    """
    k = len(x)
    x.sort()
    return x[int(q * k)]



class BY:

    def __init__(self, params=default_params, 
                       z_grid_size=6, 
                       σ_grid_size=6,
                       mc_draw_size=2500):

        # Unpack all params
        self.β, self.γ, self.ψ, self.μ_c, self.ρ, self.ϕ_z, \
                self.v, self.d, self.ϕ_σ, \
                self.μ_d, self.α, self.ϕ_d = params

        # Build grid and draw shocks for Monte Carlo
        self.z_grid, self.σ_grid, self.shocks = \
                build_grid_and_shocks(self.pack_params(), 
                                      z_grid_size,
                                      σ_grid_size,
                                      mc_draw_size)
        self.w_star = None

    def pack_params(self):
        params = self.β, self.γ, self.ψ, self.μ_c, self.ρ, self.ϕ_z, \
            self.v, self.d, self.ϕ_σ, \
            self.μ_d, self.α, self.ϕ_d 
        return params


    def compute_recursive_utility(self,
                                  w_init=None, 
                                  tol=1e-4, 
                                  max_iter=10000,
                                  verbose=True):
        """
        Solves for the fixed point of T.

        """

        if w_init is None:
            w = np.ones((len(self.z_grid), len(self.σ_grid)))
        else:
            w = w_init

        params = self.pack_params()
        error = tol + 1
        i = 1
        while error > tol and i < max_iter:
            w_next = T(params, w, self.z_grid, self.σ_grid, self.shocks)
            error = np.max(np.abs(w - w_next))
            i += 1
            w = w_next

        if verbose and i < max_iter:
            print(f"Converged in {i} iterations")
        if i == max_iter:
            print(f"Hit iteration upper bound!")

        self.w_star = w


    def compute_spec_rad(self, 
                         z_0=None,
                         σ_0=None,
                         n=750, 
                         num_reps=2000,
                         with_sup=False):

        assert self.w_star is not None, "w_star has not been initialized"
        params = self.pack_params()

        if z_0 is None:
            z_0 = 0.0
        if σ_0 is None:
            σ_0 = np.sqrt(self.d / (1 - self.v))

        if with_sup:
            sup_val = -np.inf
            for z in self.z_grid:
                for σ in self.σ_grid:
                    s = compute_spec_rad_given_utility(params, 
                                                       self.z_grid,
                                                       self.σ_grid,
                                                       self.w_star,
                                                       n, 
                                                       z, 
                                                       σ, 
                                                       num_reps)
                    sup_val = max(sup_val, s)
            rV = sup_val
        else:
            rV = compute_spec_rad_given_utility(params, 
                                                self.z_grid,
                                                self.σ_grid,
                                                self.w_star,
                                                n, 
                                                z_0, 
                                                σ_0, 
                                                num_reps)

        return rV


@jit(nopython=True, parallel=True)
def T(params, w, z_grid, σ_grid, shocks):
    """ 
    Apply the operator 

            Tw(x) = 1 + (Kw^θ(x))^(1/θ)

    induced by the Bansal-Yaron model to a function w.

    Uses Monte Carlo for Integration.

    """
    # Unpack all params
    β, γ, ψ, μ_c, ρ, ϕ_z, v, d, ϕ_σ, μ_d, α, ϕ_d = params
    θ = (1 - γ) / (1 - 1/ψ)

    num_shocks = shocks.shape[1]

    nz = len(z_grid)
    nσ = len(σ_grid)

    g = w**θ
    Kg = np.empty_like(g)

    # Apply the operator K to g, computing Kg
    for i in prange(nz):

        z = z_grid[i]

        for j in range(nσ):
            σ = σ_grid[j]

            mf = np.exp((1 - γ) * (μ_c + z) + (1 - γ)**2 * σ**2 / 2)
            g_expec = 0.0

            for k in range(num_shocks):
                ε1, ε2 = shocks[:, k]
                zp = ρ * z + ϕ_z * σ * ε1
                σp2 = np.maximum(v * σ**2 + d + ϕ_σ * ε2, 0)
                σp = np.sqrt(σp2)
                g_expec += lininterp_2d(z_grid, σ_grid, g, (zp, σp))
            g_expec = g_expec /  num_shocks

            Kg[i, j] = mf * g_expec

    return 1.0 + β * Kg**(1/θ)  # Tw




@jit(nopython=True)
def compute_spec_rad_given_utility(params, 
                                   z_grid,
                                   σ_grid,
                                   w_star,
                                   n, 
                                   z_0, 
                                   σ_0, 
                                   num_reps):
    """
    Uses fact that

        M_j = β**θ * exp( -γ * g_c_j) * (w(x_j) / (w(x_{j-1}) - 1)**(θ - 1)

    where w is the value function.

    This code is written for speed, not clarity, and has some repetition.

    """
    # Unpack all params
    β, γ, ψ, μ_c, ρ, ϕ_z, v, d, ϕ_σ, μ_d, α, ϕ_d = params

    θ = (1 - γ) / (1 - 1/ψ)
    phi_obs = np.empty(num_reps)

    for m in range(num_reps):

        # Set seed
        np.random.seed(m)

        # Reset accumulator and state to initial conditions
        phi_prod = 1.0
        z, σ = z_0, σ_0

        for t in range(n):
            # Calculate W_t
            W = lininterp_2d(z_grid, σ_grid, w_star, (z, σ))
            # Calculate g^c_{t+1}
            g_c = μ_c + z + σ * randn()
            # Calculate g^d_{t+1}
            g_d = μ_d + α * z + ϕ_d * σ * randn()
            # Update state to t+1
            z = ρ * z + ϕ_z * σ * randn()
            σ2 = v * σ**2 + d + ϕ_σ * randn()
            σ = np.sqrt(max(σ2, 0))
            # Calculate W_{t+1}
            W_next = lininterp_2d(z_grid, σ_grid, w_star, (z, σ))
            # Calculate M_{t+1} without β**θ 
            M =  exp(-γ * g_c) * (W_next / (W - 1))**(θ - 1)
            phi_prod = phi_prod * M * exp(g_d)

        phi_obs[m] = phi_prod

    return β**θ * np.mean(phi_obs)**(1/n)




@jit(nopython=True)
def build_grid_and_shocks(params,
                          z_grid_size,
                          σ_grid_size,
                          mc_draw_size,
                          ts_length=100_000,  # sim used when building grid
                          seed=1234):

    # Unpack all params
    β, γ, ψ, μ_c, ρ, ϕ_z, v, d, ϕ_σ, μ_d, α, ϕ_d = params

    # Set seed and time series length for sampling and grid construction
    np.random.seed(seed)
 
    # Allocate memory and intitialize
    z_vec = np.empty(ts_length)
    σ_vec = np.empty(ts_length)
    z_vec[0] = 0.0
    σ_vec[0] = np.sqrt(d / (1 - v))

    # Generate state
    for t in range(ts_length-1):
        # Update state
        z_vec[t+1] = ρ * z_vec[t] + ϕ_z * σ_vec[t] * randn()
        σ2 = v * σ_vec[t]**2 + d + ϕ_σ * randn()
        σ_vec[t+1] = np.sqrt(max(σ2, 0))

    q1, q2 = 0.05, 0.95  # quantiles
    z_min, z_max = quantile(z_vec, q1), quantile(z_vec, q2)
    σ_min, σ_max = quantile(σ_vec, q1), quantile(σ_vec, q2)
    # Build grid along each state axis
    z_grid = np.linspace(z_min, z_max, z_grid_size)
    σ_grid = np.linspace(σ_min, σ_max, σ_grid_size)

    shocks = randn(2, mc_draw_size)

    return z_grid, σ_grid, shocks

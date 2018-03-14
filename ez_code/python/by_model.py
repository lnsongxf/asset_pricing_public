# -*- coding: utf-8 -*- 
"""
Computes recursive utility, Bansal--Yaron model

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




class BY:
    """
    Class for the Bansal Yaron wealth / consumption process, with 

        g_c = μ_c + z + σ η                 # consumption growth, μ_c is μ

        z' = ρ z + ϕ_z σ e'                 # z, ϕ_z here is x, ϕ_e in BY

        g_d = μ_d + α z + ϕ_d σ η           # div growth, α here is ϕ in BY

        (σ^2)' = v σ^2 + d + ϕ_σ w'         # v, d, ϕ_σ is v_1, σ^2(1-v_1), σ_w

    Innovations are IID and N(0, 1). 

    See table IV on page 1489 for parameter values.

    """
    
    def __init__(self, 
                 params=default_params,
                 z_grid_size=6,
                 σ_grid_size=6,
                 read_from_file=False,
                 mc_draw_size=8000,
                 w_star=None):

        # Parameters
        self.β, self.γ, self.ψ, \
            self.μ_c, self.ρ, self.ϕ_z, self.v, self.d, self.ϕ_σ, \
                self.μ_d, self.α, self.ϕ_d = params

        # Derived parameters
        self.θ = (1 - self.γ) / (1 - 1/self.ψ)

        # Grid
        self.z_grid_size, self.σ_grid_size = z_grid_size, σ_grid_size 

        # Draw some shocks for Monte Carlo integration, each row one shock seq
        self.shocks = randn(2, mc_draw_size)

        # Calculate grid and w values on grid, or read them from file
        if read_from_file:
            self.z_grid = np.load('z_grid.npy')
            self.σ_grid = np.load('σ_grid.npy')
            self.w_star = np.load('w_star.npy')
        else:
            self.compute_grid_and_solution()


    def compute_grid_and_solution(self):
            # Simulate the state and pick upper and lower bounds for the grid
            z, σ = self.simulate_state()
            q1, q2 = 5, 95  # percentiles
            z_min, z_max = np.percentile(z, q1), np.percentile(z, q2)
            σ_min, σ_max = np.percentile(σ, q1), np.percentile(σ, q2)
            # Build grid along each state axis
            self.z_grid = np.linspace(z_min, z_max, self.z_grid_size)
            self.σ_grid = np.linspace(σ_min, σ_max, self.σ_grid_size)
            # Solve for w_star
            self.w_star = self.compute_recursive_utility()
            # Save the results
            self.write_data()

    def write_data(self):
        np.save('z_grid', self.z_grid)
        np.save('σ_grid', self.σ_grid)
        np.save('w_star', self.w_star)

    def is_dividend_parameter(self, string):
        if string in ('μ_d', 'α', 'ϕ_d'):
            return True
        else:
            return False

    def repack_params(self):
        """
        Repackage the parameter set --- a convenience function.

        """
        params = self.β, self.γ, self.ψ, \
                 self.μ_c, self.ρ, self.ϕ_z, self.v, self.d, self.ϕ_σ, \
                    self.μ_d, self.α, self.ϕ_d 

        return params

    def simulate_state(self, ts_length=1000, z_0=None, σ_0=None, seed=1234):
        """

        Simulate the state process x = (z, σ)

        Returns 
        
            * x_0, ..., x_{T-1} 

        where T = ts_length

        """
        σ_0 = np.sqrt(self.d / (1 - self.v)) if σ_0 is None else σ_0
        z_0 = 0 if z_0 is None else z_0

        params = self.repack_params()
        return _simulate_state(params, ts_length, z_0, σ_0, seed)


    def compute_recursive_utility(self, w=None, verbose=False, tol=1e-6, max_iter=50000):
        """
        Solves for the fixed point of T and writes it to self.w_star

        """
        params = self.repack_params()

        # If no initial condition, start w_star = 1
        if w is None:
            w = np.ones((len(self.z_grid), len(self.σ_grid)))

        error = tol + 1
        i = 1
        while error > tol and i < max_iter:
            w_next = T(params, w, self.z_grid, self.σ_grid, self.shocks)
            error = np.max(np.abs(w - w_next))
            i += 1
            w = w_next

        if i == max_iter:
            print("Hit iteration upper bound when computing fixed point of T!")

        if verbose:
            msg = f"""

            Recursive utility calculation converged in {i} iterations 
            using tolerance {tol}.
            """
            print(msg)

        return w



    def compute_spec_rad(self, z_0=None, σ_0=None, n=750, num_reps=10000):

        if self.w_star is None:
            self.compute_grid_and_solution()

        σ_0 = np.sqrt(self.d / (1 - self.v)) if σ_0 is None else σ_0
        z_0 = 0 if z_0 is None else z_0

        params = self.repack_params()

        return _compute_spec_rad(params, 
                                 self.w_star,
                                 self.z_grid,
                                 self.σ_grid,
                                 n,               # time series length
                                 z_0, 
                                 σ_0, 
                                 num_reps)

    def compute_spec_rad_with_sup(self, n=750, num_reps=10000):

        sup_val = -np.inf

        for z in self.z_grid:
            for σ in self.σ_grid:
                s = self.compute_spec_rad(z_0=z, σ_0=σ, n=n, num_reps=num_reps)
                sup_val = max(sup_val, s)

        return sup_val



    def test_ez_solution(self):
        """
        A function for testing purposes.  The solution w = w_star should satisfy

            w(x) = 1 + (Kw^θ(x))^(1/θ)

        """
        f = self.w_star - self.T(self.w_star)
        return np.max(f)


    def test_euler_equation(self, 
                        T=100000,  
                        z_0=None, 
                        σ_0=None, 
                        seed=1234,
                        test=False):
        """
        We should have
       
           β**θ E exp( (1-γ) * g_c[t+1]) (w(X[t+1]) / (w(X[t] - 1)**θ = 1
         
        Test this, approximately, using ergodicity

        """

        # Unpack
        μ_c, μ_d, α, ϕ_d = self.μ_c, self.μ_d, self.α, self.ϕ_d 
        β, θ = self.β, self.θ 
        γ = self.γ

        # Generate X[0], ..., X[T]
        z_vec, σ_vec = self.simulate_state(z_0=z_0, 
                                           σ_0=σ_0,
                                           ts_length=T+1,
                                           seed=seed)

        # Generate g_c[0], ..., g_c[T] and same for g_d, ignore 0-th elements 
        g_c, g_d = np.empty(T+1), np.empty(T+1)
        g_c[1:] = μ_c + z_vec[:-1] + σ_vec[:-1] * randn(T)
        g_d[1:] = μ_d + α * z_vec[:-1] + ϕ_d * σ_vec[:-1] * randn(T)


        w_star_func = interpolate((self.z_grid, self.σ_grid), 
                                        self.w_star, 
                                        bounds_error=False,
                                        fill_value=None)
        W = w_star_func((z_vec, σ_vec))         # W[0], ..., W[T]

        a = β**θ 
        b = np.exp((1 - γ) * g_c[1:])
        c = (W[1:] / (W[:-1] - 1))**θ
        print("Accuracy stat = ", a * np.mean(b * c))




## == Now the jitted functions == ##

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

        for j in prange(nσ):
            σ = σ_grid[j]

            mf = np.exp((1 - γ) * (μ_c + z) + (1 - γ)**2 * σ**2 / 2)
            g_expec = 0.0

            for k in prange(num_shocks):
                ε1, ε2 = shocks[:, k]
                zp = ρ * z + ϕ_z * σ * ε1
                σp2 = np.maximum(v * σ**2 + d + ϕ_σ * ε2, 0)
                σp = np.sqrt(σp2)
                g_expec += lininterp_2d(z_grid, σ_grid, g, (zp, σp))
            g_expec = g_expec /  num_shocks

            Kg[i, j] = mf * g_expec

    return 1.0 + β * Kg**(1/θ)  # Tw



@jit(nopython=True)
def _simulate_state(params, ts_length, z_0, σ_0, seed):
    """
    As it sounds.

    """
    # Unpack all params
    β, γ, ψ, μ_c, ρ, ϕ_z, v, d, ϕ_σ, μ_d, α, ϕ_d = params

    # Set seed
    np.random.seed(seed)

    # Allocate memory
    z_vec = np.empty(ts_length)
    σ_vec = np.empty(ts_length)

    
    z_vec[0] = z_0
    σ_vec[0] = σ_0

    for t in range(ts_length-1):
        # Update state
        z_vec[t+1] = ρ * z_vec[t] + ϕ_z * σ_vec[t] * randn()
        σ2 = v * σ_vec[t]**2 + d + ϕ_σ * randn()
        σ_vec[t+1] = np.sqrt(max(σ2, 0))

    return z_vec, σ_vec

@jit(nopython=True)
def _compute_spec_rad(params, 
                           w_star,
                           z_grid,
                           σ_grid,
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




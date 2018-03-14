"""

Schorfheide--Song--Yaron code

"""


import numpy as np
from numpy.random import randn


class SSY:
    """
    Class for the SSY model, with state process

            g_c = μ_c + z + σ_c η'

            g_d = μ_d + α z + δ σ_c η' + σ_d υ'

            z' = ρ z + sqrt(1 - ρ^2) σ_z e'

            h_z' = ρ_hz h_z + σ_hz u'

            h_c' = ρ_hc h_c + σ_hc w'

            h_d' = ρ_hd h_d + σ_hd w'

            σ_z = ϕ_z σ_bar exp(h_z)

            σ_c = ϕ_c σ_bar exp(h_c)

            σ_d = ϕ_d σ_bar exp(h_d)


        Innovations are IID and N(0, 1).  

        Default values from May 2017 version of Schorfheide, Song and Yaron.
        See p. 28.

    """
        
    def __init__(self, 
                 β=0.999, 
                 γ=8.89, 
                 ψ=1.97,
                 μ_c=0.0016,
                 ρ=0.987,
                 ϕ_z=0.215,
                 σ_bar=0.0032,
                 ϕ_c=1.0,
                 ρ_hz=0.992,
                 σ_hz=np.sqrt(0.0039),
                 ρ_hc=0.991,
                 σ_hc=np.sqrt(0.0096),
                 μ_d=0.001,
                 α=3.65,                     # = ϕ in SSY's notation
                 δ=1.47,                     # = π in SSY's notation
                 ϕ_d=4.54,
                 ρ_hd=0.969,
                 σ_hd=np.sqrt(0.0447)):


        # Preferences
        self.β, self.γ, self.ψ = β, γ, ψ
        # Consumption
        self.μ_c, self.ρ, self.ϕ_z = μ_c, ρ, ϕ_z
        self.σ_bar, self.ϕ_c, self.ρ_hz = σ_bar, ϕ_c, ρ_hz
        self.σ_hz, self.ρ_hc, self.σ_hc  = σ_hz, ρ_hc, σ_hc  
        # Dividends
        self.μ_d, self.α, self.δ = μ_d, α, δ 
        self.ϕ_d, self.ρ_hd, self.σ_hd = ϕ_d, ρ_hd, σ_hd


    def simulate_state(self, ts_length=1000000, seed=1234):
        """
        Returns state process.

        """
        np.random.seed(seed)

        # Unpack parameters for state process
        ρ, ϕ_z, σ_bar = self.ρ, self.ϕ_z, self.σ_bar
        ρ_hz, σ_hz, ρ_hc, σ_hc = self.ρ_hz, self.σ_hz, self.ρ_hc, self.σ_hc
        ϕ_d, ρ_hd, σ_hd = self.ϕ_d, self.ρ_hd, self.σ_hd

        # Allocate memory for states with initial conditions at the stationary
        # mean, which is zero
        z_vec = np.zeros(ts_length)
        h_z_vec = np.zeros(ts_length)
        h_c_vec = np.zeros(ts_length)
        h_d_vec = np.zeros(ts_length)

        # Simulate 
        for t in range(ts_length-1):

            # Simplify names
            h_z, h_c, h_d = h_z_vec[t], h_c_vec[t], h_d_vec[t]
            # Map h to σ
            σ_z = ϕ_z * σ_bar * np.exp(h_z)
            # Update states
            z_vec[t+1] = ρ * z_vec[t] + np.sqrt(1 - ρ**2) * σ_z * randn()
            h_z_vec[t+1] = ρ_hz * h_z + σ_hz * randn()
            h_c_vec[t+1] = ρ_hc * h_c + σ_hc * randn()
            h_d_vec[t+1] = ρ_hd * h_d + σ_hd * randn()

        return z_vec, h_z_vec, h_c_vec, h_d_vec


    def simulate_consumption_given_state(self, z_vec, h_z_vec, h_c_vec, h_d_vec, seed=None):

        if seed is not None:
            np.random.seed(seed)

        # Unpack
        μ_c, σ_bar, ϕ_c = self.μ_c, self.σ_bar, self.ϕ_c
        μ_d, α, δ = self.μ_d, self.α, self.δ
        ϕ_d = self.ϕ_d

        # Map h to σ
        σ_c = ϕ_c * σ_bar * np.exp(h_c_vec)
        
        # Evaluate consumption 
        c_vec = μ_c + z_vec + σ_c * randn(len(z_vec))

        return c_vec

    def simulate_dividends_given_state(self, z_vec, h_z_vec, h_c_vec, h_d_vec, seed=None):

        if seed is not None:
            np.random.seed(seed)

        # Unpack
        σ_bar, ϕ_c = self.σ_bar, self.ϕ_c
        μ_d, α, δ = self.μ_d, self.α, self.δ
        ϕ_d = self.ϕ_d

        # Map h to σ
        σ_c = ϕ_c * σ_bar * np.exp(h_c_vec)
        σ_d = ϕ_d * σ_bar * np.exp(h_d_vec)
        
        # Evaluate dividends
        n = len(z_vec)
        w_1, w_2 = randn(n), randn(n)
        d_vec = μ_d + α * z_vec + δ * σ_c * w_1 + σ_d * w_2

        return d_vec



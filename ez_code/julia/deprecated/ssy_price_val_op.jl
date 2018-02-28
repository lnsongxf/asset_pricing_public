#=

Apply the valuation operator Phi to h, producing Phi h.

=# 

include("recursive_utility_solvers/ssy_compute_recursive_util.jl")


"""
Apply the operator 

    Phi h(x) = int h(x') m(x, x') Pi(x, dx')

or, more explicitly,

    Phi h(x) = β^θ k s(x) d(x) int h(x') n(x') Pi(x, dx')

where

    k = exp(μ_d -γμ_c)

    x = (z, h_z, h_c, h_d)

    n(x') = w(x')^(θ - 1)

    s(x) = exp((α - γ) z + ((δ-γ)^2 σ_c^2 + σ_d^2) / 2)

    d(x) = (w(x) - 1)^(1 - θ)

    σ_c = ϕ_c σ_bar exp(h_c)

    σ_d = ϕ_d σ_bar exp(h_d)

and (z', h_z', h_c', h_d') update via the dynamics
for the SSY model given previously.  

Apart from updating h to Phi h, we also compute the L1 norm

    || Phi h || := int (Phi h)(x) pi(x) dx

using Monte Carlo

Input data is h.  Returns Phi h

"""
function Phi(h, scm::SSYComputableModel)

    # Unpack structs
    ez, sc, sd = scm.ez, scm.sc, scm.sd
    # Unpack preference parameters
    β, γ, θ = ez.β, ez.γ, ez.θ
    # Unpack consumption parameters
    μ_c, ρ, ϕ_z, σ_bar, ϕ_c = sc.μ_c, sc.ρ, sc.ϕ_z, sc.σ_bar, sc.ϕ_c
    ρ_hz, σ_hz, ρ_hc, σ_hc = sc.ρ_hz, sc.σ_hz, sc.ρ_hc, sc.σ_hc
    # Unpack dividend parameters
    μ_d, α, δ, ϕ_d, ρ_hd, σ_hd = sd.μ_d, sd.α, sd.δ, sd.ϕ_d, sd.ρ_hd, sd.σ_hd

    # Unpack grids
    p_vec, s_vec = scm.p_vec, scm.s_vec
    z_grid, h_z_grid = scm.z_grid, scm.h_z_grid
    h_c_grid, h_d_grid = scm.h_c_grid, scm.h_d_grid

    # Some useful constants
    λ = sqrt(1 - ρ^2)
    τ_z = ϕ_z * σ_bar
    τ_c = ϕ_c * σ_bar
    τ_d = ϕ_d * σ_bar
    ex_const = exp(μ_d - γ * μ_c)

    # Interpolate h 
    h_func = interpolate((z_grid, h_z_grid, h_c_grid, h_d_grid), 
                         h, 
                         Gridded(Linear()))

    # Allocate memory for return values
    h_next = similar(h)

    # Interpolate w_star 
    w = interpolate((z_grid, h_z_grid, h_c_grid), 
                         scm.w_star, 
                         Gridded(Linear()))

    # Apply the operator Phi to h
    for (i_z, z) in enumerate(z_grid)
        for (i_hc, h_c) in enumerate(h_c_grid)
            σ_c = τ_c * exp(h_c)
            for (i_hd, h_d) in enumerate(h_d_grid)
                σ_d = τ_d * exp(h_d)
                # Evaluate s(x)
                sx = exp(z * (α - γ) + 0.5 * ((δ * - γ)^2 * σ_c^2 + σ_d^2))
                for (i_hz, h_z) in enumerate(h_z_grid)
                    # Evaluate d(x)
                    dx = (w[z, h_z, h_c] - 1)^(1 - θ)
                    # Now calculate h(x') * n(x')
                    hxnx = 0.0
                    for (p, η) in zip(p_vec, s_vec)
                        # Update z
                        σ_z = τ_z * exp(h_z)
                        zp = ρ * z + λ * σ_z * η
                        for (q, ω) in zip(p_vec, s_vec)
                            # Update h_z
                            h_zp = ρ_hz * h_z + σ_hz * ω
                            for (r, ϵ) in zip(p_vec, s_vec)
                                # Update h_c
                                h_cp = ρ_hc * h_c + σ_hc * ϵ
                                nx = w[zp, h_zp, h_cp]^(θ - 1) 
                                for (s, ξ) in zip(p_vec, s_vec)
                                    # Update h_d
                                    h_dp = ρ_hd * h_d + σ_hd * ξ
                                    # Compute product h(x') n(x')
                                    hxnx += h_func[zp, h_zp, h_cp, h_dp] * nx * p * q * r * s
                                end
                            end
                        end
                    end
                    result = β^ez.θ * ex_const * dx * sx * hxnx
                    h_next[i_z, i_hz, i_hc, i_hd] = result
                end
            end
        end
    end

    # Integrate Phi h = scm.h_next with respect to the stationary distribution
    # to compute it's L1 norm
    Phih = interpolate((z_grid, h_z_grid, h_c_grid, h_d_grid), 
                         h_next, 
                         Gridded(Linear()))

    n = size(scm.stationary_draws)[2]  
    l1norm = 0.0
    for i in 1:n
        z, hz, hc, hd = scm.stationary_draws[:, i]
        l1norm += Phih[z, hz, hc, hd]
    end
    return l1norm / n, h_next
end




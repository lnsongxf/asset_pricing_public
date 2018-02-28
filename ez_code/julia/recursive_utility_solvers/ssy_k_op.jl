
include("ssy_model.jl")

using Interpolations

"""
Apply the operator 

    Kg(z, h_z, h_c) 
        =  exp((1-γ)(μ_c + z) + (1-γ)^2 σ_c^2 / 2) E g(z', h_z', h_c')

where σ_c = ϕ_c σ_bar exp(h_c), and (z', h_z', h_c') update via the dynamics
for the SSY model.  When we write x as the state, the meaning is

    x = (z, h_z, h_c)

induced by the SSY model to a function g.  Compute

    || Kg || := sup_x | Kg(x) |

Uses Monte Carlo for numerical integration.  

Input data is g.  

"""
function K_interp(g, cm::SSYComputableModel)

    # Unpack parameters
    ez, cp = cm.ez, cm.cp
    β, γ = ez.β, ez.γ
    μ_c, ρ, ϕ_z, σ_bar, ϕ_c = cp.μ_c, cp.ρ, cp.ϕ_z, cp.σ_bar, cp.ϕ_c
    ρ_hz, σ_hz, ρ_hc, σ_hc = cp.ρ_hz, cp.σ_hz, cp.ρ_hc, cp.σ_hc

    # Some useful constants
    δ = sqrt(1 - ρ^2)
    τ_z = ϕ_z * σ_bar
    τ_c = ϕ_c * σ_bar

    # Storage
    Kg = similar(g)

    # Interpolate g 
    g_func = interpolate((cm.z_grid, cm.h_z_grid, cm.h_c_grid), 
                         g, 
                         Gridded(Linear()))

    n = length(cm.shocks)

    # Apply the operator K to g, computing Kg and || Kg ||
    for (i, z) in enumerate(cm.z_grid)
        for (k, h_c) in enumerate(cm.h_c_grid)
            σ_c = τ_c * exp(h_c)
            mf = exp((1 - γ) * (μ_c + z) + (1 - γ)^2 * σ_c^2 / 2)
            for (j, h_z) in enumerate(cm.h_z_grid)
                σ_z = τ_z * exp(h_z)
                g_exp = 0.0
                for (η, ω, ϵ) in cm.shocks
                    zp = ρ * z + δ * σ_z * η
                    h_cp = ρ_hc * h_c + σ_hc * ω
                    h_zp = ρ_hz * h_z + σ_hz * ϵ
                    g_exp += g_func[zp, h_zp, h_cp]
                end
                Kg[i, j, k] = β^ez.θ * mf * (g_exp / n)
            end
        end
    end

    return Kg
end



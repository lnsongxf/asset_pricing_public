include("by_model.jl")

using Interpolations

""" 
Apply the operator K induced by the Bansal-Yaron model to a function g.

Uses Monte Carlo for Integratoin.

"""
function K_interp(g, cm::BYComputableModel)      

    # Unpack parameters
    ez, cp = cm.ez, cm.cp
    β, γ, ψ = ez.β, ez.γ, ez.ψ
    ρ, v, d, ϕ_z, ϕ_σ = cp.ρ, cp.v, cp.d, cp.ϕ_z, cp.ϕ_σ
    μ_c = cp.μ_c
    θ = ez.θ
    z_grid, σ_grid = cm.z_grid, cm.σ_grid

    # Storage
    Kg = similar(g)
    n = length(cm.shocks)

    # Interpolate g and allocate memory for new g
    g_func = interpolate((z_grid, σ_grid), g, Gridded(Linear()))

    # Apply the operator K to g, computing Kg and || Kg ||
    for (i, z) in enumerate(z_grid)
        for (j, σ) in enumerate(σ_grid)
            mf = exp((1 - γ) * (μ_c + z) + (1 - γ)^2 * σ^2 / 2)
            g_exp = 0.0
            for (η,  ω) in cm.shocks
                zp = ρ * z + ϕ_z * σ * η
                σp2 = v * σ^2 + d + ϕ_σ * ω
                σp = σp2 < 0 ? 1e-8 : sqrt(σp2)
                g_exp += g_func[zp, σp]
            end
             
            Kg[i, j] = β^θ * mf * (g_exp / n)
        end
    end

    return Kg
end



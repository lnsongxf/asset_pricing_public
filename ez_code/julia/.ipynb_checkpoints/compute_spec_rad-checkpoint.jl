#=

Simulate the process

    Phi_{t+1} = M_{t+1} (D_{t+1} / D_t)

having solved the recursive utility problem for the return on wealth function
and hence the SDF.

Use the fact that

    M_{t+1} = β^θ (C_{t+1} / C_t)^{-γ} (W_{t+1} / (W_t - 1))^{θ - 1}
=#


include("recursive_utility_solvers/compute_recursive_util.jl")


"""
Simulates the path for W_t = w(X_t), for the SSY model

Returns that path plus 

"""
function sim_ez_paths(cm::SSYComputableModel; n=1000, use_seed=false) # Length of simulation

    # Interpolate w_star 
    w = interpolate((cm.z_grid, cm.h_z_grid, cm.h_c_grid), 
                     cm.w_star, 
                     Gridded(Linear()))

    # Simulate state process and consumption, dividends for n+1 periods
    c_growth, d_growth, z_vals, h_z_vals, h_c_vals, h_d_vals =
        simulate(cm.cp, cm.dp, n, use_seed)

    W_path = Array{Float64}(n)

    for t in 1:n
        W_path[t] = w[z_vals[t], h_z_vals[t], h_c_vals[t]]
    end

    return c_growth, d_growth, W_path
end

"""
Same as above but for the BY model.  Consider merging these functions because
they share a lot of logic.

"""

function sim_ez_paths(cm::BYComputableModel; n=1000, use_seed=false) # Length of simulation

    # Interpolate w_star 
    w = interpolate((cm.z_grid, cm.σ_grid), 
                     cm.w_star, 
                     Gridded(Linear()))

    # Simulate state process and consumption, dividends for n+1 periods
    c_growth, d_growth, z_vals, σ_vals = simulate(cm.cp, cm.dp, n, use_seed)

    W_path = Array{Float64}(n)

    for t in 1:n
        W_path[t] = w[z_vals[t], σ_vals[t]]
    end

    return c_growth, d_growth, W_path
end




function test_ez_euler(cm::ComputableModel; n=10000)

    γ, β, θ = cm.ez.γ, cm.ez.β, cm.ez.θ

    # Simulate state process and consumption, dividends 
    c_growth, d_growth, W_path = sim_ez_paths(cm, n=n+1, use_seed=false)

    W_ratio = W_path[2:end] ./ (W_path[1:end-1] - 1)

    C_term = Array{Float64}(n)
    W_term = Array{Float64}(n)

    for t in 1:n
        C_term[t] = exp((1-γ) * c_growth[t]) 
        W_term[t] = W_ratio[t]^θ
    end

    return β^θ * mean(C_term .* W_term)

end




function sim_sdf(cm::ComputableModel; n=1000) # Length of simulation

    γ, β, θ = cm.ez.γ, cm.ez.β, cm.ez.θ

    # Simulate state process and consumption, dividends 
    c_growth, d_growth, W_path = sim_ez_paths(cm, n=n+1, use_seed=false)

    W_ratio = W_path[2:end] ./ (W_path[1:end-1] - 1)

    Phi = Array{Float64}(n)
    M = Array{Float64}(n)

    for t in 1:n
        M[t] = β^θ * exp(-γ * c_growth[t]) * W_ratio[t]^(θ - 1)
        Phi = M[t] * exp(d_growth[t])
    end

    return M, Phi, c_growth, d_growth
end




function compute_spec_rad_by_sim(cm; n=1000, m=1000)

    out = Array{Float64}(m)
    for i in 1:m
        M, Phi, c_growth, d_growth = sim_sdf(cm, n=n)
        p = prod(Phi)
        out[i] = p
    end
    
    return (mean(out))^(1/n)

end



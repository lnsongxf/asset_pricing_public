#=

The Schorfheide, Song and Yaron model consumption process. 
    
Log consumption growth g_c is given by

    g_c = μ_c + z + σ_c η'

    z' = ρ z + sqrt(1 - ρ^2) σ_z e'

    σ_z = ϕ_z σ_bar exp(h_z)

    σ_c = ϕ_c σ_bar exp(h_c)

    h_z' = ρ_hz h_z + σ_hz u'

    h_c' = ρ_hc h_c + σ_hc w'

Here {e}, {u} and {w} are IID and N(0, 1).  


=#

using StatsBase

include("ez_parameters.jl")
include("consumption_models.jl")


"""
Consumption process parameters of SSY model


"""
struct SSYConsumption{T <: Real}  
    μ_c::T 
    ρ::T 
    ϕ_z::T 
    σ_bar::T
    ϕ_c::T
    ρ_hz::T
    σ_hz::T
    ρ_hc::T 
    σ_hc::T 
end


"""
Default consumption process values from May 2017 version of Schorfheide, Song
and Yaron.  See p. 28.

"""
function SSYConsumption(;μ_c=0.0016,
                         ρ=0.987,
                         ϕ_z=0.215,
                         σ_bar=0.0032,
                         ϕ_c=1.0,
                         ρ_hz=0.992,
                         σ_hz=sqrt(0.0039),
                         ρ_hc=0.991,
                         σ_hc=sqrt(0.0096))  
                       
    return SSYConsumption(μ_c,
                          ρ,
                          ϕ_z,
                          σ_bar,
                          ϕ_c,
                          ρ_hz,
                          σ_hz,
                          ρ_hc,
                          σ_hc)
end



"""
Divdend process parameters of SSY model

Log dividend growth g_d is given by

    g_d = μ_d + α z + δ σ_c η' + σ_d υ'

    σ_d = ϕ_d σ_bar exp(h_d)

    h_d' = ρ_hd h_d + σ_hd w'

Shocks are IID and standard normal.

"""
struct SSYDividends{T <: Real}  
    μ_d::T 
    α::T 
    δ::T 
    ϕ_d::T
    ρ_hd::T
    σ_hd::T
end


"""
Default dividend process values from Schorfheide, Song
and Yaron.  See p. 29.  Note that ϕ is α here and π is δ.

"""
function SSYDividends(; μ_d=0.0010,
                        α=3.65,     # = ϕ in SSY's notation
                        δ=1.47,     # = π in SSY's notation
                        ϕ_d=4.54,
                        ρ_hd=0.969,
                        σ_hd=sqrt(0.0447))
                       
    return SSYDividends(μ_d, α, δ, ϕ_d, ρ_hd, σ_hd)
end



"""
Simulate the state process and consumption, dividends for the SSY model.  

Returns

    * X[1], ..., X[ts_length]
    * gc[2], ..., gc[ts_length]
    * gd[2], ..., gd[ts_length]

where

    gc[t] = ln(C[t]) - ln(C[t-1])
    gd[t] = ln(D[t]) - ln(D[t-1])


"""
function simulate(cp::SSYConsumption, dp::SSYDividends, ts_length=1000000, use_seed=true)

    # Set the seed to minimize variation
    if use_seed
        srand(1234)
    end

    # Unpack
    μ_c, ρ, ϕ_z, σ_bar, ϕ_c = cp.μ_c, cp.ρ, cp.ϕ_z, cp.σ_bar, cp.ϕ_c
    ρ_hz, σ_hz, ρ_hc, σ_hc = cp.ρ_hz, cp.σ_hz, cp.ρ_hc, cp.σ_hc
    μ_d, α, δ, ϕ_d, ρ_hd, σ_hd = dp.μ_d, dp.α, dp.δ, dp.ϕ_d, dp.ρ_hd, dp.σ_hd

    # Map h to σ
    tz(h_z) = ϕ_z * σ_bar * exp(h_z)
    tc(h_c) = ϕ_c * σ_bar * exp(h_c)
    td(h_d) = ϕ_d * σ_bar * exp(h_d)

    # Allocate memory for states with initial conditions at the stationary
    # mean, which is zero
    z_vals = zeros(ts_length)
    h_z_vals = zeros(ts_length)
    h_c_vals = zeros(ts_length)
    h_d_vals = zeros(ts_length)

    # Allocate memory consumption and dividends
    c_vals = zeros(ts_length)
    d_vals = zeros(ts_length)

    # Simulate all stochastic processes 
    for t in 1:(ts_length-1)
        # Simplify names
        h_z, h_c, h_d = h_z_vals[t], h_c_vals[t], h_d_vals[t]
        σ_z, σ_c, σ_d = tz(h_z), tc(h_c), td(h_d)
        
        # Evaluate consumption and dividends
        c_vals[t+1] = μ_c + z_vals[t] + σ_c * randn()
        d_vals[t+1] = μ_d + α * z_vals[t] + δ * σ_c * randn() + σ_d * randn()

        # Update states
        z_vals[t+1] = ρ * z_vals[t] + sqrt(1 - ρ^2) * σ_z * randn()
        h_z_vals[t+1] = ρ_hz * h_z + σ_hz * randn()
        h_c_vals[t+1] = ρ_hc * h_c + σ_hc * randn()
        h_d_vals[t+1] = ρ_hd * h_d + σ_hd * randn()
    end

    return c_vals[2:end], d_vals[2:end], z_vals, h_z_vals, h_c_vals, h_d_vals
end



"""
The model constructed for computation, including parameters, derived
parameters, grids, etc.

The order for arguments of g is g(z, h_z, h_c)

"""
struct SSYComputableModel <: ComputableModel

    ez::EpsteinZin{Float64}
    cp::SSYConsumption{Float64}   # Consumption process
    dp::SSYDividends{Float64}     # Divdend process

    z_grid::Vector{Float64}
    h_z_grid::Vector{Float64}
    h_c_grid::Vector{Float64}
    h_d_grid::Vector{Float64}

    w_star::Array{Float64}      # Stores fixed point of T once computed

    shocks::Array{Array{Float64, 1}, 1}    # Array of Gaussian shocks for MC
end



"""
And its constructor.

"""
function SSYComputableModel(ez::EpsteinZin, 
                            cp::SSYConsumption,
                            dp::SSYDividends;
                            n_shocks::Int64=1000,      # for MC integrals
                            q=0.01,                    # quantitle for states
                            gs_z=8,                    # z grid size
                            gs_h_z=4,                  # h_z grid size
                            gs_h_c=4,                  # h_c grid size
                            gs_h_d=4,                  # h_d grid size
                            w_star=Array{Float64}(gs_z, gs_h_z, gs_h_c)) # guess of fp

    # Simulate the state process
    c_vals, d_vals, z_vals, h_z_vals, h_c_vals, h_d_vals =
        simulate(cp, dp)

    # Obtain upper bounds for states
    z_max = quantile(z_vals, 1 - q)
    h_z_max = quantile(h_z_vals, 1 - q)
    h_c_max = quantile(h_c_vals, 1 - q)
    h_d_max = quantile(h_d_vals, 1 - q)

    # Obtain lower bounds for states
    z_min = quantile(z_vals, q)
    h_z_min = quantile(h_z_vals, q)
    h_c_min = quantile(h_c_vals, q)
    h_d_min = quantile(h_d_vals, q)

    # Now build the grids
    z_grid = collect(linspace(z_min, z_max, gs_z))
    h_z_grid = collect(linspace(h_z_min, h_z_max, gs_h_z))
    h_c_grid = collect(linspace(h_c_min, h_c_max, gs_h_c))
    h_d_grid = collect(linspace(h_d_min, h_d_max, gs_h_d))
    
    # Draws for MC integration
    shocks = [randn(3) for n in 1:n_shocks]

    return SSYComputableModel(ez, 
                              cp, 
                              dp, 
                              z_grid, h_z_grid, h_c_grid, h_d_grid,
                              w_star, 
                              shocks) 
end

"""
If the constructor is called without a dividend model, use the SSY default
paramters.

"""
function SSYComputableModel(ez::EpsteinZin, cp::SSYConsumption)
    dp = SSYDividends()
    return SSYComputableModel(ez, cp, dp) 
end

"""
If called with nothing, use all defaults.

"""
function SSYComputableModel()
    ez = EpsteinZinSSY()
    cp = SSYConsumption()
    dp = SSYDividends()
    return SSYComputableModel(ez, cp, dp) 
end



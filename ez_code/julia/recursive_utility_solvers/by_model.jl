#=

Code for solving the Bansal-Yaron model based around Monte Carlo and
interpolation.

The Bansal Yaron consumption / dividend process is (see p. 1487)

    z' = ρ z + ϕ_z σ e'                 # z, ϕ_z here is x, ϕ_e in BY

    g = μ_c + z + σ η                   # consumption growth, μ_c is μ

    g_d = μ_d + α z + ϕ_d σ η           # div growth, α here is ϕ in BY

    (σ^2)' = v σ^2 + d + ϕ_σ w'         # v, d, ϕ_σ is v_1, σ^2(1-v_1), σ_w

where {e} and {w} are IID and N(0, 1). 


=#

using StatsBase

include("ez_parameters.jl")
include("consumption_models.jl")


"""
Struct for parameters of the BY model as described above.

"""
struct BYConsumption{T <: Real}  
    μ_c::T
    ρ::T
    ϕ_z::T
    v::T
    d::T
    ϕ_σ::T
end


"""
A constructor using parameters from the BY paper.  See table IV on page 1489.

"""
function BYConsumption(;μ_c=0.0015,
                        ρ=0.979,
                        ϕ_z=0.044,
                        v=0.987,
                        d=7.9092e-7,
                        ϕ_σ= 2.3e-6)

    return BYConsumption(μ_c,
                        ρ,
                        ϕ_z,
                        v,
                        d,
                        ϕ_σ)     
end


"""
Divdend process parameters of BY model

Log dividend growth g_d is given by

    g_d = μ_d + α z + φ_c σ η' + φ_d σ υ'

Shocks are IID and standard normal.

"""
struct BYDividends{T <: Real}  
    μ_d::T 
    α::T 
    ϕ_d::T
end


"""
Default dividend process values from Bansal and
and Yaron. 

"""
function BYDividends(; μ_d=0.0015,
                       α=3.0,     
                       ϕ_d=4.5)
                       
    return BYDividends(μ_d, α, ϕ_d)
end



"""
Simulate the state process and consumption, dividends for the BY model.  

Returns

    * X[1], ..., X[ts_length]
    * gc[2], ..., X[ts_length]
    * gd[2], ..., X[ts_length]

where

    gc[t] = ln(C[t]) - ln(C[t-1])
    gd[t] = ln(D[t]) - ln(D[t-1])


"""
function simulate(cp::BYConsumption, 
                  dp::BYDividends, 
                  ts_length=1000000, 
                  use_seed=true)

    # Set the seed to minimize variation
    if use_seed
        srand(1234)
    end

    # Unpack
    ρ, ϕ_z, v, d, ϕ_σ = cp.ρ, cp.ϕ_z, cp.v, cp.d, cp.ϕ_σ
    μ_c = cp.μ_c
    μ_d, α, ϕ_d = dp.μ_d, dp.α, dp.ϕ_d 

    # Allocate memory
    z_vals = zeros(ts_length)
    σ_vals = zeros(ts_length)
    c_vals = zeros(ts_length)
    d_vals = zeros(ts_length)

    σ_vals[1] = d / (1 - v)

    for t in 1:(ts_length-1)
        # Evaluate consumption and dividends
        c_vals[t+1] = μ_c + z_vals[t] + σ_vals[t] * randn()
        d_vals[t+1] = μ_d + α * z_vals[t] + ϕ_d * σ_vals[t] * randn()

        # Update state
        σ2 = v * σ_vals[t]^2 + d + ϕ_σ * randn()
        σ = sqrt(max(σ2, 0))
        σ_vals[t+1] = σ
        z_vals[t+1] = ρ * z_vals[t] + ϕ_z * σ * randn()
    end

    return c_vals[2:end], d_vals[2:end], z_vals, σ_vals
end


"""
The model constructed for computation.

"""
struct BYComputableModel <: ComputableModel

    ez::EpsteinZin{Float64}
    cp::BYConsumption{Float64}
    dp::BYDividends{Float64}
    z_grid::Vector{Float64}
    σ_grid::Vector{Float64}
    w_star::Matrix{Float64}   # Extra storage
    shocks::Array{Array{Float64, 1}, 1}    # Array of shocks for MC
end



function BYComputableModel(ez::EpsteinZin, 
                           cp::BYConsumption,
                           dp::BYDividends;
                           q=0.01,                    # quantitle for states
                           n_shocks::Int64=1000,      # for MC integrals
                           gs_z=8,       # z grid size
                           gs_σ=8,       # σ grid size
                           w_star=Array{Float64}(gs_z, gs_σ)) # guess of fp

    # Simulate the state process
    c_vals, d_vals, z_vals, σ_vals = simulate(cp, dp)

    # Obtain upper bounds for states
    z_max = quantile(z_vals, 1 - q)
    σ_max = quantile(σ_vals, 1 - q)

    # Obtain lower bounds for states
    z_min = quantile(z_vals, q)
    σ_min = quantile(σ_vals, q)

    # Now build the grids
    z_grid = linspace(z_min, z_max, gs_σ)
    σ_grid = linspace(σ_min, σ_max, gs_z)
    z_grid = collect(z_grid)
    σ_grid = collect(σ_grid)

    # Draws for MC integration
    shocks = [randn(2) for n in 1:n_shocks]

    return BYComputableModel(ez, 
                             cp, 
                             dp, 
                             z_grid, σ_grid, 
                             w_star, 
                             shocks) 
end


"""
If the constructor is called without a dividend model, use the BY default
paramters.

"""
function BYComputableModel(ez::EpsteinZin, cp::BYConsumption)
    dp = BYDividends()
    return BYComputableModel(ez, cp, dp) 
end

"""
If called with nothing, use all defaults.

"""
function BYComputableModel()
    ez = EpsteinZinBY()
    cp = BYConsumption()
    dp = BYDividends()
    return BYComputableModel(ez, cp, dp) 
end


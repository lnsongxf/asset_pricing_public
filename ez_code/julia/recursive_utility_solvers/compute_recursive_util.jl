#=

Code for solving for recursive utility in the Schorfheide, Song and Yaron
model.  We will compute the fixed point of the Koopmans operator 

    Tw(x) = ζ + (Kw^θ(x))^(1/θ)

by iteration.
=#

include("ssy_k_op.jl")
include("by_k_op.jl")

"""
Apply the operator 

    Tw(x) = ζ + (Kw^θ(x))^(1/θ)

"""
function T(w, cm::ComputableModel)

    θ, ζ = cm.ez.θ, cm.ez.ζ

    g = w.^θ
    Kg = K_interp(g, cm)   # Will dispatch on the second argument

    Tw = ζ .+ Kg.^(1/θ)
    return Tw

end




"""
Solve for the fixed point of T and store as cm.w_star 

"""
function compute_recursive_utility!(cm::ComputableModel;
                                    w=nothing,
                                    verbose=false, 
                                    tol=1e-6,     
                                    max_iter=50000) 
    # Initial condition
    if w == nothing
        w = ones(size(cm.w_star))
    end

    error = tol + 1
    i = 1
    while error > tol && i < max_iter
        w_next = T(w, cm)
        error = maximum(abs, w - w_next)
        i += 1
        w = w_next
    end

    if i == max_iter
        warn("Hit iteration upper bound when computing fixed point of T!")
    end

    if verbose == true
        msg = """

        Recursive utility calculation converged in $i iterations 
        using tolerance $tol.
        """
        println(msg)
    end

    copy!(cm.w_star, w)
end


"""
A function for testing purposes.  The solution w = w_star should satisfy


    w(x) = ζ + (Kw^θ(x))^(1/θ)

"""

function test_ez_solution(cm::ComputableModel, recompute=false)

    # Solve the model so that w_star contains the fixed point of ϕ ∘ K
    if recompute
        compute_recursive_utility!(cm)
    end

    w = cm.w_star

    f = w - T(w, cm)
    return maximum(f)
end


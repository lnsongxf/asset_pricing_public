#=

Stability test values for prices, for the SSY model.

=#


include("ssy_price_val_op.jl")

"""
Compute the local spectral radius (and hence the spectral radius)
of the valuation operator Phi by iteration.

"""
function compute_price_spec_rad_coeff!(scm::SSYComputableModel;
                                       tol=1e-7, 
                                       max_iter=5000,
                                       recompute=true) 

    # Compute the function w_star, which forms a part of the valuation
    # operator Phi, and write it to scm.w_star.  
    # Note that this will be faster if you supply a good initial guess of
    # w_star
    if recompute
        compute_recursive_utility!(scm, verbose=true)
    end

    h = ones(size(scm.q_star))

    error = tol + 1
    r = 1
    i = 1

    while error > tol && i < max_iter
        # Take scm.h and use it to compute Phi h, write to scm.h_next
        s, h_next = Phi(h, scm)
        new_r = s^(1/i)
        error = abs(new_r - r)
        i += 1
        r = new_r
        h = h_next
    end


    println("Spec rad computation converged in $i iterations.")
    return r
    
end


function compute_price_spec_rad_coeff(ez::EpsteinZin,
                                      sc::SSYConsumption,
                                      sd::SSYDividends)
    println("Generating a fresh instance of SSYComputableModel.")
    scm = SSYComputableModel(ez, sc, sd)
    compute_price_spec_rad_coeff!(scm)
end


"""
For testing purposes:

"""
function compute_price_spec_rad_coeff()

    ez = EpsteinZinSSY(ζ=1.0)
    sc = SSYConsumption()
    sd = SSYDividends()

    compute_price_spec_rad_coeff(ez, sc, sd)
end



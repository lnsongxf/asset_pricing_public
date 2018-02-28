
include("compute_spec_rad.jl")

using PyPlot
plt = PyPlot


function stability_plot(model_type::String,
                        param1::Symbol,    # parameter on x axis
                        p1_reduction_factor::Float64,   # min value for param1
                        p1_boost_factor::Float64,   # min value for param1
                        param2::Symbol,    # parameter on y axis
                        p2_reduction_factor::Float64,   # min value for param2
                        p2_boost_factor::Float64;   # min value for param2
                        coords=(-225, 30),
                        save=true,
                        G=20)              # grid size for x and y axes


    # Allocate arrays, set up parameter grid
    R = Array{Float64}(G, G)

    # Set text, generate default instances, set names
    if model_type == "by"
        default_cm = BYComputableModel()
        text = "Bansal and Yaron "
        EpsteinZin = EpsteinZinBY
        Consumption = BYConsumption
        Dividends = BYDividends
        ComputableModel = BYComputableModel
    else
        default_cm = SSYComputableModel()
        text = "Schorfheide, Song and Yaron "
        EpsteinZin = EpsteinZinSSY
        Consumption = SSYConsumption
        Dividends = SSYDividends
        ComputableModel = SSYComputableModel
    end

    # Simplify names
    default_ez = default_cm.ez
    default_cp = default_cm.cp
    default_dp = default_cm.dp

    # Extract parameter names
    ez_names = fieldnames(default_ez)
    cp_names = fieldnames(default_cp)
    dp_names = fieldnames(default_dp)

    # Extract values from defaults for plotting 
    if param1 in cp_names
        param1_value = getfield(default_cp, param1)
    elseif param1 in dp_names
        param1_value = getfield(default_dp, param1)
    elseif param1 in ez_names
        param1_value = getfield(default_ez, param1)
    else
        println("parameter 1 not found")
        throw(DomainError)
    end

    if param2 in cp_names
        param2_value = getfield(default_cp, param2)
    elseif param2 in dp_names
        param2_value = getfield(default_dp, param2)
    elseif param2 in ez_names
        param2_value = getfield(default_ez, param2)
    else
        println("parameter 2 not found")
        throw(DomainError)
    end

    p1_min = param1_value * (1 - p1_reduction_factor)
    p1_max = param1_value * (1 + p1_boost_factor)
    p2_min = param2_value * (1 - p2_reduction_factor)
    p2_max = param2_value * (1 + p2_boost_factor)

    x_vals = linspace(p1_min, p1_max, G)   # values for param1 
    y_vals = linspace(p2_min, p2_max, G)   # values for param2

    compute_recursive_utility!(default_cm, tol=1e-4)
    current_w = default_cm.w_star

    # Loop through parameters computing test coefficient
    for (i, x) in enumerate(x_vals)
        for (j, y) in enumerate(y_vals)

            # Construct dictionaries used to build instances
            ez_dict = Dict()
            cp_dict = Dict()
            dp_dict = Dict()

            for (param, val) in zip((param1, param2), (x, y))
                if param in ez_names
                    ez_dict[param] = val
                elseif param in cp_names
                    cp_dict[param] = val
                else param in dp_names
                    dp_dict[param] = val
                end
            end

            # Construct instances
            ez = EpsteinZin(; ez_dict...)
            cp = Consumption(; cp_dict...)
            dp = Dividends(; dp_dict...)

            cm = ComputableModel(ez, cp, dp)
            compute_recursive_utility!(cm, w=current_w, tol=1e-7)

            r = compute_spec_rad_by_sim(cm, n=600, m=2000)
            R[i, j] = r

            # Use current fixed point as initial condition
            current_w = cm.w_star
        end
    end

    fig, ax = plt.subplots(figsize=(10, 5.7))

    cs1 = ax[:contourf](x_vals, 
                        y_vals, 
                        R', # cmap=plt.cm[:jet],
                        alpha=0.5)

    ctr1 = ax[:contour](x_vals, 
                        y_vals, 
                        R', 
                        levels=[1.0])

    plt.clabel(ctr1, inline=1, fontsize=13)
    plt.colorbar(cs1, ax=ax, format="%.6f")


    ax[:annotate](text, 
             xy=(param1_value, param2_value),  
             xycoords="data",
             xytext=coords,
             textcoords="offset points",
             fontsize=12,
             arrowprops=Dict("arrowstyle" => "->"))

    ax[:plot]([param1_value], [param2_value],  "ko", alpha=0.6)

    #ax[:set_title]("Spectral radius")
    ax[:set_xlabel](String(param1), fontsize=16)
    ax[:set_ylabel](String(param2), fontsize=16)


    ax[:ticklabel_format](useOffset=false)


    if save == true
        filename = String(param1) * String(param2) * model_type 
        filename = filename * "_" * ".pdf"
        plt.savefig(filename)
    end


    plt.show()

end

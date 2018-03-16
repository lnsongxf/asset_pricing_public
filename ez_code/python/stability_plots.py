import numpy as np
from by_model import BY
import matplotlib.pyplot as plt
import unicodedata

def generate_plot_by(param1,                # string
                     p1_reduction_factor,   # min value for param1
                     p1_boost_factor,       # min value for param1
                     param2,                # string 
                     p2_reduction_factor,   # min value for param2
                     p2_boost_factor,       # min value for param2
                     one_step=False,        # one step contraction coeff
                     xlabel=None,           # optional
                     ylabel=None,           # optional
                     coords=(-225, 30),     # relative location of text
                     G=3):                 # grid size for x and y axes

    # Normalize unicode identifiers
    param1 = unicodedata.normalize('NFKC', param1)
    param2 = unicodedata.normalize('NFKC', param2)

    # Allocate arrays, set up parameter grid
    R = np.empty((G, G))

    by = BY(z_grid_size=10, σ_grid_size=10)

    param1_value = by.__getattribute__(param1)
    param2_value = by.__getattribute__(param2)

    p1_min = param1_value * p1_reduction_factor
    p1_max = param1_value * p1_boost_factor
    p2_min = param2_value * p2_reduction_factor
    p2_max = param2_value * p2_boost_factor

    x_vals = np.linspace(p1_min, p1_max, G)   # values for param1 
    y_vals = np.linspace(p2_min, p2_max, G)   # values for param2


    # Recompute utility, unless both parameters only relate to dividends
    must_recompute_utility = True
    if by.is_dividend_parameter(param1) and by.is_dividend_parameter(param2):
        must_recompute_utility = False

    # Loop through parameters computing test coefficient
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):

            by.__setattr__(param1, x)
            by.__setattr__(param2, y)
            if must_recompute_utility:
                by.compute_grid_and_solution()

            if one_step:
                r = by.compute_spec_rad_with_sup(n=1, num_reps=2000)
            else:
                r = by.compute_spec_rad(n=1000, num_reps=2000)

            R[i, j] = r

    # Now the plot
    point_location=(param1_value, param2_value)

    fig, ax = plt.subplots(figsize=(10, 5.7))

    cs1 = ax.contourf(x_vals, y_vals, R.T, alpha=0.5)
    ctr1 = ax.contour(x_vals, y_vals, R.T, levels=[1.0])

    plt.clabel(ctr1, inline=1, fontsize=13)
    plt.colorbar(cs1, ax=ax, format="%.6f")

    ax.annotate("Bansal-Yaron", 
             xy=point_location,  
             xycoords="data",
             xytext=coords,
             textcoords="offset points",
             fontsize=12,
             arrowprops={"arrowstyle" : "->"})

    ax.plot(*point_location,  "ko", alpha=0.6)

    if one_step:
        title = "One step contraction coefficient"
    else:
        title = "Spectral radius"

    ax.set_title(title)

    if xlabel is None:
        xlabel = param1
    ax.set_xlabel(xlabel, fontsize=16)

    if ylabel is None:
        ylabel = param2
    ax.set_ylabel(ylabel, fontsize=16)

    ax.ticklabel_format(useOffset=False)

    if one_step:
        filename = param1 + param2 + "by" + "_onestep_" + ".pdf"
    else:
        filename = param1 + param2 + "by" + "_" + ".pdf"

    plt.savefig(filename)
    
    plt.show()


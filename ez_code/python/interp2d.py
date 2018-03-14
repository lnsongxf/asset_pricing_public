from numba import njit
import numpy as np


@njit(cache=True)
def lininterp_2d(x_grid, y_grid, vals, s):
    """
    Fast 2D interpolation.  Uses linear extrapolation for points outside the
    grid.  

    Parameters
    ----------

        x_grid: grid points for x, one dimensional

        y_grid: grid points for y, one dimensional

        vals: vals[i, j] = f(x[i], y[j])
         
        s: 2D point at which to evaluate

    """

    nx = len(x_grid)
    ny = len(y_grid)

    ax, bx = x_grid[0], x_grid[-1]
    ay, by = y_grid[0], y_grid[-1]

    s_0 = s[0]
    s_1 = s[1]

    # (sn_1, ..., sn_d) : normalized evaluation point (in [0,1] inside the grid)
    sn_0 = (s_0 - ax)/(bx - ax)
    sn_1 = (s_1 - ay)/(by - ay)

    # q_k : index of the interval "containing" s_k
    q_0 = max(min(int(sn_0 *(nx - 1)), (nx - 2) ), 0)
    q_1 = max(min(int(sn_1 *(ny - 1)), (ny - 2) ), 0)

    # lam_k : barycentric coordinate in interval k
    lam_0 = sn_0 * (nx-1) - q_0
    lam_1 = sn_1 * (ny-1) - q_1

    # v_ij: values on vertices of hypercube "containing" the point
    v_00 = vals[(q_0), (q_1)]
    v_01 = vals[(q_0), (q_1+1)]
    v_10 = vals[(q_0+1), (q_1)]
    v_11 = vals[(q_0+1), (q_1+1)]

    # interpolated/extrapolated value
    out = (1-lam_0) * ((1-lam_1) * (v_00) + \
                (lam_1) * (v_01)) + (lam_0) * ((1-lam_1) * (v_10) \
                + (lam_1) * (v_11))

    return out




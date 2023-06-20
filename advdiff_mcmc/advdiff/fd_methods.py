"""
Functions associated with building a 1D advection diffusion model using the Crank-Nicolson method.
fd_methods.py

This version is an improvement on the original version used in the JGR JAMES journal (Edge, 2023).
Improvements are mostly related to code conciseness and improved documentation for functions, although some
material changes have been made.

Planned updates:
 - non-regular gridding
 - inclusion of a 4th-order compact advection scheme as an option (see Steinle PhD thesis, p57)
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


def compile_CN_forwardmatrix(Kprof, wprof, lmbda, alpha):
    '''
    Create a ZxZ sparse forward matrix using the implicit Crank-Nicolson method in 1D with variable advection and diffusion.

    ....
    Inputs
    ------
        Kprof: profile of variable diffusivity for a single timestep (t+1). Ordered from the erodible bed upwards.
        wprof: profile of variable velocities in the spatial dimension for a single timestep (t+1). Ordered from the erodible bed upwards.
        lmbda: diffusion coefficient for a regular (aka. rectilinear or constant) grid, as calculated by 'CN_coefficients'.
                **Future work could update this to a variable coefficient.
        alpha: advection coefficient for a regular (aka. rectilinear or constant) grid, as calculated by 'CN_coefficients'.
                **Future work could update this to a variable coefficient.

    Outputs
    -------
        matrix A: the forward matrix (t+1) in the Crank-Nicolson equation, an ZxZ scipy sparse matrix in csc format.
    '''

    # Courant number c = w del.t/del.x (not required for now)

    A_back = -1 * alpha * wprof[1:-1] \
                - lmbda * Kprof[1:-1]

    A_ford = +1 * alpha * wprof[1:-1] \
                - lmbda * Kprof[1:-1]

    A_midd =  1 + alpha * wprof[1:] \
                - alpha * wprof[:-1] \
                + lmbda * Kprof[:-1] \
                + lmbda * Kprof[1:]

    return diags([A_back, A_midd, A_ford], [-1, 0, 1], format = 'csc')


def compile_CN_backmatrix(Kprof, wprof, lmbda, alpha):
    '''
    Create a ZxZ sparse backward matrix using the implicit Crank-Nicolson method in 1D with variable advection and diffusion.

    ....
    Inputs
    ------
    All as per 'compile_CN_forwardmatrix' but for time=t (for Kprof and wprof)

    Outputs
    -------
        matrix B: the backward matrix (t) in the Crank-Nicolson equation, an ZxZ scipy sparse matrix in csc format.
    '''
    return compile_CN_forwardmatrix(Kprof, wprof, -1*lmbda, -1*alpha)


def CN_coefficients(time_grid, mean_grid, reg_grid=True):
    '''
    Compute the advection and diffusion coefficients for the the Crank-Nicolson method with a regular grid.
    Could be updated for non-regular grids in future.

    ....
    Inputs
    ------
        time_grid: well-spaced time steps (only uses the first time step)
        mean_grid: well-spaced grid points (only uses the first grid spacing)
        reg_grid: throws an error if false(a reminder to update for non-regular grids later)

    Outputs
    -------
        lmbda: diffusion coefficient
        alpha: advection coefficient
        beta: source term coefficient
    '''
    
    if not reg_grid: # Need to update this - just a prompt for now
        raise ValueError('Model not setup for variable grid sizes')
    else:
        del_t = np.diff(time_grid)[0]
        del_z = np.diff(mean_grid)[0]

        lmbda = del_t / (2 * del_z**2)
        alpha = del_t / (4 * del_z)
        beta  = del_t / (2 * del_z)
    return lmbda, alpha, beta


def eflux_mod1(bed_strss, tcr, e0, bc):
    '''
    Function to return erosion flux for model E1
    '''
    return e0 * (bed_strss - tcr)**bc


def eflux_mod2(bed_strss, mass_eroded, tcr, e0, mc, bc):
    '''
    Function to return erosion flux for model E1
    '''
    tcr_eff = tcr + mc * mass_eroded**bc
    return e0 * (bed_strss - tcr_eff)


def conc_etxr(c_prof):
    '''
    Function to estimate the tracer concentration at C_bed using quadratic extrapolation.
    '''
    return (3*c_prof[0] - 3*c_prof[1] + c_prof[0])/2
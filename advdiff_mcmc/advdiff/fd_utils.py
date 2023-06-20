"""
Utility functions for evaluating a 1D advection - diffusion model
fd_utils.py

This version is an improvement on the original version used in the JGR JAMES journal (Edge, 2023).
Improvements are mostly related to code conciseness and improved documentation for functions, although some
material changes have been made. 
"""

import numpy as np
from scipy.interpolate import interp1d


def generate_grid(H, del_h, T, del_t):
    """
    Function to calculate evenly spaced grid points

    ....
    Inputs
    ------
        H: total model height [m].
        del_h: spatial grid cell spacing [m].
        T: total time [s].
        del_T: time step [s].

    Outputs
    -------
        fg: flux grid at cell boundaries [m].
        mg: mean grid at cell centres [m].
        tg: time grid [s].
    """

    fg = np.arange(0, H + 0.5*del_h, del_h)
    mg = np.arange(0.5*del_h, H, del_h)
    tg = np.arange(0, T + 0.5*del_t, del_t)
    
    return fg, mg, tg
    

def calc_Ks(u_b, fg, bbl_h, km_bg):
    """
    Function to calculate diffusivity profile (time, space)

    ....
    Inputs
    ------
        u_b: friction velocity
        fg: flux grid
        bbl_h: time varying bottom boundary layer thickness
        km_bg: background diffusivity above bottom boundary layer
    
    Outputs
    -------
        Ks: time-space grid of diffusivity
    """

    k_par = np.full((len(fg),len(u_b)), np.nan)
    Ks = np.full((len(fg),len(u_b)), np.nan)
    
    for ix, (bh, ubxx) in enumerate(zip(bbl_h, u_b)):
        
        k_par[:,ix] = 0.4 * ubxx * fg * (bh - fg)/bh
        k_ind = (fg > bh/2) & (k_par[:,ix] < km_bg)
        k_par[:,ix][k_ind] = km_bg
    
        k_cix = (fg > bh+0.6) | (fg < bh-0.6)
        if np.sum(fg[k_cix])>0:
            k_ficub = interp1d(fg[k_cix], k_par[:,ix][k_cix], kind='cubic')
            Ks[:,ix] = k_ficub(fg)
        else:
            Ks[:,ix] = k_par[:,ix]
        
    k_par_ix = Ks[1:,:] < km_bg
    Ks[1:,:][k_par_ix] = km_bg
    Ks[0,:] = 0
        
    return Ks.T


def calc_Ks_new(u_b, fg, bbl_h, km_bg):
    """
    New function to calculate diffusivity profile (time, space)
    as suggested by O. Fringer during paper review.
    Ks = Ku*z(1 + a - z/bbl_h)
    where a = km_bg / (Ku*bbl_h)

    ....
    Inputs
    ------
        u_b: friction velocity
        fg: flux grid
        bbl_h: time varying bottom boundary layer thickness
        km_bg: background diffusivity above bottom boundary layer
    
    Outputs
    -------
        Ks: time-space grid of diffusivity
    """

    Ks = np.full((len(fg),len(u_b)), np.nan)
    
    for ix, (bh, ubxx) in enumerate(zip(bbl_h, u_b)):
        
        alp = km_bg / (0.4 * ubxx * bh)
        Ks[:,ix] = 0.4 * ubxx * fg * (1 + alp - fg/bh)
        
    return Ks.T
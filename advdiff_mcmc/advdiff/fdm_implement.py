"""
Function to build and run a 1D Crank-Nicolson advection-diffusion model
fd_implement.py

This version is an improvement on the original version used in the JGR JAMES journal (Edge, 2023).
Improvements are mostly related to code conciseness and improved documentation for functions, although some
material changes have been made. 

Notes:
 - new deposition formula implemented as recommended by O. Fringer. Note that with this setup the solving is not implicit
   and so the deposition can only be calculated based on the tracer concetration at the current timestep, not the future one. 
   I think both should be an option in future.
"""

import numpy as np
from advdiff_mcmc.advdiff.fd_methods import *


def obj_CN(theta, erosion_model, model_spec):
    '''
    Build and run the 1D finite-difference model in a loop using the Crank-Nicolson method.

    ....
    Inputs
    ------
        theta: vector of parameters in correct order (see function).
        erosion_model: string for either erosion model 1 ('E1') or erosion model 2 ('E2').
        model_spec: list of objects used to force the model in order [time grid, mean grid, flux grid, initial profile,
                    diffusivity time-space series, BBL height time series, bed stress time series]

    Outputs
    -------
        cf: the full time-space tracer concentration grid.
    '''

    # Initiate model domain
    tg, mg, fg = model_spec[0], model_spec[1], model_spec[2]
    del_z = np.diff(mg)[0]

    # Create empty tracer concentration grid and fill initial profile values
    cf = np.zeros((len(tg), len(mg)))
    cf[0,:] = model_spec[3]
    
    # Load other variables required for model
    Ks_all, h_tot, tau_bed = model_spec[4], model_spec[5], model_spec[6]

    # Set up model coefficients
    lmbda_m, alpha_m, beta_m = CN_coefficients(tg, mg, reg_grid=True)

    # Adjust parameter scaling
    w_s_mod = theta[0] / 10**3
    e_0_mod = theta[1] / 10**4
    t_c_mod = theta[2] / 10**2
    if erosion_model == 'E2':
        m_c_mod = theta[3] / 10**1
    b_c_mod = theta[4]

    # Set the settling velocity and deposition rules
    w_prof_mod = np.zeros((len(tg), len(fg))) - w_s_mod
    w_prof_mod[:,0] = 0.0        
    w_prof_mod[:,-1] = 0.0
    
    # Loop through timesteps and calculate
    for itxx, tstep in enumerate(tg[:-1]):
        
        #### Create matrices
        A = compile_CN_forwardmatrix(Ks_all[itxx+1], w_prof_mod[itxx+1,:], lmbda_m, alpha_m)
        B = compile_CN_backmatrix(Ks_all[itxx], w_prof_mod[itxx,:], lmbda_m, alpha_m)

        # Copy out C-profile at t and blank profiles for source terms
        Ct = np.copy(cf[itxx,:])
        # b = np.copy(cf[itxx,:])
        b = np.copy(cf[itxx+1,:]) # blank vector
        a = np.copy(cf[itxx+1,:])

        # Compute erosion flux at t & t+1 and add to conc. profiles
        if tau_bed[itxx] > t_c_mod:
            if erosion_model == 'E1':
                b[0] = b[0] + beta_m * eflux_mod1(tau_bed[itxx], t_c_mod, e_0_mod, b_c_mod)
            elif erosion_model == 'E2':
                total_mass = np.sum(cf[itxx,:])*del_z
                b[0] = b[0] + beta_m * eflux_mod2(tau_bed[itxx], total_mass, t_c_mod, e_0_mod, m_c_mod, b_c_mod)

        if tau_bed[itxx+1] > t_c_mod:
            if erosion_model == 'E1':
                a[0] = a[0] - beta_m * eflux_mod1(tau_bed[itxx+1], t_c_mod, e_0_mod, b_c_mod)
            elif erosion_model == 'E2':
                total_mass = np.sum(cf[itxx+1,:])*del_z
                a[0] = a[0] - beta_m * eflux_mod2(tau_bed[itxx+1], total_mass, t_c_mod, e_0_mod, m_c_mod, b_c_mod)

        # Compute deposition flux by quadratic extrapolation to bed (Chou, 2008) (only based on timestep t)
        b[0] = b[0] - beta_m * w_s_mod * conc_etxr(cf[itxx,:])
        a[0] = a[0] + beta_m * w_s_mod * conc_etxr(cf[itxx,:])

        # Implicit sparse matrix solve
        # F = B.dot(b) - a
        F = B.dot(Ct) + b - a
        cf[itxx+1,:] = spsolve(A, F)

    return cf
## Functions for 

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import theano.tensor as tt
from scipy.interpolate import interp1d, interp2d

def generate_grid(H, del_h, T, del_t):
    ## Function to calculate evenly spaced grid 
    # fg = flux grid at cell boundaries
    # mg = mean grid at cell centres
    # tg = time grid
    fg = np.arange(0, H + 0.5*del_h, del_h)
    mg = np.arange(0.5*del_h, H, del_h)
    tg = np.arange(0, T + 0.5*del_t, del_t)
    
    return fg, mg, tg
    
def calc_Ks(u_b, fg, bbl_h, km_bg):
    ## Function to calculate diffusivity profile (time, space)
    # u_b = bed stress
    # fg = flux grid
    # bbl_h = time varying bottom boundary layer thickness
    # km_bg = background diffusivity above bottom boundary layer
    k_par = np.full((len(fg),len(u_b)), np.nan)
    k_par_cub = np.full((len(fg),len(u_b)), np.nan)
    
    for ix, (bh, ubxx) in enumerate(zip(bbl_h, u_b)):
        
        k_par[:,ix] = 0.4 * ubxx * fg * (bh - fg)/bh
        k_ind = (fg > bh/2) & (k_par[:,ix] < km_bg)
        k_par[:,ix][k_ind] = km_bg
    
        k_cix = (fg > bh+0.6) | (fg < bh-0.6)
        if np.sum(fg[k_cix])>0:
            k_ficub = interp1d(fg[k_cix], k_par[:,ix][k_cix], kind='cubic')
            k_par_cub[:,ix] = k_ficub(fg)
        else:
            k_par_cub[:,ix] = k_par[:,ix]
        
    k_par_ix = k_par_cub[1:,:] < km_bg
    k_par_cub[1:,:][k_par_ix] = km_bg
    k_par_cub[0,:] = 0
        
    return k_par_cub.T
    
def my_loglike_mean(theta, data, erosion_model, model_spec, obs_htx, deposition):
    model_out = obj(theta, erosion_model, model_spec, deposition)
    logppp = -np.sum(0.5*(np.log(2*np.pi*(theta[-1]/100)**2) + \
                            ((data - np.mean(model_out[:,obs_htx], axis=1))/(theta[-1]/100))**2))
    return logppp
    
def my_loglike(theta, data, erosion_model, model_spec, obs_htx, deposition):
    model_out = obj(theta, erosion_model, model_spec, deposition)
    logppp = -np.sum(0.5*(np.log(2*np.pi*(theta[-1]/1000)**2) + \
                            ((data[:,obs_htx] - model_out[:,obs_htx])/(theta[-1]/1000))**2))
    return logppp    
    
class LogLike(tt.Op):
    itypes = [tt.dvector]
    otypes = [tt.dscalar]

    def __init__(self, loglike, data, erosion_model, model_spec, obs_htx, deposition):
        self.likelihood = loglike
        self.data = data
        self.erosion_model = erosion_model
        self.model_spec = model_spec
        self.obs_htx = obs_htx
        self.deposition = deposition

    def perform(self, node, inputs, outputs):
        (theta,) = inputs
        # call the log-likelihood function
        logl = self.likelihood(theta, self.data,\
                self.erosion_model, self.model_spec, self.obs_htx, self.deposition)
        outputs[0][0] = np.array(logl)
        
    
    
def obj(theta, erosion_model, model_spec, deposition):
    ## The advection diffusion model

    # Initiate model domain
    tg = model_spec[0]
    mg = model_spec[1]
    fg = model_spec[2]
    del_t = np.diff(tg)[0]
    del_z = np.diff(mg)[0]
    
    cf = np.zeros((len(tg), len(mg)))
    cf[0,:] = model_spec[3]
    
    Ks_all = model_spec[4]
    h_tot = model_spec[5]
    tau_bed = model_spec[6]

    # Set up model coefficients
    fir = del_t / del_z
    sec = del_t / del_z**2

    # Adjust parameter scaling
    w_s_mod = theta[0] / 10**3
    e_0_mod = theta[1] / 10**4
    t_c_mod = theta[2] / 10**2
    if erosion_model == 'E2':
        m_r_mod = theta[3] / 10**1
    b_c_mod = theta[4]

    # Set the settling velocity and deposition rules
    w_prof_mod = np.zeros((len(tg), len(fg))) - w_s_mod
    if not deposition:
        w_prof_mod[:,0] = 0.0        
    w_prof_mod[:,-1] = 0.0
    
    # Loop through timesteps and calculate
    for itxx, tstep in enumerate(tg[:-1]):
        
        Ks_t = Ks_all[itxx]
        Ks_x = Ks_all[itxx+1]
        
        #### Create matrices
        A_diag_bak = -1*(fir/4)*w_prof_mod[itxx+1,1:-1] \
                        + (2 + fir*w_prof_mod[itxx+1,1:-1]**2)/12 \
                        - (sec/2)*Ks_x[1:-1]

        A_diag_for = (fir/4)*w_prof_mod[itxx+1,1:-1] \
                        + (2 + fir*w_prof_mod[itxx+1,1:-1]**2)/12 \
                        - (sec/2)*Ks_x[1:-1]

        A_diag_c_all = 1 + (fir/4)*w_prof_mod[itxx+1,1:] \
                            - (fir/4)*w_prof_mod[itxx+1,0:-1] \
                            - (2 + fir*w_prof_mod[itxx+1,0:-1]**2)/12 \
                            - (2 + fir*w_prof_mod[itxx+1,1:]**2)/12 \
                            + (sec/2)*Ks_x[0:-1] \
                            + (sec/2)*Ks_x[1:]

        if deposition:
            A_diag_c_all[0] = A_diag_c_all[0] \
                                - 3*(fir/4)*w_prof_mod[itxx+1,0] \
                                + 3*(2 + fir*w_prof_mod[itxx+1,0]**2)/12
        
            A_diag_for[0] = A_diag_for[0] \
                                + 3*(fir/4)*w_prof_mod[itxx+1,0] \
                                - 3*(2 + fir*w_prof_mod[itxx+1,0]**2)/12
        
            A_diag_for22 = np.zeros(len(A_diag_for)-1)
            A_diag_for22[0] = - (fir/4)*w_prof_mod[itxx+1,0] \
                                + (2 + fir*w_prof_mod[itxx+1,0]**2)/12
        
            A = diags([A_diag_bak, A_diag_c_all, A_diag_for, A_diag_for22], [-1, 0, 1, 2], format = 'csc')
        else:
            A = diags([A_diag_bak, A_diag_c_all, A_diag_for], [-1, 0, 1], format = 'csc')

        ####
        C_diag_bak = (fir/4)*w_prof_mod[itxx,1:-1] \
                        + (2 + fir*w_prof_mod[itxx,1:-1]**2)/12 \
                        + (sec/2)*Ks_t[1:-1]

        C_diag_for = -1*(fir/4)*w_prof_mod[itxx,1:-1] \
                        + (2 + fir*w_prof_mod[itxx,1:-1]**2)/12 \
                        + (sec/2)*Ks_t[1:-1]

        C_diag_c_all = 1 - (fir/4)*w_prof_mod[itxx,1:] \
                            + (fir/4)*w_prof_mod[itxx,0:-1] \
                            - (2 + fir*w_prof_mod[itxx,0:-1]**2)/12 \
                            - (2 + fir*w_prof_mod[itxx,1:]**2)/12 \
                            - (sec/2)*Ks_t[0:-1] \
                            - (sec/2)*Ks_t[1:]

        if deposition:
            C_diag_c_all[0] = C_diag_c_all[0] \
                                + 3*(fir/4)*w_prof_mod[itxx,0] \
                                + 3*(2 + fir*w_prof_mod[itxx,0]**2)/12
    
            C_diag_for[0] = C_diag_for[0] \
                                - 3*(fir/4)*w_prof_mod[itxx,0] \
                                - 3*(2 + fir*w_prof_mod[itxx,0]**2)/12

            C_diag_for22 = np.zeros(len(C_diag_for)-1)
            C_diag_for22[0] = + (fir/4)*w_prof_mod[itxx,0] \
                            + (2 + fir*w_prof_mod[itxx,0]**2)/12

            C = diags([C_diag_bak, C_diag_c_all, C_diag_for, C_diag_for22], [-1, 0, 1, 2], format = 'csc')
        else:
            C = diags([C_diag_bak, C_diag_c_all, C_diag_for], [-1, 0, 1], format = 'csc')

        # Copy out C-profiles at t & t+1
        b = np.copy(cf[itxx,:])
        d = np.copy(cf[itxx+1,:])

        # Compute erosion flux at t & t+1 and add to C profiles
        if erosion_model == 'E1':
            t_z = t_c_mod
            if tau_bed[itxx] > t_z:
                b[0] = b[0] + fir*e_0_mod*((tau_bed[itxx] - t_z))**b_c_mod
            if tau_bed[itxx+1] > t_z:
                d[0] = d[0] - fir*e_0_mod*((tau_bed[itxx+1] - t_z))**b_c_mod
                
        elif erosion_model == 'E2':
            t_z = t_c_mod + m_r_mod*(np.sum(cf[itxx,:])*del_z)**b_c_mod
            if tau_bed[itxx] > t_z:
                b[0] = b[0] + fir*e_0_mod*((tau_bed[itxx] - t_z))
            if tau_bed[itxx+1] > t_z:
                d[0] = d[0] - fir*e_0_mod*((tau_bed[itxx+1] - t_z))

        # Implicit sparse matrix solve
        F = C.dot(b) - d
        cf[itxx+1,:] = spsolve(A, F)

    return cf
    
    
# def obj_SS(theta, erosion_model, model_spec):

    # # Initiate model domain
    # tg = model_spec[0]
    # mg = model_spec[1]
    # fg = model_spec[2]
    # del_t = np.diff(tg)[0]
    # del_z = np.diff(mg)[0]
    
    # cf = np.zeros((len(tg), len(mg)))
    # cf[0,:] = model_spec[3]
    
    # Ks_all = model_spec[4]
    # h_tot = model_spec[5]
    # tau_bed = model_spec[6]

    # # Set up model coefficients
    # fir = del_t / del_z
    # sec = del_t / del_z**2

    # # Adjust parameter scaling
    # w_s_mod = theta[0] / 10**3
    # e_0_mod = theta[1] / 10**4
    # t_c_mod = theta[2] / 10**2
    # if erosion_model == 'E2':
        # m_r_mod = theta[3] / 10**1
    # b_c_mod = theta[4]        

    # # Set the settling velocity and deposition rules
    # w_prof_mod = np.zeros((len(tg), len(fg))) - w_s_mod
    # w_prof_mod[:,-1] = 0.0    # No advection at the top of model
    # w_prof_mod[:,0] = 0.0     # No advection at the bottom of model (no deposition!!)

    # # Loop through timesteps and calculate
    # for itxx, tstep in enumerate(tg[:-1]):

        # # Calculate instantaneous diffusivity profile
        # Ks_t = Ks_all[itxx]
        # Ks_x = Ks_all[itxx+1]

        # #### Create matrices (LHS)
        # A_diag_bak = -1*(fir/4)*w_prof_mod[itxx+1,1:-1] \
                        # + (2 + fir*w_prof_mod[itxx+1,1:-1]**2)/12 \
                        # - (sec/2)*Ks_x[1:-1]

        # A_diag_for = (fir/4)*w_prof_mod[itxx+1,1:-1] \
                        # + (2 + fir*w_prof_mod[itxx+1,1:-1]**2)/12 \
                        # - (sec/2)*Ks_x[1:-1]

        # A_diag_c_all = 1 + (fir/4)*w_prof_mod[itxx+1,1:] \
                            # - (fir/4)*w_prof_mod[itxx+1,0:-1] \
                            # - (2 + fir*w_prof_mod[itxx+1,0:-1]**2)/12 \
                            # - (2 + fir*w_prof_mod[itxx+1,1:]**2)/12 \
                            # + (sec/2)*Ks_x[0:-1] \
                            # + (sec/2)*Ks_x[1:]

        # A = diags([A_diag_bak, A_diag_c_all, A_diag_for], [-1, 0, 1], format = 'csc')

        # #### Create matrices (RHS)
        # C_diag_bak = (fir/4)*w_prof_mod[itxx,1:-1] \
                        # + (2 + fir*w_prof_mod[itxx,1:-1]**2)/12 \
                        # + (sec/2)*Ks_t[1:-1]

        # C_diag_for = -1*(fir/4)*w_prof_mod[itxx,1:-1] \
                        # + (2 + fir*w_prof_mod[itxx,1:-1]**2)/12 \
                        # + (sec/2)*Ks_t[1:-1]

        # C_diag_c_all = 1 - (fir/4)*w_prof_mod[itxx,1:] \
                            # + (fir/4)*w_prof_mod[itxx,0:-1] \
                            # - (2 + fir*w_prof_mod[itxx,0:-1]**2)/12 \
                            # - (2 + fir*w_prof_mod[itxx,1:]**2)/12 \
                            # - (sec/2)*Ks_t[0:-1] \
                            # - (sec/2)*Ks_t[1:]

        # C = diags([C_diag_bak, C_diag_c_all, C_diag_for], [-1, 0, 1], format = 'csc')

        # # Copy out C-profiles at t & t+1
        # b = np.copy(cf[itxx,:])
        # d = np.copy(cf[itxx+1,:])

        # # Compute erosion flux at t & t+1 and add to C profiles
        # if erosion_model == 'E1':
            # t_z = t_c_mod
            # if tau_bed[itxx] > t_z:
                # b[0] = b[0] + fir*e_0_mod*((tau_bed[itxx] - t_z))**b_c_mod
            # if tau_bed[itxx+1] > t_z:
                # d[0] = d[0] - fir*e_0_mod*((tau_bed[itxx+1] - t_z))**b_c_mod
                
        # elif erosion_model == 'E2':
            # t_z = t_c_mod + m_r_mod*(np.sum(cf[itxx,:])*del_z)**b_c_mod
            # if tau_bed[itxx] > t_z:
                # b[0] = b[0] + fir*e_0_mod*((tau_bed[itxx] - t_z))
            # if tau_bed[itxx+1] > t_z:
                # d[0] = d[0] - fir*e_0_mod*((tau_bed[itxx+1] - t_z))

        # # Implicit sparse matrix solve
        # F = C.dot(b) - d
        # cf[itxx+1,:] = spsolve(A, F)

    # return cf
    
# def obj_SS2(theta, erosion_model, model_spec):

    # # Initiate model domain
    # tg = model_spec[0]
    # mg = model_spec[1]
    # fg = model_spec[2]
    # del_t = np.diff(tg)[0]
    # del_z = np.diff(mg)[0]
    
    # cf = np.zeros((len(tg), len(mg)))
    # cf[0,:] = model_spec[3]
    
    # Ks_all = model_spec[4]
    # h_tot = model_spec[5]
    # tau_bed = model_spec[6]

    # # Set up model coefficients
    # fir = del_t / del_z
    # sec = del_t / del_z**2

    # # Adjust parameter scaling
    # w_s_mod = theta[0] / 10**3
    # e_0_mod = theta[1] / 10**4
    # t_c_mod = theta[2] / 10**2
    # if erosion_model == 'E2':
        # m_r_mod = theta[3] / 10**1
    # b_c_mod = theta[4]

    # # Set the settling velocity and deposition rules
    # w_prof_mod = np.zeros((len(tg), len(fg))) - w_s_mod
    # w_prof_mod[:,-1] = 0.0
    
    # # Loop through timesteps and calculate
    # for itxx, tstep in enumerate(tg[:-1]):
        
        # Ks_t = Ks_all[itxx]
        # Ks_x = Ks_all[itxx+1]
        
        # #### Create matrices
        # A_diag_bak = -1*(fir/4)*w_prof_mod[itxx+1,1:-1] \
                        # - (sec/2)*Ks_x[1:-1]

        # A_diag_for = (fir/4)*w_prof_mod[itxx+1,1:-1] \
                        # - (sec/2)*Ks_x[1:-1]

        # A_diag_c_all = 1 + (fir/4)*w_prof_mod[itxx+1,1:] \
                            # - (fir/4)*w_prof_mod[itxx+1,0:-1] \
                            # + (sec/2)*Ks_x[0:-1] \
                            # + (sec/2)*Ks_x[1:]

        # A_diag_c_all[0] = A_diag_c_all[0] \
                            # - 3*(fir/4)*w_prof_mod[itxx+1,0]
        
        # A_diag_for[0] = A_diag_for[0] \
                            # + 3*(fir/4)*w_prof_mod[itxx+1,0]
        
        # A_diag_for22 = np.zeros(len(A_diag_for)-1)
        # A_diag_for22[0] = - (fir/4)*w_prof_mod[itxx+1,0]
        
        # A = diags([A_diag_bak, A_diag_c_all, A_diag_for, A_diag_for22], [-1, 0, 1, 2], format = 'csc')

        # ####
        # C_diag_bak = (fir/4)*w_prof_mod[itxx,1:-1] \
                        # + (sec/2)*Ks_t[1:-1]

        # C_diag_for = -1*(fir/4)*w_prof_mod[itxx,1:-1] \
                        # + (sec/2)*Ks_t[1:-1]

        # C_diag_c_all = 1 - (fir/4)*w_prof_mod[itxx,1:] \
                            # + (fir/4)*w_prof_mod[itxx,0:-1] \
                            # - (sec/2)*Ks_t[0:-1] \
                            # - (sec/2)*Ks_t[1:]

        # C_diag_c_all[0] = C_diag_c_all[0] \
                            # + 3*(fir/4)*w_prof_mod[itxx,0]
    
        # C_diag_for[0] = C_diag_for[0] \
                            # - 3*(fir/4)*w_prof_mod[itxx,0]

        # C_diag_for22 = np.zeros(len(C_diag_for)-1)
        # C_diag_for22[0] = + (fir/4)*w_prof_mod[itxx,0]

        # C = diags([C_diag_bak, C_diag_c_all, C_diag_for, C_diag_for22], [-1, 0, 1, 2], format = 'csc')

        # # Copy out C-profiles at t & t+1
        # b = np.copy(cf[itxx,:])
        # d = np.copy(cf[itxx+1,:])

        # # Compute erosion flux at t & t+1 and add to C profiles
        # if erosion_model == 'E1':
            # t_z = t_c_mod
            # if tau_bed[itxx] > t_z:
                # b[0] = b[0] + fir*e_0_mod*((tau_bed[itxx] - t_z))**b_c_mod
            # if tau_bed[itxx+1] > t_z:
                # d[0] = d[0] - fir*e_0_mod*((tau_bed[itxx+1] - t_z))**b_c_mod
                
        # elif erosion_model == 'E2':
            # t_z = t_c_mod + m_r_mod*(np.sum(cf[itxx,:])*del_z)**b_c_mod
            # if tau_bed[itxx] > t_z:
                # b[0] = b[0] + fir*e_0_mod*((tau_bed[itxx] - t_z))
            # if tau_bed[itxx+1] > t_z:
                # d[0] = d[0] - fir*e_0_mod*((tau_bed[itxx+1] - t_z))

        # # Implicit sparse matrix solve
        # F = C.dot(b) - d
        # cf[itxx+1,:] = spsolve(A, F)

    # return cf

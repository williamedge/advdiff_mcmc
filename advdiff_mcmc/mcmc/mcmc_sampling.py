"""
# Functions for evaluating a 1D advection - diffusion model using MCMC

This version is directly associated with a JGR: JAMES journal submission from
late 2022. A more generic version of the fucntions is planned. 

In later versions the functions will be split into those relating to the model 
and those relating to MCMC sampling and evlauation. 

The core of these functions has been taken from an example on the Pymc3 website ("Using a “black box” likelihood function"). 

To do:
 - improve commenting
 - fix loglike mean function
"""

import numpy as np
from advdiff_mcmc.advdiff.fdm_implement import obj_CN
import pytensor.tensor as tt
    
def my_loglike_mean(theta, data, erosion_model, model_spec, obs_htx):
    
    model_out = obj(theta, erosion_model, model_spec)
    logppp = -np.sum(0.5*(np.log(2*np.pi*(theta[-1]/100)**2) + \
                            ((data - np.mean(model_out[:,obs_htx], axis=1))/(theta[-1]/100))**2))
    return logppp
    
def my_loglike(theta, data, erosion_model, model_spec, obs_htx):

    model_out = obj_CN(theta, erosion_model, model_spec)
    n = len(model_out[:,obs_htx].flatten())

    # Update of formula
    logppp = ((n/2)*np.log(2*np.pi))/2 - n*np.log(theta[-1]) - \
                np.sum(0.5*((1000*model_out[:,obs_htx] - 1000*data[:,obs_htx])**2)/theta[-1]**2)
    # logppp = ((n/2)*np.log(2*np.pi))/2 - n*np.log(theta[-1]**2) - \
    #             np.sum(0.5*((1000*model_out[:,obs_htx] - 1000*data[:,obs_htx])**2)/theta[-1]**2)
    return logppp


class LogLike(tt.Op):
    itypes = [tt.dvector]
    otypes = [tt.dscalar]

    def __init__(self, loglike, data, erosion_model, model_spec, obs_htx):
        self.likelihood = loglike
        self.data = data
        self.erosion_model = erosion_model
        self.model_spec = model_spec
        self.obs_htx = obs_htx

    def perform(self, node, inputs, outputs):
        (theta,) = inputs
        # call the log-likelihood function
        logl = self.likelihood(theta, self.data,\
                self.erosion_model, self.model_spec, self.obs_htx)
        outputs[0][0] = np.array(logl)
        


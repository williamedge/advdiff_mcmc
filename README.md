# advdiff_mcmc

Estimate parameters from a 1-D advection-diffusion equation (ADE) numerical model using MCMC sampling. 

Code has been cleaned up and commented since the original archived version. Numerical model functions used to setup the ADE are in 'advdiff'. MCMC sampling functions associated with *Pymc* are in 'mcmc/mcmc_sampling.py'. 

Testing contains several new notebooks summarising the use of the functions. 

To do:
 - Notebooks in 'case_studies' have not been updated to work with the improved code (yet). These will probably throw an error in their current state.
 - Most of the testing on the numerical model and some extra functionality is still buried in old PhD notebooks and needs to be cleaned and added to this package. 
 - Bilby could be made the default method. Still need to write some more code for better post processing.
 - Need to turn the standard plotting and posterior time series sampling code into functions. 
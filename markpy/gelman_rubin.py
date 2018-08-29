import numpy as np
import scipy

def gelman_reubin(chain):

    #function that will calculate the Gelman-rubin R statistic for each parameter. This is averaged over each chain so
    # the input paramter chain needs to be of dimension (niter, nchain, ndim) where niter is the mcmc step amount,
    # nchain is the nwalkers or number of chains, and ndim is the dimension of the chain or number of sampling params
    # Returns an array of dimension (ndim) that returns the GR R statistic for each sampling param or each mcmc dimension
    ndim = len(chain[0,0,:])
    nwalkers = len(chain[0,:,0])
    gelman_r = np.zeros([ndim])


    return gelman_r
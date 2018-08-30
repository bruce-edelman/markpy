import numpy as np

def gelman_rubin(chain, start=0, end=None):

    #function that will calculate the Gelman-rubin R statistic for each parameter. This is averaged over each chain so
    # the input paramter chain needs to be of dimension (niter, nchain, ndim) where niter is the mcmc step amount,
    # nchain is the nwalkers or number of chains, and ndim is the dimension of the chain or number of sampling params
    # Returns an array of dimension (ndim) that returns the GR R statistic for each sampling param or each mcmc dimension


    chains = np.array(chain)
    ndim, nwalkers, nlen = chain.shape
    gelman_r = np.zeros([ndim])

    if end is None:
        end = nlen-1

    chains = chains[start:end,:,:]

    for i in range(ndim):

        between = nlen/(nwalkers-1)*np.sum(np.mean(chains[:,:,i], axis=0) - np.mean(chain[:,:,i]))**2
        wihtin = np.sum((chain[:,:,i] - np.mean(chain[:,:,i],axis=0))**2 / (nlen - 1))/nwalkers

        var = (1/nlen)*between + (nlen-1)/nlen*wihtin
        post_var = var + between/nwalkers

        gelman_r[i] = np.sqrt(post_var/wihtin)

    return gelman_r


import numpy as np
import matplotlib.pyplot as plt

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


def geweke(chain, seg_len, ref_start, first_start=0, ref_end=None):

    # Function that will calculate the Geweke convergeence stattistic of the mcmc chain
    # input parameters are chain whcih is a SINGLE chain (numpy array 1-d) of dimension(niter),
    #  the start id of the reference segment used,
    # first_start, the first id to start calcuating geweke for the chain array defaults to beginning of array (first_start=0),
    #  and lastly the end of the reference segment, defaulted to None which will put the end of the reference segment
    # at the end of the array
    # this function returns the geweke statistic for the given chain in an array of dimension (nsegs) that has the z-score
    # and an array of the start id's for each segment that we calculated and an array of the end ids of each segment
    # all three arrays should be 1-D of length (nsegs)

    geweke_stats = []
    ends = []

    starts = np.arange(first_start, ref_end, seg_len)
    ref_segment = chain[ref_start:ref_end]

    for start in starts:
        seg = chain[start:int(start+seg_len)]
        geweke_stats.append((seg.mean()-ref_segment.mean())/np.sqrt(seg.var()+ref_segment.var()))

        ends.append(int(start+seg_len))

    return np.array(geweke_stats), np.array(starts), np.array(ends)


def plot_geweke(chain, seglen, ref_start, start=0, end=None):

    niter, ndim, nchains = chain.shape
    data = np.zeros([int(start+(niter-end)/seglen), ndim, nchains])
    ends = np.zeros([int(start+(niter-end)/seglen), ndim, nchains])
    starts = np.zeros([int(start+(niter-end)/seglen), ndim, nchains])
    fig, ax = plt.figure()
    for i in range(ndim):
        for j in range(nchains):
            data[:,i,j], starts[:,i,j], ends[:,i,j] = geweke(chain[:,i,j], seglen, ref_start, start, end)
            ax.plot(0.5*(starts[:,i,j]+ends[:,i,j]), data[:,i,j], 'r.')
    plt.xlabel('Iteration')
    plt.ylabel('Geweke statistic z-score')
    plt.title('Geweke Convergence Test - %s chains, %s dimensions' %(nchains, ndim))
    plt.savefig('Geweke_%schains_%sdimensions.png' %(nchains,ndim))
    plt.show()


def plot_gelman_rubin(chain):

    niter, ndim, nchains = chain.shape
    R = gelman_rubin(chain)
    plt.figure()
    plt.scatter(R, 'b.')
    plt.ylabel('Gelman-Rubin Statistic')
    plt.xlabel('chain dimension')
    plt.title('Gelman-Rubin Statistic for each dimension of the chain')
    plt.savefig('Gelman_Rubin_%sdimensions.png' % ndim)






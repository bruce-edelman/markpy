
# Copyright (C) 2018  Bruce Edelman
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""
markPy is a python package developed by Bruce Edelman to implement MCMC sampling among other things

"""

import numpy as np
import matplotlib.pyplot as plt

"""
This file contains some useful functions for computing tests for convergence on mcmc chains. It implements the 
Gelman-Rubin statistic as well as the geweke z-score and also a function for plotting a visualization of each
"""

def gelman_rubin(chain, start=0, end=None):
    """
    This is a function to calcuate the Gelman-rubin statistic for the mcmc chain for each sampling parameter.
    This statistic only makes sense when we are running our mcmc chain with multiple independent
    chains or walkers for each of the sampling parameters:
    Formulas for the tests given  at: https://pymc-devs.github.io/pymc/modelchecking.html
    :param chain: This is an attribute of the MarkChain object and should have shape (niter, nchains, ndim)
    :param start:  This is the starting index of where to start calculating, defaults to the beginnning
    :param end:  THis is the ending index of where to calculate the GR statistic from, defaults to the end of the chain
    :return: This returns an array of shape (ndim) that has the GR R stat for each of the sampling params or each
    mcmc dimension
    """

    # initialize some variables to use
    chains = np.array(chain)
    ndim, nwalkers, nlen = chain.shape
    gelman_r = np.zeros([ndim])

    # check if we put an end index, put it at end if we didn't
    if end is None:
        end = nlen-1

    # slice the parts off we don't want
    chains = chains[start:end,:,:]

    # loop through the sampling params
    for i in range(ndim):

        # calculate the between chain variance as shown in: https://pymc-devs.github.io/pymc/modelchecking.html
        between = nlen/(nwalkers-1)*np.sum(np.mean(chains[:,:,i], axis=0) - np.mean(chain[:,:,i]))**2

        #calculate the witin chain variance as shown in: https://pymc-devs.github.io/pymc/modelchecking.html
        wihtin = np.sum((chain[:,:,i] - np.mean(chain[:,:,i],axis=0))**2 / (nlen - 1))/nwalkers

        #calcuate the total and post variances to find R
        var = (1/nlen)*between + (nlen-1)/nlen*wihtin
        post_var = var + between/nwalkers

        # store the R statistic for each parameter
        gelman_r[i] = np.sqrt(post_var/wihtin)

    return gelman_r


def geweke(chain, seg_len, ref_start, first_start=0, ref_end=None):
    """
    This function calcualtes the Geweke convergence statisitic of a sinmgle mcmc chain of shape (niter, ndim)
    This calculates the geweke statistic for each individual mcmc chain. This calcuates the z-score based from:
    https://pymc-devs.github.io/pymc/modelchecking.html
    :param chain: single mcmc chain of samples so it has a shape of (niter, ndim)
    :param seg_len: the length of sengments to use in calculating geweke
    :param ref_start: the start index of the reference segment
    :param first_start: the first segments starting index, defualts to the beginning (0)
    :param ref_end: the end index of the reference segment, defaults to the end (None)
    :return: returns three numpy arrays each of length (nsegments) of however many segments we used in calculation
    the first array has the z-score for each segment, the second array is an array of starting indexes for each segment
    and lastly the third array is an array of ending indexes for the segments
    """

    # initialize the arrays
    geweke_stats = []
    ends = []

    # set up the array of starting indexes
    starts = np.arange(first_start, ref_end, seg_len)

    # set up the reference segment
    ref_segment = chain[ref_start:ref_end]

    # loop through the start of each segment
    for start in starts:
        # find the segment
        seg = chain[start:int(start+seg_len)]

        # find the z-score based from: https://pymc-devs.github.io/pymc/modelchecking.html
        # and append it to our array of z-scores
        geweke_stats.append((seg.mean()-ref_segment.mean())/np.sqrt(seg.var()+ref_segment.var()))

        # Add the end index to the list of ends
        ends.append(int(start+seg_len))

    #return all three numpy arrays
    return np.array(geweke_stats), np.array(starts), np.array(ends)


def plot_geweke(chain, seglen, ref_start, start=0, end=None):
    """
    This is a function to generate a quick visualization of convergence statistic geweke. This, if used, will be instead
    directly using the geweke function in this file. This function will take the attributes MarkChain.states = chains
    which has a shape of (niter, ndim, nchains). This function takes each individual chain ((ndim*nchains) of them ) and
    finds teh geweke for it and plots it on the y-axis at each segment
    :param chain: This is MarkChain.states so it has shape (niter, ndim, nchains)
    :param seglen: the len of segments to use in geweke
    :param ref_start: start index of reference segment
    :param start: start index of the first segment, defaults to the beginning (0)
    :param end: end index of the first segment, defaults to the end (None)
    :return: Function returns None, but will show the plot and save it as a .png file
    """

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
    return None

def plot_gelman_rubin(chain):
    """
    This is simalr function to plot_geweke, but only takes in a total MarkChain.states of shape (niter, ndim, nchains)
    and will plot the Gelman-R statistic for the mcmc chain to determin if it has converged
    :param chain: this is the MarkChain.sates object atttribute of shape (niter, ndim, nchains)
    :return: Function returns None, but will show the plot and save it as a .png file
    """

    # get variables
    niter, ndim, nchains = chain.shape

    # calcualte the GR R stat
    R = gelman_rubin(chain)

    # plot the data and save the fig
    plt.figure()
    plt.scatter(R, 'b.')
    plt.ylabel('Gelman-Rubin Statistic')
    plt.xlabel('chain dimension')
    plt.title('Gelman-Rubin Statistic for each dimension of the chain')
    plt.savefig('Gelman_Rubin_%sdimensions.png' % ndim)
    plt.show()
    return None





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

import corner
import numpy as np
import matplotlib.pyplot as plt

"""
This file provides some functions to use the corner python package to plot convenient looking plots
comparing the posterior samples distributions  
"""

def corner_plot(chain, params, filename):
    """
    this function plots the corner plot for visualizeing the chain. It does so by taking the average of all chains if
    multiple are given
    :param chain: this is the chain.states data structure of shape (niter, ndim, nchains)
    :param params: this is a list of sampling params present in the chain
    :param filename: this is a name of the file we want to save the plot to
    :return: returns nothing, just plots the figure and saves it as a .png file
    """

    # get shape of chain
    niter, ndim, nchains = chain.shape

    # initialize data for corner plots
    data = np.zeros([niter, ndim])

    # get the mean of each chain
    for i in range(ndim):
        for j in range(niter):
            data[j,i] = np.mean(chain[j,i,:])

    # generate the plot and save it
    figure = corner.corner(data, labels=params,quantities=(0.05,0.90), levels=(1-np.exp(-0.5),), show_titles=True,
                           title_kwargs = {"fontsize": 12})

    plt.suptitle('Corner Plot generated in markPy %s-d Sampling' % ndim)
    plt.savefig(filename)
    return None

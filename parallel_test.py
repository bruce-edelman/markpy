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
import markpy as mp
from itertools import cycle

COLOR_CYCLE = cycle('bgrcmk')

"""
This is a python file that tests the markpy sampler on a simple gaussian analytic model and tests the parallel version

"""

def main():

    dimension = 8
    # means = np.array(np.random.normal(0,1,dimension))
    # sigs = np.array(np.random.normal(0.8,0.05,dimension))

    means = np.full(dimension, 0)
    sigs = np.full(dimension, .08)
    stats = np.array([means, sigs])
    nwalkers = 6
    params = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']
    sigmaprop = 0.08
    norm_model = mp.NormModelAnalytic(params, sigs, means, prior_stats=stats)
    mc = mp.ParallelMarkChain(nwalkers,norm_model, dimension, sigmaprop)

    Nsteps = 100000
    mc.run(Nsteps)
    c = mc.get_burn_samps
    file = "Parallel_test_%swalkers.png" %nwalkers

    fig, axs = plt.subplots(dimension, sharex='col')
    for i in range(dimension):
        ax = axs[i]
        for j in range(nwalkers):
            ax.plot(c[:,:,j], c=next(COLOR_CYCLE), markersize=0.035, alpha=0.1)
        ax.set_ylabel(params[i])
    plt.suptitle("Parallel Mark Chain On Normal Analytic Model\n "
                    "%s walkers used" % nwalkers)
    plt.xlabel("Iteration")
    plt.savefig(file)
    plt.show()

    return None



if __name__ == "__main__":
    main()

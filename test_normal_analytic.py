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
import markpy


"""
This is a python file that tests the markpy sampler on a simple gaussian analytic model

"""

def main():

    dimension = 8
    means = np.array(np.random.normal(0,1,dimension))
    sigs = np.array(np.random.normal(0.8,0.05,dimension))

    means = np.full(dimension, 0)
    sigs = np.full(dimension, .08)
    stats = np.array([means, sigs])
    params = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']
    sigmaprop = 0.08
    norm_model = markpy.NormModelAnalytic(params, sigs, means, prior_stats=stats)
    mc = markpy.MarkChain(norm_model, dimension, sigmaprop)

    Nsteps = 40000
    mc.run(Nsteps)
    c = mc.get_burn_samps()
    chain = np.zeros([len(c[:,0]),len(c[0,:]),1])
    chain[:,:,0] = c
    #mp.corner_plot(chain, params, 'norm_analytic_%s-d.png' % dimension)
    file = 'geweke_test_normal_analytic_%s-d.png' % dimension
    markpy.plot_geweke(chain, 80, file)
    return None



if __name__ == "__main__":
    main()


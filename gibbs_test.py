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
import markpy


"""
This is a python file that tests the Gibbs sampler on a simple gaussian analytic model

"""

def main():

    dimension = 4
    A = np.random.rand(dimension, dimension)
    cov = A + A.T + dimension * np.eye(dimension)
    means = np.zeros([dimension])
    sigs = np.array([cov[0,0], cov[1,1], cov[2,2], cov[3,3]])
    stats = np.zeros([2, dimension])
    stats[0, :] = means
    stats[1, :] = sigs
    params = ['x1', 'x2', 'x3', 'x4']
    sigmaprop = 0.4
    norm_model = markpy.NormModelAnalytic(params, sigs, means, prior_stats=stats)
    mc = markpy.MarkChain(norm_model, dimension, sigmaprop, stepper=markpy.GibbsStepper)

    Nsteps = 40000
    c = mc.run(Nsteps, progress=True)

    chain = np.zeros([len(c[:,0]),len(c[0,:]),1])
    chain[:,:,0] = c
    markpy.corner_plot(chain, params, 'GibbsSampling_CornerPlot_%s-d.png' % dimension)
   # file = '~PycharmProjects/markPy/test_plots/geweke_test_normal_analytic_%s-d.png' % dimension
    #markpy.plot_geweke(chain, 80, file)
    return None



if __name__ == "__main__":
    main()


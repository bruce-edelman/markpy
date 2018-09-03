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
import scipy.stats as st

"""
This File sets up our Models to be used in our MarkChain Obejcts in sampler.py
We first set up a BaseModel class then setup the other models off of that one
"""

class BaseModel(object):
    """"
    Model Class will soon get combined with the Liklie classes. Want to create a BaseModel Class, then create the other
    models overtop that. (always require data? maybe?) other classes will be maybe class NormModel(Model): or
    class RosenbachModel(Model): for example.
    """

    def __init__(self, model, d, sig, D, samp_params, liklie, static_params=None, prior=1):
        """
        This is the initialization of the BaseModel class. This class will the be the base class that we generate our
        model (likliehood) objects out of. This model will require data. They will be subclasses of this Base Model
        Class to be used in our MarkChain objects.
        :param model: This is a function of the distribution we want to sample
        :param d: this is the observed data we inferring from
        :param sig: this is the sigma of our model #TODO: this may need switched to our NormModel(BaseModel): class
        :param D: this is the dimensionality of the problem or len(sampling_params)
        :param samp_params: this is a list of the names of each of the parameters we are sampling
        :param liklie: #TODO: THIS IS AN OUTDATED PARAMETER NEEDS REMOVED
        :param static_params: optional variable to add in static_params where we wont sample in (don't add
        to the dimensionality of our problem)
        :param prior: This is the prior to be used. #TODO: right now this will only work for the default uniform
        #TODO: prior = 1 for each parameter. i.e. Postulate of equal a-prior probabilities:
        #TODO: need to figure out a more elegant way of adding option of specifying the prior for each sampling parameter
        #TODO: individually
        """

        # Initialize our instance attributes of the BaseModel class
        self.data = d # data
        self.model = model # primary model using
        self.dim = D # dimension of model
        self.sig = sig # sigma of model
        self.prior = prior # prior fct (default to uniform prior=1)
        self.liklie = liklie #variable to store if we use the default liklie or other
        self.params = samp_params
        self.static = static_params

        # Error check to make sure len(self.param) is same asa self.dim
        if self.dim != len(self.params):
            print("ERROR: Dimension must be equal to the number of sampling parameters:")

    def get_posterior(self, samp, *args):
        if self.prior != 1:
            return self.liklie._get_posterior(samp, *args)*self.prior
        else:
            return self.liklie._get_posterior(samp, *args)

    def get_log_posterior(self, samp, *args):
        if self.prior != 1:
            return self.liklie._get_log_posterior(samp, *args) + self.get_log_prior()
        else:
            return  self.liklie._get_log_posterior(samp, *args)

    def get_name(self):
        return self.name

    @property
    def get_log_prior(self):
        return np.log(self.prior)


class LiklieBase(object):

    def __init__(self, model_func, data):
        self.func = model_func
        self.data = data

    def _residual(self, samp, *args):
        return (self.data-self.func(samp, *args))**2

    def lnprob(self):
        return

    def lnpost(self):
        return

    def get_name(self):
        return self.name


class LiklieNorm(LiklieBase):
    __model_name__ = 'Norm-Liklie'

    def __init__(self, model_func, sig, mean, data):
        self.sig = sig
        self.mean = mean
        super(LiklieNorm, self).__init__(model_func, data)

    def _get_posterior(self, samp, *args):
        return np.exp(-self.mean*(self._residual(samp, *args).sum()/self.sig**2))

    def _get_log_posterior(self, samp, *args):
        return np.log(np.exp(-self.mean*(self._residual(samp,*args).sum()/self.sig**2)))

    def get_name(self):
        return self.name
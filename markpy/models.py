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
    This is our Base Model Class that will be used via class inheritence to generate the other classes we use
    """
    name = "BaseModel"
    subtype = 'Base'
    def __init__(self, samp_params, static_params=None, logprior=None):
        """
        This is the initialization of the BaseModel class. This class will the be the base class that we generate our
        model (likliehood) objects out of. This model will not require data. The MarkChain object must take in a Model
        class with a valid get_posterior(self, samp, *args) method.
        ** AS OF NOW BASE MODEL DOES NOT HAVE THAT SO THE ONLY BUILT IN CLASSES THAT MarkChain objects CAN TAKE ARE
        listed here AFTER BASE MODEL **

        :param model_func: This is a function of the distribution we want to sample
        :param samp_params: this is a list of the names of each of the parameters we are sampling
        :param static_params: optional variable to add in static_params where we wont sample in (don't add
        to the dimensionality of our problem) This needs to be a dictionary with the names to be the name of each static
        parameter we use and the value of each to be the fixed value for that parameter
        :param logprior: This is the prior to be used. Needs to be either a single value which will be used for each
         of the sampling parameters or a list that must be length of ndim with the prior value for each parameter in the
         same order they appear in samp_params
        #TODO: prior = 1 (logprior=0) for each parameter. i.e. Postulate of equal a-prior probabilities:
        #TODO: need to figure out a more elegant way of adding option of specifying the prior for each sampling parameter
        #TODO: individually
        """

        # Initialize our instance attributes of the BaseModel class
        self.dim = len(samp_params) # dimension of model

        # If no prior is given we use the postulate of equal a prior probabilities
        if logprior is None:
            self.prior = [0]
        else:
            self.prior = [logprior] # prior fct (default to uniform prior=1)

        self.params = samp_params # this is our list that holds the names of the parameters we sample in

        # check to see if there are static params
        if static_params is None:
            self.static_params = {} # if none set it to an empty dict
        self.static_params = static_params
        # TODO: NEED TO IMPLEMENT ADDING THESE STATIC PARAMETERS INTO OUR CODE BASE FOR THE POSTERIOR CALCS

        # Error check to make sure that the prior given must either be a single value of a ndim-D array
        if len(self.prior) > 1 and len(self.prior) != self.dim:
            raise ValueError("prior must be single value of an array of length=ndim:")

    def get_log_posterior(self, samp, *args):
        """
        This is a base func that will be used from all child classes from Base Model and is imperative to be working
        if making your own BaseModel Child class to be used in a MarkChain object
        :param samp: This is the current point in our parameter space
        :param args: This is other args passed necessary for the model_func
        :return: this function returns the log posterior value at a given point in parameter space to be used in the
        Metropolis-Hastings Stepper

        WARNING: AS of now the get_lik and get_log_like methods in Base Model are only there for child inhertence and
        do NOT work on their own in the BaseModel parent class
        """
        return self.get_log_lik(samp, *args) + self.get_log_prior

    def get_posterior(self, samp, *args):
        """
        This is a base func that will be used from all child classes from Base Model and is imperative to be working
        if making your own BaseModel Child class to be used in a MarkChain object
        :param samp: This is the current point in our parameter space
        :param args: This is other args passed necessary for the model_func
        :return: this function returns the posterior value at a given point in parameter space to be used in the
        Metropolis-Hastings Stepper

        WARNING: AS of now the get_lik and get_log_like methods in Base Model are only there for child inhertence and
        do NOT work on their own in the BaseModel parent class
        """
        return self.get_lik(samp, *args) * np.exp(self.prior)

    def get_log_lik(self, samp, *args):
        """
        This is only here for child inheritence and does not work in an instance of this Parent Object
        :param samp: will take in the current sample, or point in parameter space
        :param args: again also takes the *args that may be needed for whatever model_func we gave
        :return: IN CHILD CLASSES ONLY: will return the log likliehood value

        WARNING: this method does NOT work in parent class Base Model
        """
        return None

    def get_lik(self, samp, *args):
        """
        This is only here for child inheritence and does not work in an instance of this Parent Object
        :param samp: will take in the current sample, or point in parameter space
        :param args: again also takes the *args that may be needed for whatever model_func we gave
        :return: IN CHILD CLASSES ONLY: will return the log likliehood value

        WARNING: this method does NOT work in parent class Base Model
        """
        return None

    @property
    def get_name(self):
        """
        property function to return the name of the model
        :return: returns the name of the model out of the possibilities listed in __models__ in __init__.py of markPy
        """
        return self.name

    @property
    def get_log_prior(self):
        """
        property function to return the log prior
        :return: returns the log prior as either a single value if we use it for each param or a list of length(ndim)
        that has the prior for each sampling param
        """
        return self.prior

    @property
    def get_sampling_params(self):
        """
        property function to return the sampling params for our model
        :return: this returns a list of strings with the name of each sampling parameter
        """
        return self.params

    @property
    def get_static_params(self):
        """
        property function to return the static params used in the model
        :return: returns the given dictionary in the __init__ method of Base Clasee or an empty dictionary if None was
        provided
        """
        return self.static_params

    @property
    def get_subtype(self):
        """
        property function that will return the subtype of the class the determine if it is a valid object for the
        model parameter in intializing a MarkChain object
        :return: returns a string with the name of the subtype.
        """
        return self.subtype

class BaseInferModel(BaseModel):
    """
    This is the Base class that will use data as inference. This is a child of the parent class BaseModel
    AS A WARNING: This is the only other class along with BaseModel with subtype  = 'Base' meaning that it is an
    invalid object as the model parameter in our MarkChain objects
    """
    name = "BaseInferModel"
    subtype = 'Base'

    def __init__(self, model_func, data, samp_params, **kwargs):
        """
        This is the initialization of the BaseInferModel class. This class will the be the base class that we generate
        our model (likliehood) objects that require data to infer out of. This model will not require data. The
        MarkChain object must take in a Model class with a valid get_posterior(self, samp, *args) method.
        ** AS OF NOW BASEINFERMODEL DOES NOT HAVE THAT SO THE ONLY BUILT IN CLASSES THAT MarkChain objects CAN TAKE ARE
        listed here AFTER BASEINFERMODEL **

        :param model_func: This is a function of the distribution we want to sample
        :param data: this is an array of data that was observed to be used for inference
        :param samp_params: this is a list of the names of each of the parameters we are sampling
        :param **kwargs: this is the possible passed stat_params dict and also logprior which will be defaulted
        as shown in the BaseModel class if not given
        """
        # Initialize our class instance variables
        self.data = data
        self.func = model_func

        # Inherit the rest of __init__ from the parent class (BaseModel) with samp_params and possible **kwargs
        super(BaseInferModel, self).__init__(samp_params, **kwargs)

    def _residual(self, samp, *args):
        """
        This is a function to use the given data to calcuate the residual to be used in any future model that
        requires data to infer
        :param samp: this is current sample or point in parameter space the chain is at
        :param args: these are other necessary arguments to pass for the model_func
        :return: this returns the residual of our model_func at the given point in parameter space
        """
        return (self.data-self.func(samp, *args))**2

    @property
    def get_data(self):
        """
        property function to retrieve the observed data that we infer from
        :return: returns a np.array of the observed data
        """
        return self.data

class NormModelInfer(BaseInferModel):
    """
    This is A model class that represents Normal Noise and a normal gaussian likliehood function for our observed
    data to infer from (most used model class probably)
    """
    name = "NormModelInfer"
    subtype = 'Likliehood'

    def __init__(self, sig, model_func, data, samp_params, **kwargs):
        """
        This is the initialization function of the child class NormModelInfer of Parent BaseInferModel and grandparent
        BaseMdodel. this class requires one additional parameter than the BaseInferModel whihc is the sigma (std dev) of
        the assumed normal gaussian noise
        :param sig: std dev of assumed gaussian noise
        :param model_func: This is a function of the distribution we want to sample
        :param data: this is an array of data that was observed to be used for inference
        :param samp_params: this is a list of the names of each of the parameters we are sampling
        :param **kwargs: this is the possible passed stat_params dict and also logprior which will be defaulted
        as shown in the BaseModel class if not given
        """
        # Initialize our class instance variables
        self.sig = sig

        # Inherit the rest of __init__ from the parent class (BaseInferModel) with model_func,
        # data, samp_params and possible **kwargs
        super(NormModelInfer, self).__init__(model_func, data, samp_params,**kwargs)

    def get_log_lik(self, samp, *args):
        return  np.log(np.exp(-0.5*(self._residual(samp,*args).sum()/self.sig**2)))

    def get_lik(self, samp, *args):
        return np.exp(-0.5*(self._residual(samp,*args).sum()/self.sig**2))

class NormModelAnalytic(BaseModel):
    name = "NormModelAnalytic"

    def __init__(self,samp_params, sig=None, mean=None, **kwargs):
        if sig is None:
            sig = [1.]*self.dim
        self.sig = sig
        if mean is None:
            mean = [0.]*self.dim
        self.mean = mean
        super(NormModelAnalytic, self).__init__(samp_params,**kwargs)
        if len(self.mean) != self.dim or len(self.sig) != self.dim:
            raise ValueError("Dimension of sig and mean must match len(samp_params)")
        self.distribution = st.multivariate_normal(mean=mean, cov=sig)

    def get_log_lik(self, samp, *args):
        return self.distribution.logpdf(samp)

    def get_lik(self, samp, *args):
        return self.distribution.pdf(samp)

class EggBoxAnalytic(BaseModel):
    name = "EggBoxAnalytic"

    def __init__(self, samp_params, **kwargs):
        super(EggBoxAnalytic, self).__init__(samp_params, **kwargs)

    def get_lik(self, samp, *args):
        return np.exp((2 + np.prod(np.cos(samp)))**5)

    def get_log_lik(self, samp, *args):
        return (2 + np.prod(np.cos(samp)))**5

class RosenbrockAnalytic(BaseModel):
    name = "RosenbrockAnalyitc"

    def __init__(self, samp_parms, **kwargs):
        super(RosenbrockAnalytic, self).__init__(samp_parms, **kwargs)

    def get_log_lik(self, samp, *args):
        log_like = 0
        for i in range(len(samp)-1):
            log_like -= ((1-samp[i])**2 + 100 *(samp[i+1]-samp[i]**2)**2)
        return log_like

    def get_lik(self, samp, *args):
        return np.exp(self.get_log_lik(samp, *args))




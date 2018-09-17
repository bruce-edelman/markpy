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


"""
This file Sets up the stepper classes to be used in our samplers for various types of decides
"""

class BaseStepper(object):
    """
    This is our base stepper class that will be used via inheritence in creating our other steppers in this file
    """

    name = 'BaseStepper'
    subtype = 'Base'

    def __init__(self, model, dim, priorrange, **kwargs):
        """
        #TODO: Once we add more than one stepper we need to update our base class to include the things we can in
        #TODO: common between all our child steppers
        :param model: model object passed to this stepper object through intialization within the MarkChain objects
        :param dim: dimension of the model
        :param: priorrange the acceptable range of the prior
        :param kwargs:
        :return:
        """
        self.dim = dim
        self.model = model
        self.priorrange = priorrange

    def proposal(self, samp):
        """
        #TODO: Once we add more than one stepper we need to update our base class to include the things we can in
        #TODO: common between all our child steppers
        :param samp:
        :return: this returns our proposed newsamp in which we decide to accept or not later
        """
        return samp

    def decide(self, newsamp, oldsamp, *args):
        """
        This decide function must take in these things for sure:
        #TODO: Once we add more than one stepper we need to update our base class to include the things we can in
        #TODO: common between all our child steppers
        :param newsamp: the proposed sample calculated from the proposal method
        :param oldsamp: the old sampl of where the chain is at before
        :param args:  these are necessary args to pass to the self.model.get_posterior(samp, *args) methods help in the
        model objects
        :return: This must always return two things, a bool acc of whether the proposal was accepted or not, and
        samp which is the newsamp if acc==True and oldsamp if acc==False
        """
        acc = np.random.choice([True,False], p=[0.5,0.5])
        if acc:
            return acc, newsamp
        elif not acc:
            return acc, oldsamp

class MetropolisHastings(BaseStepper):
    """
    This is a child class from parent BaseStepper that is a stepper to be used in the MarkChain objects that performs
    a class generic MetropolisHastings Algorithm for our Markov Chain decides
    """

    name = 'MetropolisHastings'
    subtype = 'stepper'

    def __init__(self, sigma, model, dim, priorrange, **kwargs):
        """
        Initialization function for the Metropolis Hastings stepper object that is a child of parent class BaseStepper
        :param sigma: sigma used for the normal proposal fct used in M-H algorithm
        :param args:
        :param kwargs:
        """

        self.sigmaprop = sigma
        super(MetropolisHastings, self).__init__(model, dim, priorrange, **kwargs)


    def proposal(self, samp):
        """
        Function that finds the proposed sample that we later evaulate if we want to accpet it or not:
        for the Metropolis-Hastings stepper we just use a gaussian distributed random walk proposal
        :param samp: This is the oldsamp of where our chain is
        :return:  this returns the proposed sample
        """
        return samp+np.random.normal(0., self.sigmaprop, self.dim)

    def decide(self, newsamp, oldsamp, *args):
        """
        This is the decide function that will calculate the if we accept the proposed sample or not and return the next
        sample to save in our markov chain
        :param newsamp: the proposed sample calculated from the proposal method
        :param oldsamp: the old sampl of where the chain is at before
        :param args:  these are necessary args to pass to the self.model.get_posterior(samp, *args) methods help in the
        model objects
        :return: This must always return two things, a bool acc of whether the proposal was accepted or not, and
        samp which is the newsamp if acc==True and oldsamp if acc==False
        """
        # function calculates the hastings ratio and decides to accept or reject proposed step
        # Check to make sure the proposed sample is still in the allowed range according to the priorrange array
        if not ((np.array([p1 - p2 for p1, p2 in zip(newsamp, np.transpose(self.priorrange)[:][0])]) > 0).all()
                and (np.array([p2 - p1 for p1, p2 in zip(newsamp, np.transpose(self.priorrange)[:][1])]) > 0).all()):
            # if newsamp is not in the allowed prior range we set accept to False and return the oldsamp instead of the
            # newsamp
            acc = False
            return acc, oldsamp

        # now we know we are accepting with at least some prob so we need probabilities calculated through our Model
        # class and some the functions we created it with
        newp = self.model.get_posterior(newsamp, *args)
        oldp = self.model.get_posterior(oldsamp, *args)

        if newp >= oldp:
            # if new step is more prob than old we accept immediately and return the new samp
            acc = True
            return acc, newsamp
        else:
            # oldp is bigger so we only accept the step probablistically as in MH algo
            prob = newp[0] / oldp[0]
            acc = np.random.choice([True, False], p=[prob, 1. - prob])
            return acc, acc * newsamp + (1. - acc) * oldsamp


class GibbsStepper(MetropolisHastings):
    """
    This is a classs that is resonsible for  the stepper that will implplement Gibbs Sampling for MarkPy
    """
    name = 'GibbsStepper'
    subtype = 'stepper'

    def __init__(self, sigma, model, dim, priorrange, **kwargs):
        """
        This is same as in The metroppolis-hastings stepper but does a simple error check to make sure the problem is
        multi-variate
        :param sigma: this sigma for the gaussain proposal
        :param model: the model used (Model Object)
        :param dim: dimension of problem
        :param priorrange: priorrange passed from MarkChain
        :param kwargs: other args needed later (addded in Base maybe)
        """

        # Check to make sure it is multi-dimensional
        if dim < 2:
            raise ValueError("Problem must be multivariate to implement Gibbs Sampling")

        # inherit rest from MHstepper and BaseStepper
        super(GibbsStepper, self).__init__(sigma, model, dim, priorrange, **kwargs)

    def proposal(self, samp, *args):
        """
        This is the overridden proposal function for the Gibbs sampler (uses the same decide function)
        :param samp: sample we are at (point in parameter space)
        :param args: args needed to be passed to our model function (and decide function)
        :return: returns the proposed next step after performing the individual parameter sampling here which
        is the core of Gibbs Sampling
        """
        # Initialize
        prop_samp = samp

        # loop through each parameter
        for i in range(self.dim):
            # propose with np.random.normal around the given parameter we sample from
            prop_samp[i] = samp[i] + np.random.normal(0., self.sigmaprop)
            # decide wheter to accept or not then go to next sample
            acc, prop_samp = self.decide(prop_samp, samp, *args)

        # return the sampled vector
        return prop_samp




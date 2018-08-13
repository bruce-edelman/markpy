
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
markPy is a python package developed my Bruce Edelman to implement MCMC sampling among other things

"""

import numpy as np



class MarkChain(object):
    '''
    MarkChain is an object of the markov chain we are sampling. all parameters stored in MarkChain.states
    takes in a an object of the Model class as PDF, d is the dimension of our model, ranges for all parameters and
    proposed step sigma for our normally distributed randomwalk

    '''
    def __init__(self, PDF, d, priorrange, sigprop):
        # initialize the first sample in correct data structure
        self.oldsamp = np.array([np.random.uniform(priorrange[i][0],priorrange[i][1]) for i in range(d)])
        self.acc = 0 #store how many accepted samples for AR
        self.dim = d #dimension sampling
        self.sigmaprop = sigprop
        self.model= PDF #Model object created for specific problem
        self.priorrange = priorrange
        self.states = np.array([[self.oldsamp]]) #initialize our output

    def step(self, *kargs):
        # this function performs the Metropolis-Hastings Algorthim stepping

        newsamp = self.proposedStep() #first we propose where we step next
        acc, newsamp = self.hastingsRatio(newsamp, *kargs) # compute the ratio and decide if we accept

        self.states = np.append(self.states, [[newsamp]], axis=1) #add new value to our chain
        if acc: #if we accepted a new value update for our AR
            self.acc += 1
        self.oldsamp = newsamp #rese oldsamp variable for next iteration

    def proposedStep(self):
        #random walk proposed step with a normal distribution
        return self.oldsamp+np.random.normal(0.,self.sigmaprop,self.dim)

    def AcceptenceRatio(self):
        #function return the acceptence ratio of the chain
        return 1.*self.acc/self.N

    def run(self, N, *kargs):
        #function called to run the chain
        self.N = N
        for i in range(N):
            self.step(*kargs)

    def hastingsRatio(self, newsamp, *kargs):
        #function calculates the hastings ratio and decides to accept or reject proposed step
        if not ((np.array([p1-p2 for p1,p2 in zip(newsamp, np.transpose(self.priorrange)[:][0])])>0).all()
                and (np.array([p2-p1 for p1,p2 in zip(newsamp, np.transpose(self.priorrange)[:][1])])>0).all()):
            #This is the rejection criteria
            acc = False
            return acc, self.oldsamp #return false for acc and the old samp

        #now we know we are accepting with at least some prob so we need probabilities
        newp = self.model.get_posterior(newsamp, *kargs)
        oldp = self.model.get_posterior(self.oldsamp, *kargs)

        if newp >= oldp:
            #if new step is more prob than old we accept immediately and return the new samp
            acc = True
            return acc, newsamp
        else:
            # oldp is bigger so we only accept the step probablistically as in MH algo
            prob = newp/oldp
            acc = np.random.choice([True, False], p=[prob,1.-prob])
            return acc, acc*newsamp + (1. - acc)*self.oldsamp


class Model(object):
    """"
    Model class is an object that stores information about the model we choose to sample
    takes in three different fcts, model, res, lik and prior to initilialize. These fcts can be grabbed from markpy or
    programmed in when using markpy
    """
    def __init__(self, model, d, sig, D, res, lik, prior=1):
        self.data = d # data
        self.model = model # primary model using
        self.dim = D # dimension of model
        self.sig = sig # sigma of model
        self.res = res # fct to calcuate the residual
        self.lik = lik # likliehood fct
        self.prior = prior # prior fct (default to uniform prior=1)

    def get_posterior(self, samp, *kargs):
        #function that returns the posterior for the model (used in sampling)
        resid = self.res(self.data, self.model, samp, *kargs) #calc the resid with given fct
        likliehood = self.lik(self.sig, 0.5, resid) # calc the likliehood
        return self.prior*likliehood


def res_norm(data, model, samp, *kargs):
    return (data - model(samp, *kargs))**2


def liklie_norm(sig, mean, res):
    return np.exp(-mean*(res.sum()/sig**2))



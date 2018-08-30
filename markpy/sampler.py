
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


class MarkChain(object):
    '''
    MarkChain is an object of the markov chain we are sampling. all parameters stored in MarkChain.states
    takes in a an object of the Model class as PDF, d is the dimension of our model, ranges for all parameters and
    proposed step sigma for our normally distributed randomwalk

    '''

    name = 'Metropolis-Hastings Chain'

    def __init__(self, PDF, d, priorrange, sigprop):
        # initialize the first sample in correct data structure
        self.oldsamp = np.array([np.random.uniform(priorrange[i][0],priorrange[i][1]) for i in range(d)])
        self.acc = 0 #store how many accepted samples for AR
        self.dim = d #dimension sampling
        self.sigmaprop = sigprop
        self.model= PDF #Model object created for specific problem
        self.priorrange = priorrange
        self.states =  [self.oldsamp] #initialize our output



    def step(self, *kargs):
        # this function performs the Metropolis-Hastings Algorthim stepping

        newsamp = self.proposedStep() #first we propose where we step next
        acc, newsamp = self.hastingsRatio(newsamp, *kargs) # compute the ratio and decide if we accept

        self.states = np.append(self.states, [newsamp], axis=0) #add new value to our chain
        if acc: #if we accepted a new value update for our AR
            self.acc += 1
        self.oldsamp = newsamp #rese oldsamp variable for next iteration

    def proposedStep(self):
        #random walk proposed step with a normal distribution
        return self.oldsamp+np.random.normal(0.,self.sigmaprop,self.dim)

    @property
    def currentState(self):
        #function returns current sates of the chain
        return self.states

    @property
    def currentSamp(self):
        #function to return the current sample of the chain
        return self.oldsamp

    @property
    def chain_dimension(self):
        #function to return the dimensionality of the chain
        return self.dim

    @property
    def chain_parameters(self):
        #function returns a list of the parameters of the chain
        return self.model.params

    @property
    def AcceptenceRatio(self):
        #function return the acceptence ratio of the chain
        return 1.*self.acc/self.N

    @property
    def is_burned_in(self):
        id, isb = self.burnin_nacl()
        return isb

    @property
    def burn_idx(self):
        id, isb = self.burnin_nacl()
        return id

    def run(self, N, *kargs):
        #function called to run the chain, takes in N numbers of steps to run and *kargs which depend on The model we set up
        self.N = N
        for i in range(N):
            self.step(*kargs)
        return None

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

    def get_acl(self):
        #function to get the autocorrelation length of the markov chain
        acls = {} #initialize acls as a dict

        for i in range(self.dim): #loop through all params
            acl = compute_acl(self.states[:,i]) #compute for each param
            if np.isinf(acl.any()): # if its inf then return the len of the chain
                acl = len(self.states[:,i])
            acls[self.model.params[i]] = acl
        return acls #returns a dictionary with key values of names the params and values if acl for each param

    def burnin_nacl(self, nacls=10):
        # function to check if the chain is burned in via the nacls route
        #default to nacls to 10. burned in if it has been thorugh max(acls)*nacls iterations in the chain. THis is always same for all params

        act = self.get_act()
        burn_idx = nacls*act
        is_burned_in = burn_idx < len(self.states[:,0]) #check if burned in
        #returns abn int with index of where we burn in at and bool if if we are burnt in or not with current input chain
        return int(burn_idx), is_burned_in

    def get_burn_samps(self):
        # this function gets returns the states of the chain that are after the given burn_idx (cuts off the not burned in part)
        idx, isburn = self.burnin_nacl()
        if isburn:
            #return only burned in states
            print(len(self.states[:,0]))
            print(len(self.states[idx:,0]))
            print(len(self.states[:,0]) - idx)
            return self.states[idx:,:]
        else:
            #if not burned in print statement and return None
            print("CHAIN NOT BURNED IN - burnin")
            return self.states[0,:]

    def get_independent_samps(self):
        #function that returns the independent samples of the chain
        #first get the correlation length from the other classmethod
        corrleng = self.get_corrlen()
        burnid, isburned = self.burnin_nacl() #now get burned in to make sure we add the two
        if isburned: #if we are burned in return the correct samps
            corrleng = max(corrleng, burnid)
            print(len(self.states[:,0]))
            print(len(self.states[corrleng:,0]))
            print(len(self.states[:,0]) - corrleng)
            return self.states[corrleng:,:]
        else: # otherwise print statement to tell user we are not burned in and return None
            print("CHIAN NOT BURNED IN - corrlen")
            return self.states[0,:]

    def get_effective_AR(self):
        #function that uses the number of current independent samples to return the effective acceptence ratio of the chain
        ind_samps = self.get_independent_samps()
        return 1.*len(ind_samps)/self.N

    def plot_acls(self):
        #function to plot acls (mainly used in testing)
        acls = self.get_acl()
        fig, axs = plt.subplots(nrows=3, ncols=1, sharex='col')
        for i in range(self.dim):
            ax = axs[i]
            ax.plot(acls[self.model.params[i]])
        plt.show()

    def get_corrlen(self):
        #function that gets the correlation lenght to be used in returning independent samples
        acls = self.get_acl()
        ct = 0
        cl = np.zeros([self.dim])
        for p in range(self.dim): #find when acl drops to less than 0.08 for each param
            for i in acls[self.model.params[p]]:
                if i < 0.1:
                    cl[p] = ct
                else:
                    ct += 1
        #now we take the longest length of each param corrlength and return that
        return int(np.max(cl))

    def get_name(self):
        return self.name

    def get_act(self):
        return 2.0/self.AcceptenceRatio-1.0

def compute_acl(samps):
    #global function that computes the auto-correlation length from an array of sampkles
    #returns and array of the acl for each iteration
    m = samps-np.mean(samps)
    f = np.fft.fft(m)
    p = np.array([np.real(v)**2+np.imag(v)**2 for v in f])
    pi = np.fft.ifft(p)
    return np.real(pi)[:int(len(samps)/2)]/np.sum(m**2)


class Model(object):
    """"
    Model class is an object that stores information about the model we choose to sample
    takes in three different fcts, model, res, lik and prior to initilialize. These fcts can be grabbed from markpy or
    programmed in when using markpy
    """
    name = 'Base-Model'

    def __init__(self, model, d, sig, D, samp_params, liklie, static_params=None, prior=1):
        #this Model class has parameters:
        # model - the model function of the problem we want to sample
        # d is the data we are inferring from
        # sig is the sigma of the model,
        # D is the dimension of the model
        # prior is set to uniform (prior =1) but we can adjust this if wantegitd
        self.data = d # data
        self.model = model # primary model using
        self.dim = D # dimension of model
        self.sig = sig # sigma of model
        self.prior = prior # prior fct (default to uniform prior=1)
        self.liklie = liklie #variable to store if we use the default liklie or other
        self.params = samp_params
        self.static = static_params
        if self.dim != len(self.params):
            print("ERROR: Dimension must be equal to the number of sampling parameters:")

    def get_posterior(self, samp, *kargs):
        return self.liklie._get_posterior(samp, *kargs)

    def get_log_posterior(self, samp, *kargs):
        return self.liklie._get_log_posterior(samp, *kargs)

    def get_name(self):
        return self.name


class Liklie_Base(object):
    name = 'Base-Liklie'

    def __init__(self, model_func, data):
        self.func = model_func
        self.data = data

    def _residual(self, samp, *kargs):
        return (self.data-self.func(samp, *kargs))**2

    def lnprob(self):
        return

    def lnpost(self):
        return

    def get_name(self):
        return self.name


class Liklie_Norm(Liklie_Base):
    name = 'Norm-Liklie'

    def __init__(self, model_func, sig, mean, data):
        self.sig = sig
        self.mean = mean
        super(Liklie_Norm, self).__init__(model_func, data)

    def _get_posterior(self, samp, *kargs):
        return np.exp(-self.mean*(self._residual(samp, *kargs).sum()/self.sig**2))

    def _get_log_posterior(self, samp, *kargs):
        return np.log(np.exp(-self.mean*(self._residual(samp,*kargs).sum()/self.sig**2)))

    def get_name(self):
        return self.name


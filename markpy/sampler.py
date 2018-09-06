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
This file Sets up the MarkChain object, used as our sampler in markpy
"""


class MarkChain(object):
    """
    MarkChain is an object of the markov chain we are sampling. all parameters stored in MarkChain.states
    takes in a an object of the Model class as PDF, d is the dimension of our model, ranges for all parameters and
    proposed step sigma for our normally distributed randomwalk
    """

    name = 'Metropolis-Hastings Chain'

    def __init__(self, Model, d, sigprop, priorrange=None):
        """
        When a MarkChain object is called we pass these parameters. This object is the main sampler for the mcmc and
        contains many properties and methods useful to use:
        :param Model: This needs to be an instance of the Model class (listed in models.py and __init__.py)
         which has a method get_posterior, and also stores the sampling parameter names
        :param d: this is the dimensionality of our chain
        :param priorrange: this is a numpy array of shape (ndim,2) that gives the min/max value of the allowed
        range for each of the sampling parameters
        :param sigprop: this is the std-dev of the proposal step
        """
        # this needs to be an object of class Model
        self.model = Model

        # initialize the first sample in correct data structure
        if priorrange is None and self.model.prior_stats is None:
            raise ValueError("ERROR: If no prior-range given, Model object must have method get_prior_stats.")

        if priorrange is None:
            self.oldsamp = np.random.normal(self.model.prior_stats[0], self.model.prior_stats[1])
            self.priorrange = np.full((d,2), np.inf)
            self.priorrange[:,0] *= -1
        else:
            self.priorrange = priorrange
            self.oldsamp = np.array([np.random.uniform(self.priorrange[i][0],self.priorrange[i][1]) for i in range(d)])

        # this variable will store the number of accepted samples
        self.acc = 0
        self.dim = d # dimension of our sampling
        self.sigmaprop = sigprop

        # initlize our chain to be an array of array with the outer array being of shape (niter) (one right now) and
        # the inner array is of shape (ndim)
        self.states = [self.oldsamp] # initialize our output

    def step_mh(self, *args):
        """
        This function performs the Metropolis-Hastings algorithm step.
        :param args: This are other args that are passed to the Model function used in the HastingsRatio func
        :return: This function does not return anything just updates the chain attribute of states and acc
        """

        # here we propose the next step using our proposal method
        newsamp = self.proposedStep()

        # now we use the hastingsRatio function to decide if we accept this proposed step and update samps
        acc, newsamp = self.hastingsRatio(newsamp, *args)

        # now we append the states with newsamp, if it was accepted newsamp is different than oldsamp, if not
        # it is the same as self.oldsamp
        self.states = np.append(self.states, [newsamp], axis=0) # add new value to our chain

        if acc: # if we accepted a new value update for our AR
            self.acc += 1
        self.oldsamp = newsamp # reset oldsamp variable for next iteration

    def proposedStep(self):
        """
        This function is our proposal function for determining where to step next. There is no parameters to this
        :return: This returns a newsamp which is normally distributed with mean 0 and sigma= sigmaprop around the previous
        sample. sample is an array of shape (ndim)
        """

        # random walk proposed step with a normal distribution
        return self.oldsamp+np.random.normal(0., self.sigmaprop, self.dim)

    @property
    def currentState(self):
        """
        This is a property function of class MarkChain to get the current state
        :return: return the attribute self.states which has shape (ndim, niter_current) where niter_current
        is the number of iterations we have been through in the chain at this time
        """
        # function returns current sates of the chain
        return self.states

    @property
    def currentSamp(self):
        """
        This is a property function of class MarkChain to get the last sample used in the mcmc
        :return: this returns the attribute self.oldsamp which is the last sample used in sampling
        """
        # function to return the current sample of the chain
        return self.oldsamp

    @property
    def chain_dimension(self):
        """
        this is property function of class MarkChain to get the dimension used of the chain
        :return:  this returns and int of the dimension of the chain
        """
        # function to return the dimensionality of the chain
        return self.dim

    @property
    def chain_parameters(self):
        """
        This is a property function that calls an attribute from the class Model which is an attribute in the class
        MarkChain: Will add a similar property function the Model class for clarity
        :return: this returns a list of the sampling parameter names used in the chain
        """
        # function returns a list of the parameters of the chain
        return self.model.params

    @property
    def AcceptenceRatio(self):
        """
        this is a property function of class MarkChain that will return the current acceptence ratio of the chain
        :return: this returns a float number that is the acceptence ratio for the chain
        """
        # function return the acceptence ratio of the chain
        return 1.*self.acc/self.N

    @property
    def is_burned_in(self):
        """
        this is a property function that checks to see if the chain is currently burned in or not
        :return: this returns a bool that tells us if the chain is currently burned in. Thiis checks the burnin based on
        the nacl burnin routine later defined in this class methods.
        """
        __, isb = self.burnin_nact()
        return isb

    @property
    def burn_idx(self):
        """
        this is a property function that returns the index value where the chain becomes burned in if it is burned in
        and the last index if it is not burned in yet
        :return:
        """
        # check to see if it is burned in
        if self.is_burned_in:
            # if it is return the burn idx as expected
            idx, __ = self.burnin_nact()
            return idx
        else:
            # if it is not burned in let the user know and return the last index of the chain
            print("Chain is not burned in: Run for more iterations")
            return len(self.states[:,0])

    def run(self, n, *args):
        """
        This function is responsible for actually runnning the chain for however many steps:
        :param n: This is how many interations we run the chain for
        :param args: this is needed to pass for the PDF model Class later in stepping forward
        :return: this returns nothing
        """
        # function called to run the chain, takes in N numbers of steps to run and
        # *args which depend on The model we set up
        # TODO: maybe figure a way to move this self.N def into __init__ but that will require some reworking of structure
        self.N = n

        # now we use the step_mh method to step the chain forward N steps
        # TODO: maybe add a submodule that holds classes of steppers (MH, KDE, etc) and there we can add different types
        # TODO: of moves rather than only the metropolis hastings stepper
        for i in range(n):
            self.step_mh(*args)
        return None

    def hastingsRatio(self, newsamp, *args):
        """
        This function calcualtes the hastings ratio of our proposed new sample and decides whether to accept that sample
        or reject it
        :param newsamp: this is an array of shape (ndim) that is the proposed step we are deciding to accept or not
        :param args: these args are needed to calculate the Model Class get_posterior fcts
        :return: this returns a bool acc that tells us if we accepted the sample or not and also the sample that we
        chose to accept or the oldsample if we rejected the new one
        """
        # function calculates the hastings ratio and decides to accept or reject proposed step
        # Check to make sure the proposed sample is still in the allowed range according to the priorrange array
        if not ((np.array([p1-p2 for p1, p2 in zip(newsamp, np.transpose(self.priorrange)[:][0])]) > 0).all()
                and (np.array([p2-p1 for p1, p2 in zip(newsamp, np.transpose(self.priorrange)[:][1])]) > 0).all()):
            # if newsamp is not in the allowed prior range we set accept to False and return the oldsamp instead of the
            # newsamp
            acc = False
            return acc, self.oldsamp

        # now we know we are accepting with at least some prob so we need probabilities calculated through our Model
        # class and some the functions we created it with
        newp = self.model.get_posterior(newsamp, *args)
        oldp = self.model.get_posterior(self.oldsamp, *args)

        if newp >= oldp:
            # if new step is more prob than old we accept immediately and return the new samp
            acc = True
            return acc, newsamp
        else:
            # oldp is bigger so we only accept the step probablistically as in MH algo
            prob = newp[0]/oldp[0]
            acc = np.random.choice([True, False], p=[prob, 1.-prob])
            return acc, acc*newsamp + (1. - acc)*self.oldsamp

    def get_acl(self):
        """
        This function gets the auto-correlation length for each of the parameters as a function of interation
        :return: this returns a dictionary with key values as the names of each parameter and the value the
        acl as function of iteration for that parameter
        """
        # function to get the autocorrelation length of the markov chain
        acls = {} # initialize acls as a dict
        # TODO: when using acl instead of act to get independent samples the 90% CI seems off (possible bug???)
        # loop through each parameter
        for i in range(self.dim):
            # use the compute_acl function for each param
            acl = compute_acl(self.states[:, i]) # compute for each param
            if np.isinf(acl.any()):
                # if its inf then return the len of the chain
                acl = len(self.states[:,i])
            acls[self.model.params[i]] = acl
        return acls # returns a dictionary with key values of names the params and values if acl for each param

    def burnin_nacl(self, nacls=10):
        """
        This function evalutes if the chain is burned in yet via the nacl method (burned in if max acl of param* nacl <
        len(chain):
        :param nacls: defaults to 10 but this is how many max acls of iterations before we say its burned in at
        :return: and integer for the index where the chain burns in and a bool if the chain is currently burned in or
        not
        """
        # function to check if the chain is burned in via the nacls route
        acl = self.get_acl()  # get acls
        max_acl = 0

        # find the max acl for all the params
        for i in range(self.dim):
            if max_acl < np.max(acl[self.model.params[i]]):
                max_acl = np.max(acl[self.model.params[i]])
        burn_idx = nacls * max_acl

        # check if burned in
        is_burned_in = burn_idx < len(self.states[:, 0])
        # returns abn int with index of where we burn in at and bool if if we are burnt in or not with current
        # input chain
        return int(burn_idx), is_burned_in

    def burnin_nact(self, nacts=10):
        """
        This function evalutes if the chain is burned in yet via the nact method (burned in if estimated act*nacts <
        len(chain):
        :param nacts: defaults to 10 but this is how many max acts of iterations before we say its burned in at
        :return:and integer for the index where the chain burns in and a bool if the chain is currently burned in or
        not
        """
        # function to check if the chain is burned in via the nacts route
        act = self.get_act()
        burn_idx = nacts*act
        # check if burned in
        is_burned_in = burn_idx < len(self.states[:,0])
        # returns abn int with index of where we burn in at and bool if if we are burnt in or not with current
        # input chain
        return int(burn_idx), is_burned_in

    def get_burn_samps(self):
        """
        This function returns the bunred in samples of the states in the chain
        As of right now it uses the burnin_nact() method of evaluating burnin but also the bunrin_nacl() method is
        available and want to add more in the future
        :return: this returns the sliced self.states with the not burned in part sliced off the beginning
        """
        # this function gets returns the states of the chain that are after the given burn_idx
        # (cuts off the not burned in part)
        # check if we are burned in
        idx, isburn = self.burnin_nact()
        if isburn:
            #return only burned in states
            return self.states[idx:,:]
        else:
            #if not burned in print statement and return None
            raise ValueError("CHAIN NOT BURNED IN - burnin")


    def get_independent_samps(self):
        """
        This function returns the independent samples by computing first the correlation length of the chain
        then evaluating the burnin. It will slice off the max of (burnid, corrlength) to return only the independent
        samples. Prints error statemtent if the chain is not yet burned in
        :return:
        """
        # function that returns the independent samples of the chain
        # first get the correlation length from the other classmethod
        corrleng = self.get_corrlen()

        # now get burned in to make sure we add the two
        burnid, isburned = self.burnin_nacl()

        # if we are burned in return the correct samps
        if isburned:
            # take the max of burnid, and corrlenght and slice it off
            corrleng = max(corrleng, burnid)
            return self.states[corrleng:, :]
        else:
            # otherwise print statement to tell user we are not burned in and return None
            raise ValueError("CHIAN NOT BURNED IN - corrlen")


    def get_effective_AR(self):
        """
        This function used the independent samples to get the effective acceptence ratio. This is defined as
        eff_AR = len(ind_samps)/niter
        :return: this returns the effective AR (float) from the samples. Only works if the chain is burned in
        """
        # function that uses the number of current independent samples to return the effective
        # acceptence ratio of the chain
        # get the independent samps
        ind_samps = self.get_independent_samps()

        # check if burned in
        if ind_samps is None:
            # if we are not burned in only return the regular acceptence ratio not the effective
            print("CHAIN NOT BURNED IN: RETURNING REGULAR ACCEPTENCE RATIO")
            return self.AcceptenceRatio()
        return 1.*len(ind_samps)/self.N

    def plot_acls(self):
        """
        this function is used in testing to plot the acls for each parameter
        TODO:  This needs polishing and prettying up the plot before release
        TODO: This needs changed to be independent of our dimensionality we choose for our chain
        :return: this returns nothing but will show the matplotlib plot of the acls for each param
        """
        # function to plot acls (mainly used in testing)
        # get the acls
        acls = self.get_acl()

        # plot one for each of the dimensions (this may need fixed once we start doing other dimensions
        fig, axs = plt.subplots(self.dim, sharex='col')
        for i in range(self.dim):
            ax = axs[i]
            ax.plot(acls[self.model.params[i]])
        plt.show()
        return None

    def get_corrlen(self):
        """
        This is a method function for computing the correlation length of the chain to be used in returning the
        independent samples
        :return: this returns and integer that is the longest correlation legnth for each of the parameters
        """
        # function that gets the correlation lenght to be used in returning independent samples
        acls = self.get_acl()

        # intialize ct to 0
        ct = 0
        # initialize the cl data structure
        cl = np.zeros([self.dim])

        # find when the acl drops to less than 0.1 for each parameter.
        # the maximum correlation length of each of the parameters gets returned as the chain correlation length

        for p in range(self.dim):
            for i in acls[self.model.params[p]]:
                if i < 0.1:
                    cl[p] = ct
                else:
                    ct += 1
        # now we take the longest length of each param corrlength and return that
        return int(np.max(cl)) # needs to be an integer since it will be used as an index

    def get_name(self):
        """
        function that returns the name of the class
        :return: returns the name of the class (' Metropolis-Hastings Chain ")
        This feature will be used later on when we start adding different kinds of steppers / chains
        """
        return self.name

    def get_act(self):
        """
        This function provides an estimate of the auto correlation time of the chain.
        It is estimated as 2/AR - 1 = act
        :return: returns the act for the chain based of its current accpetence ratio
        """
        return 2.0/self.AcceptenceRatio-1.0


def compute_acl(samps):
    """
    Gloabal function used for computing the acl from a given array of data
    :param samps: this is an array of data
    :return: returns the acl for that array of data
    """

    # global function that computes the auto-correlation length from an array of sampkles
    # returns an array of the acl for each iteration
    m = samps - np.mean(samps)
    f = np.fft.fft(m)
    p = np.array([np.real(v) ** 2 + np.imag(v) ** 2 for v in f])
    pi = np.fft.ifft(p)

    return np.real(pi)[:int(len(samps) / 2)] / np.sum(m ** 2)


class ParallelMarkChain(object):
    """
    This is a Wrapper class used to construct multiple chains and parralellize the total mcmc by constructing mutliple
    independent markov chains
    """
    name = 'ParallelMarkChain'

    def __init__(self, nchains, PDF, d, sigprop, priorrange=None):
            """
            This is the intialization function of the parallel markchain wrapper class the run mutlple mcmc chains
            :param PDF: This needs to be an instance of the Model class which has a method get_posterior, and also
            stores the sampling parameter names
            :param d: this is the dimensionality of our chain
            :param priorrange: this is a numpy array of shape (ndim,2) that gives the min/max value of the allowed
            range for each of the sampling parameters
            :param sigprop: this is the std-dev of the proposal step
            """
            self.states = None
            self.nchains = nchains
            self.chains = []*self.nchains
            self.dim = d
            for i in range(self.nchains):
                self.chains[i] = MarkChain(PDF, d, sigprop, priorrange)

    def run(self, n, *args):
        """
        This function is responsible for actually runnning all the chains at same time for however many steps:
        :param n: This is how many interations we run the chain for
        :param args: this is needed to pass for the PDF model Class later in stepping forward
        :return: this returns nothing
        """

        # now we use the step_mh method to step the chain forward N steps
        # TODO: maybe add a submodule that holds classes of steppers (MH, KDE, etc) and there we can add different types
        # TODO: of moves rather than only the metropolis hastings stepper

        # loop through the number of steps
        for step in range(n):
            # for each step take one step for each chain
            for i in self.chains:
                # take the metropolis-hastings step with this chain
                i.step_mh(*args)
        self.states = np.zeros(len(self.chains[0].states[:, 0]), self.dim, self.nchains)
        return None

    @property
    def get_states(self):
        """
        propertyt function to return the states of all chains used in our parallel markchain object
        :return: returns a numpy array states of shape (niter, ndim, nchains)
        """

        # Error check to make sure we ran the chain before generating states
        if self.states is None:
            raise ValueError("ParallelMarkChain has not been ran yet:")

        # get the states out of each chain we have
        for i in range(self.nchains):
            self.states[:,:,i] = self.chains[i].states

        # return the states
        return self.states

    @property
    def get_burn_samps(self):
        """
        This is a property fct that will return the burned in states for each independent chain we use
        to keep each chain the same size we take the max burn id from each chain and cut that off from each chain
        when returning the burned in samps
        :return: this returns the burned in samps of shape (niter-max(burnids), ndim, nchains)
        """

        # Intialize list to store burn idxs for each chain
        burnids = []

        # loop through each chain
        for i in self.chains:
            # if a single chain is not burned in we raise an error and tell user to run the chains for longer
            if not i.is_burned_in():
                raise ValueError("ERROR: at least one independent chain is not burned in. Run chains for longer")

            # add in the nect chains burn idx to the list
            burnids.append(i.burn_idx())

        # return the states with max(burnids) samples sliced off
        return self.states[max(burnids):, :, :]
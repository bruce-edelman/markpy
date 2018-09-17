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
from .pbar import *
from .steppers import *
import numpy as np
import matplotlib.pyplot as plt
import time, datetime


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

    def __init__(self, Model, d, sigprop, stepper=None, priorrange=None, initial_state=None):
        """
        When a MarkChain object is called we pass these parameters. This object is the main sampler for the mcmc and
        contains many properties and methods useful to use:
        :param Model: This needs to be an instance of the Model class (listed in models.py and __init__.py)
         which has a method get_posterior, and also stores the sampling parameter names
        :param d: this is the dimensionality of our chain
        :param sigprop: this is the std-dev of the proposal step
        :param stepper: This is an instance of a stepper class listed in steppers.py and __init__.py
        This must not have a subtype='base" just as the Model parameter
        :param priorrange: this is a numpy array of shape (ndim,2) that gives the min/max value of the allowed
        range for each of the sampling parameters
        :param initial_state: Optional starting input parameter where we can give the initial state of the chain
        must be an ndim array with the initial parameter value in order of the samp_params list in Model Object
        """

        # this needs to be an object of class Model and must have subtype != 'base'
        if Model.subtype == 'Base':
            raise TypeError("Model Parameter in MarkChain Must be a Model Class created in models.py that does not have"
                            "subtype of Base")
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
        self.accepted = []
        self.acc = 0
        self.dim = d # dimension of our sampling
        self.N = None

        # check if initial state is given:
        if initial_state is not None:
            if len(initial_state) != d:
                raise IndexError("Length of initial_state must equal dimension of problem")
            else:
                self.oldsamp = [initial_state]

        # This needs to be an object isntance of a stepper class and must have subtype != 'Base'
        if stepper is None:
            stepper = MetropolisHastings
        if stepper.subtype == 'Base':
            raise TypeError("Stepper Parameter in MarkChain Must be a Stepper Class created in steppers.py that does not"
                            "have subtype of Base")
        self.stepper = stepper(sigprop, self.model, self.dim, self.priorrange)

        # attribute that stores what number chain this is in a ParallelMarkObject
        # defaults to None and is None if the MarkChain is being operated alone
        self.number = None

    def step(self, n, pbar, thin=None, progress=None, *args):
        """
        This is the step generator that will return the iterable samples of the markov chain as it evolves

        #TODO: FIGURE OUT ALL BUGS HERE FOR SURE
        WANRNING: THIS IS A GENERATOR AND JUST GOT CHANGED. MAY BE OTHER BUGS WITH THE STATES AND ACCESSIGN THEM

        :param n: this is the number of iterations to run for
        :param thin: this is a parameter that defaults to None, if we set it to a value then it will thin the mcmc by
        that ratio, if None is given no thinning
        :param progress:  defaults to False, if True will display  progress bar. Progress bar created as in:
        github.com/dfm/emcee/emcee/pbar.py
        :param pbar: SAME DOC AS IN RUN METHOD
        :param args: This are other args that are passed to the Model function used in the HastingsRatio func
        :return: This returns an iterable (since its a generator) should return an array that is shape=ndim
        """

        # Check if we set thinning up or not
        if thin is not None:
            thin = int(thin)
            # error check to make sure thin is not negative or 0
            if thin <= 0:
                raise ValueError("Thin must be strictly positive:")
            intermediate_step = thin
        else:
            # if no thin set int_step=1
            intermediate_step = 1
        # set the total iterations for pbar
        total = n * intermediate_step
        newsamp = self.oldsamp
        # setup the progress bar
        if self.number is None: # check if this is being used in ParallelMark Chain object
            with progress_bar(progress, total) as pbar:
                # loop through iterations
                for _ in range(n):
                    # loop through our thinning procedure if necessary
                    for _ in range(intermediate_step):

                        # Use the stepper object to perform whatever kind of step we need
                        newsamp = self.stepper.proposal(self.oldsamp)
                        acc, newsamp = self.stepper.decide(newsamp, self.oldsamp, *args)

                        if acc:  # if we accepted a new value update for our AR
                            self.acc += 1
                        self.oldsamp = newsamp  # reset oldsamp variable for next iteration
                        # update the pbar
                        pbar.update(1)

                    # now we return that samp from the generator
                    yield np.array(newsamp), acc
        else: # if it is being used dont create its own pbar, use the inherited one
            # loop through iterations
            for _ in range(n):
                # loop through our thinning procedure if necessary
                for _ in range(intermediate_step):
                    # Use the stepper object to perform whatever kind of step we need
                    newsamp = self.stepper.proposal(self.oldsamp)
                    acc, newsamp = self.stepper.decide(newsamp, self.oldsamp, *args)
                    if acc:  # if we accepted a new value update for our AR
                        self.acc += 1
                    self.oldsamp = newsamp  # reset oldsamp variable for next iteration
                    # update that inherited pbar
                    pbar.update(1)
                # now we return that samp from the generator
                yield np.array(newsamp), acc

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
    def AcceptenceArray(self):
        """
        This is a property function that returns the acceptence array as an array of shape (niterations)
        This stores the bool of each iteration step if it was accepted or not
        :return: numpy array of shape and detail listed above
        """
        return np.array(self.accepted)


    def is_burned_in(self, samps):
        """
        this is a property function that checks to see if the chain is currently burned in or not
        :param samps: This takes in samps for a chain. must be shape (niter, ndim)
        :return: this returns a bool that tells us if the chain is currently burned in. Thiis checks the burnin based on
        the nacl burnin routine later defined in this class methods.
        """
        __, isb = self.burnin_nact(samps)
        return isb


    def burn_idx(self, samps):
        """
        this is a property function that returns the index value where the chain becomes burned in if it is burned in
        and the last index if it is not burned in yet
        :param samps: This takes in samps for a chain. must be shape (niter, ndim)
        :return: This function returns the index of when the chain is burned in
        """
        # check to see if it is burned in
        if self.is_burned_in:
            # if it is return the burn idx as expected
            idx, __ = self.burnin_nact(samps)
            return idx
        else:
            # if it is not burned in let the user know and return the last index of the chain
            print("Chain is not burned in: Run for more iterations")
            pass

    def run(self, n, thin=None, progress=False, pbar=None, *args):
        """
        This function is responsible for actually running the chain for however many steps:
        :param n: This is how many iterations we run the chain for
        :param thin: this is a parameter that defaults to None, if we set it to a value then it will thin the mcmc by
        that ratio, if None is given no thinning
        :param progress: defaults to False, if True will display  progress bar. Progress bar created as in:
        github.com/dfm/emcee/emcee/pbar.py
        :param pbar: if using the parallel markchain it will pass a progress bar object from tqdm here to get sent to step fct
        :param args: this is needed to pass for the PDF model Class later in stepping forward
        :returns: this returns an array of the samples of shape (samp, ndim)
        """
        # function called to run the chain, takes in N numbers of steps to run and
        # *args which depend on The model we set up
        # TODO: maybe figure a way to move this self.N def into __init__ but that will require some reworking of structure
        self.N = n
        # initialize variables for generating the states
        samps = np.zeros([n, self.dim])
        ct = 0
        # use our step generator to get out the results
        for results, acc in self.step(n,pbar, thin, progress, *args):
            samps[ct,:] = results
            self.accepted.append(acc)
            ct += 1
        return samps


    def get_acl(self, samps):
        """
        This function gets the auto-correlation length for each of the parameters as a function of interation
        :param samps: This takes in samps for a chain. must be shape (niter, ndim)
        :return: this returns a dictionary with key values as the names of each parameter and the value the
        acl as function of iteration for that parameter
        """
        # function to get the autocorrelation length of the markov chain
        acls = {} # initialize acls as a dict
        # TODO: when using acl instead of act to get independent samples the 90% CI seems off (possible bug???)
        # loop through each parameter
        for i in range(self.dim):
            # use the compute_acl function for each param
            acl = compute_acl(samps[:, i]) # compute for each param
            if np.isinf(acl.any()):
                # if its inf then return the len of the chain
                acl = len(samps[:,i])
            acls[self.model.params[i]] = acl
        return acls # returns a dictionary with key values of names the params and values if acl for each param

    def burnin_nacl(self,samps, nacls=10):
        """
        This function evalutes if the chain is burned in yet via the nacl method (burned in if max acl of param* nacl <
        len(chain):
        :param samps: This takes in samps for a chain. must be shape (niter, ndim)
        :param nacls: defaults to 10 but this is how many max acls of iterations before we say its burned in at
        :return: and integer for the index where the chain burns in and a bool if the chain is currently burned in or
        not
        """
        # function to check if the chain is burned in via the nacls route
        acl = self.get_acl(samps)  # get acls
        max_acl = 0

        # find the max acl for all the params
        for i in range(self.dim):
            if max_acl < np.max(acl[self.model.params[i]]):
                max_acl = np.max(acl[self.model.params[i]])
        burn_idx = nacls * max_acl

        # check if burned in
        is_burned_in = burn_idx < len(samps[:, 0])
        # returns abn int with index of where we burn in at and bool if if we are burnt in or not with current
        # input chain
        return int(burn_idx), is_burned_in

    def burnin_nact(self,samps, nacts=10):
        """
        This function evalutes if the chain is burned in yet via the nact method (burned in if estimated act*nacts <
        len(chain):
        :param samps: This takes in samps for a chain. must be shape (niter, ndim)
        :param nacts: defaults to 10 but this is how many max acts of iterations before we say its burned in at
        :return:and integer for the index where the chain burns in and a bool if the chain is currently burned in or
        not
        """
        # function to check if the chain is burned in via the nacts route
        act = self.get_act()
        burn_idx = nacts*act
        # check if burned in
        is_burned_in = burn_idx < len(samps[:,0])
        # returns abn int with index of where we burn in at and bool if if we are burnt in or not with current
        # input chain
        return int(burn_idx), is_burned_in

    def get_burn_samps(self, samps):
        """
        This function returns the bunred in samples of the states in the chain
        As of right now it uses the burnin_nact() method of evaluating burnin but also the bunrin_nacl() method is
        available and want to add more in the future
        :param samps: This takes in samps for a chain. must be shape (niter, ndim)
        :return: this returns the sliced self.states with the not burned in part sliced off the beginning
        """
        # this function gets returns the states of the chain that are after the given burn_idx
        # (cuts off the not burned in part)
        # check if we are burned in
        idx, isburn = self.burnin_nact(samps)
        if isburn:
            #return only burned in states
            return samps[idx:,:]
        else:
            #if not burned in print statement and return None
            raise ValueError("CHAIN NOT BURNED IN - burnin")


    def get_independent_samps(self, samps):
        """
        This function returns the independent samples by computing first the correlation length of the chain
        then evaluating the burnin. It will slice off the max of (burnid, corrlength) to return only the independent
        samples. Prints error statemtent if the chain is not yet burned in
        :param samps: This takes in samps for a chain. must be shape (niter, ndim)
        :return: This returns the independent samps that are remaining in samps
        """
        # function that returns the independent samples of the chain
        # first get the correlation length from the other classmethod
        corrleng = self.get_corrlen(samps)

        # now get burned in to make sure we add the two
        burnid, isburned = self.burnin_nacl(samps)

        # if we are burned in return the correct samps
        if isburned:
            # take the max of burnid, and corrlenght and slice it off
            corrleng = max(corrleng, burnid)
            return samps[corrleng:, :]
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

        return 1.*len(ind_samps)/self.N

    def plot_acls(self, samps):
        """
        this function is used in testing to plot the acls for each parameter
        TODO:  This needs polishing and prettying up the plot before release
        TODO: This needs changed to be independent of our dimensionality we choose for our chain
        :param samps: This takes in samps for a chain. must be shape (niter, ndim)
        :return: this returns nothing but will show the matplotlib plot of the acls for each param
        """
        # function to plot acls (mainly used in testing)
        # get the acls
        acls = self.get_acl(samps)

        # plot one for each of the dimensions (this may need fixed once we start doing other dimensions
        fig, axs = plt.subplots(self.dim, sharex='col')
        for i in range(self.dim):
            ax = axs[i]
            ax.plot(acls[self.model.params[i]])
        plt.show()
        return None

    def get_corrlen(self, samps):
        """
        This is a method function for computing the correlation length of the chain to be used in returning the
        independent samples
        :param samps: This takes in samps for a chain. must be shape (niter, ndim)
        :return: this returns and integer that is the longest correlation legnth for each of the parameters
        """
        # function that gets the correlation lenght to be used in returning independent samples
        acls = self.get_acl(samps)

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

    def __init__(self, nchains, PDF, d, sigprop, priorrange=None, initial_states=None):
            """
            This is the intialization function of the parallel markchain wrapper class the run mutlple mcmc chains
            :param PDF: This needs to be an instance of the Model class which has a method get_posterior, and also
            stores the sampling parameter names
            :param d: this is the dimensionality of our chain
            :param sigprop: this is the std-dev of the proposal step
            :param priorrange: this is a numpy array of shape (ndim,2) that gives the min/max value of the allowed
            range for each of the sampling parameters
            :param initial_states: This is optional if we want to start with an initial states given . Must be of shape
            (ndim, nchains)

            """

            self.nchains = nchains
            self.chains = []
            self.dim = d
            # Create an array of MarkChain objects and set each to have an ordered number attribute
            for i in range(self.nchains):
                if initial_states is not None:
                    chain = MarkChain(PDF, d, sigprop, priorrange, initial_states[:,i])
                else:
                    chain = MarkChain(PDF,d,sigprop, priorrange)
                chain.number = i
                self.chains.append(chain)

    def run(self, n, thin=None, progress=False, burn=False, ind=False, mean=False, verbose=False, *args):
        """
        This function is responsible for actually running the chain for however many steps:
        :param n: This is how many iterations we run the chain for
        :param thin: this is a parameter that defaults to None, if we set it to a value then it will thin the mcmc by
        that ratio, if None is given no thinning
        :param progress: defaults to False, if True will display  progress bar. Progress bar created as in:
        github.com/dfm/emcee/emcee/pbar.py
        :param burn: bool to decide to return burn samps or just reg sampls defaults to reg
        :param ind: bool to decide to return ind. samps or just reg. defaults to reg burn has priority over ind if both
        are set to True
        :param mean: defaults to False, If True will avearge over nchains for samples
        :param verbose: if set to true will give more stats on the mcmc as it finishes
        :param args: this is needed to pass for the PDF model Class later in stepping forward
        :returns: this returns an array of the samples of shape (samp, ndim, nchains)
        """
        # intialize data structure
        parallel_samps = np.zeros([n, self.dim, self.nchains])

        # time calc for verbose
        start_time = time.clock()

        #thinning procedure check
        total = n*self.nchains
        if thin is not None:
            if thin <= 0:
                raise ValueError("thin value must be strictly positive")
            total *= thin
        out = None
        chain_st_time = datetime.datetime.now()
        # setup progress bar
        with progress_bar(progress, total) as pbar:
            # run each chain for given amount and store the results
            for i in self.chains:
                parallel_samps[:,:,i.number] = i.run(n, thin, progress, pbar, *args)
                if burn:
                    for j in self.chains:
                        burn_samps = j.get_burn_samps(parallel_samps[:,:,j.number])
                        out = np.zeros([len(burn_samps),self.dim, self.nchains])
                        out[:,:,i.number] = burn_samps
                elif ind:
                    for k in self.chains:
                        ind_samps = k.get_independent_samps(parallel_samps[:,:,k.number])
                        out = np.zeros([len(ind_samps), self.dim, self.nchains])
                        out[:,:,k.number] = ind_samps
                else:
                    out = parallel_samps

        if verbose:
            # these are the verbose outputs for extra information about runtime
            end_time = datetime.datetime.now()
            time_elapsed = round(time.clock() - start_time, 2)
            print("START TIME: %s \nEND TIME: %s \nTIME ELAPSED: %.2f sec" %(chain_st_time, end_time, time_elapsed))
            print("Average Step / sec : %.3f" % float(total/time_elapsed))
        if mean:
            # check if we want mean samps or all of the samps
            return self.get_mean_states(out)
        else:
            return out

    def get_mean_states(self, samps):
        """
            function to return the states of the mean of the chains used in our parallel markchain object
            :param samps: arrays of data of shape (niter, nchains, ndim)
            :return: returns a numpy array states of shape (niter, ndim)
        """

        niter, nchains, ndim = samps.shape
        # initialize the means array
        means = np.zeros([niter, ndim])


        # get the means out
        for i in range(ndim):
            for j in range(niter):
                means[j,i] = np.mean(samps[j,:,i])

        # return the means
        return means

    @property
    def get_chain_acceptence_ratios(self):
        """
        This is a property function that will return an array of the acceptence ratios for each of the chains
        :return: this returns an array of length = nchains that has the acceptence ratio for each chain
        """
        AR = []
        for i in self.chains:
            if not i.is_burned_in:
                raise ValueError("ERROR: at least one independent chain is not burned in. Run chains for longer")
            AR.append(i.AcceptenceRatio)
        return np.array(AR)

    @property
    def get_chain_effective_acceptence_ratios(self):
        """
        This is a property function that will return an array of the effective acceptence ratios for each of the chains
        :return: this returns an array of length = nchains that has the effective acceptence ratio for each chain
        """
        eAR = []
        for i in self.chains:
            if not i.is_burned_in:
                raise ValueError("ERROR: at least one independent chain is not burned in. Run chains for longer")
            eAR.append(i.get_effective_AR)
        return np.array(eAR)

    @property
    def AcceptenceFraction(self):
        """
        NEEDS COMMENTING
        :return:
        """
        for i in self.chains:
            if not i.is_burned_in:
                raise ValueError("ERROR: at least one independent chain is not burned in. Run chains for longer")

        iterations = len(self.chains[0].accepted)
        a = np.zeros([iterations])
        for ct in range(iterations):
            temp = 0
            for i in self.chains:
                temp += i.accepted[ct]
            a[ct] = temp / self.nchains
        return a





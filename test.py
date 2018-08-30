import numpy as np
import matplotlib.pyplot as plt
import markpy.sampler as mp

"""
This is a python file that tests the markpy sampler on some simple problems one is 3-dimesnional and one is 6-dimensional

"""


def get_data():
    """
    function that generates 20 data points for three parameters(3 dim), A, f, phi, and on an x-axis of time
    takes no input parameters:
    This function simulates generating the observed data that the mcmc infers off of
    :return: returns two np.arrays, one for time t, and one is the data we normally jittered around a simple cosine signal
    """
    # actual values for A, f and phi
    A = 2.
    f = 4. * 2. * np.pi/20
    phi = 1.

    # can set a random seed here to get deterministic resutls
    # np.random.seed(10)

    # jitter around the actual signal with n
    n = np.random.normal(0., 1, 20)
    t = np.arange(20)

    # return time and the necessary data
    return t, A*np.cos(f*t+phi) + n


def get_data_two():
    """
        function that generates 30 data points for three parameters(6 dim), A, f, phi, A2, f2, phi2 and on an x-axis of time
        takes no input parameters:
        This function simulates generating the observed data that the mcmc infers off of
        :return: returns two np.arrays, one for time t, and one is the data we normally jittered around a simple cosine signal
        """

    # actual values for A, f, phi, A2, f2, and phi2
    A1 = 2.
    A2 = 3.4
    f1 = 4. * 2. * np.pi/30
    f2 = 3.6 * 1.3 *np.pi/30
    phi1 = 1.
    phi2 = 2.7

    # can set a random seed here to get deterministic resutls
    # np.random.seed(10)

    # jitter around the actual signal with n
    n = np.random.normal(0., 1, 30)
    t = np.arange(30)

    # return time and the necessary data
    return t, A1*np.cos(f1*t+phi1)+A2*np.sin(f2*t+phi2)+n


def model_exp(samp, data, t):
    """
    This is the function that will evaluate the posterior of our model. We setup this function that takes in a sample
    and time values and will return the signal value at those values
    :param samp: this is a 3 dimensional array of the params, A, f, phi
    :param data: this is not used here but we need to pass it to mass class structure in sampler.py (possible could fix
    in the future)
    :param t: the array that saves the values of times where we observe data
    :return: returns the posterior signal value at this state
    """
    return samp[0]*np.cos(samp[1]*t+samp[2])


def model_two(samp, data, t):
    """
    This is the function that will evaluate the posterior of our model. We setup this function that takes in a sample
    and time values and will return the signal value at those values
    :param samp: this is a 6 dimensional array of the params, A, f, phi, A2, f2, phi2
    :param data: this is not used here but we need to pass it to mass class structure in sampler.py (possible could fix
    in the future)
    :param t: the array that saves the values of times where we observe data
    :return: returns the posterior signal value at this state
    """
    return samp[0]*np.cos(samp[1]*t+samp[2])+samp[3]*np.sin(samp[4]*t+samp[5])


def main():
    """
    Main function that controls the testing script
    :return: has no return
    """

    # optionally can choose a seed for deterministic results
    # np.random.seed(10)

    # intialize some parameters about the mcmc
    Nsteps = 10000 # iterations to use in the chain
    sigprop = 0.11 # the std dev sigma to use when proposing the new normally distributed Metropolis-Hastings Step
    sig = 1 # prior std deviation
    D = 6 # dimensionality of the problem
    t, data = get_data_two() # get the data from on of the built in functions
    mean = 0.5 # mean of the prior

    # list to store the names of each of the problem parameters that we sample in
    params = ['Amp', 'Freq', 'Phase', 'Amp2', 'Freq2', 'Phase2']

    # range of allowed values for all the sampling parameters listed in params
    priorrange = np.array([[0,5],[0,np.pi],[0,np.pi],[0,5],[0,np.pi],[0,np.pi]])

    # Here we setup the Model/Liklie class for our problem with given parameters set above
    liklie = mp.Liklie_Norm(model_two, mean, sig, data)
    test_model = mp.Model(model_two, data, sig, D, params, liklie)

    # now using the setup model we can create the chain using the mp.MarkChain class
    mc = mp.MarkChain(test_model, D, priorrange, sigprop)

    # now we run the chain for Nsteps iterations
    mc.run(Nsteps, data, t)

    # now we can interpret results with some plotting functions
    plot_chains(mc.states) # plots the chain for each parameter (only one chain until paralleization)

    # here we get the burned in samps using the mc.get_burn_samps() method
    samps = mc.get_burn_samps()

    # now we plot the burned in chain for each parameters
    plot_chains(samps)

    # Lastly we plot the signal witht he 90% credible interval from our MCMC chain
    plot_signal(mc, t, data)


def plot_chains(chain):
    """
    This function will plot the chain for each parameter (this is currently setup for the 6-dim problem and without
    parallelization. We will need to update it to the mean of all chaines for each parameter once new feature added)

    :param chain: This function takes in a parameter that is chain which is a (niter, ndim) size array that stores
    the chains for each  sampling parameters
    :return: This function does not return anything but will show the plot via matplotlib (does not save plot)
    """

    # Plotting statements
    # TODO fix up the parallelization here and make the plot as pretty as possible
    fig, axs = plt.subplots(nrows=3, ncols=2, sharex='col')
    ax = axs[0,0]
    ax.plot(chain[:,0], 'r')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Amplitude")
    ax = axs[1,0]
    ax.plot(chain[:,1], 'b')
    ax.set_ylabel('Frequency')
    ax = axs[2,0]
    ax.plot(chain[:,2], 'g')
    ax.set_ylabel('Phase')
    ax = axs[0,1]
    ax.plot(chain[:, 3], 'r')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Amplitude")
    ax = axs[1,1]
    ax.plot(chain[:, 4], 'b')
    ax.set_ylabel('Frequency')
    ax = axs[2,1]
    ax.plot(chain[:, 5], 'g')
    ax.set_ylabel('Phase')
    plt.show()


def plot_signal(mc, t, data):
    """
    This function takes in a MarkChain object , time series array, and the data we observed. It then will calculate the
    90% CI and plot the signal from the data and the 90% interval overtop

    :param mc: this is a MarkChain object of a chain and must have been ran before this. Will output error if mc has
    not been ran yet
    :param t: this is the time series array of when we observed data
    :param data: this is the array of data we observed to infer from
    :return: This function does not return anything but will show the plot via matplotlib (does not save plot)
    """

    # we first check to make sure that the chain is burned in
    if mc.is_burned_in:
        chain = mc.get_burn_samps()
        # if chain is burned in we take the bunred in samps only
    else:
        # if chain is not burned in we return nothing and print statement notifying user
        print("CHAIN IS NOT BURNED IN YET PLEASE RUN FOR MORE ITERATIONS")
        return

    # intialize some data structures for 90% CI
    percentile5 = np.zeros(len(t))
    percentile95 = np.zeros(len(t))

    # find the confindence interval using the model function
    for time in t:
        d = model_two(chain, data, time)
        percentile5[time] = np.percentile(d, 5)
        percentile95[time] = np.percentile(d, 95)

    # plotting lines, to plot both the data and the interval overtop
    # TODO: fix this for parrallelization and make plot pretty
    fig, ax = plt.subplots()
    ax.fill_between(t, percentile5, percentile95, color='orchid', alpha=0.5, label="90% CI")
    plt.plot(data, 'x')
    plt.plot(data, 'k')
    plt.xlabel('time')
    plt.ylabel('data')
    plt.legend()
    plt.show()


# this checks to make sure we are running this file as main and not just accessing the fcts
if __name__ == "__main__":
    main()
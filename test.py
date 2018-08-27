import numpy as np
import matplotlib.pyplot as plt
import markpy.chain as mp
from markpy.models.models import Model
from markpy.models import funcs


def get_data():
    A = 2.
    f = 4. * 2. * np.pi/20
    phi = 1.
    #np.random.seed(10)
    n = np.random.normal(0., 1, 20)
    t = np.arange(20)
    return t, A*np.cos(f*t+phi) + n


def model_exp(samp, data, t):
    return samp[0]*np.cos(samp[1]*t+samp[2])


def main():
    #np.random.seed(10)
    Nsteps = 10000
    sigprop = 0.11
    sig = 1
    D = 3
    t, data = get_data()
    params = ['Amp', 'Freq', 'Phase']
    priorrange = np.array([[0,5],[0,np.pi],[0,np.pi]])
    test_model = Model(model_exp, data, sig, D, funcs.res_norm, funcs.liklie_norm)
    mc = mp.MarkChain(test_model, D, priorrange, sigprop, params)
    mc.run(Nsteps, data, t)
    plot_chains(mc.states)
    samps = mc.get_independent_samps()
    plot_chains(samps)
    plot_signal(samps, t, data)
    mc.plot_acls()
    plot_signal(mc.states, t, data)

def plot_chains(chain):
    fig, axs = plt.subplots(nrows=3, ncols=1, sharex='col')
    ax = axs[0]
    ax.plot(chain[0,:,0], 'r')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Amplitude")
    ax = axs[1]
    ax.plot(chain[0,:,1], 'b')
    ax.set_ylabel('Frequency')
    ax = axs[2]
    ax.plot(chain[0,:,2], 'g')
    ax.set_ylabel('Phase')
    plt.show()




def plot_signal(chain, t, data):
    percentile5 = np.zeros(len(t))
    percentile95 = np.zeros(len(t))
    for time in t:
        d = model_exp(chain[0,:,:], data, time)
        percentile5[time] = np.percentile(d, 5)
        percentile95[time] = np.percentile(d, 95)

    fig, ax = plt.subplots()
    ax.fill_between(t, percentile5, percentile95, color='orchid', alpha=0.5, label="90% CI")
    plt.plot(data, 'x')
    plt.plot(data, 'k')
    plt.xlabel('time')
    plt.ylabel('data')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
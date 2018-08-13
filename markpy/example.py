import numpy as np
import matplotlib.pyplot as plt
import markpy as mp

def get_data():
    A = 2.
    f = 4. * 2. * np.pi/20
    phi = 1.
    np.random.seed(10)
    n = np.random.normal(0., 1, 20)
    t = np.arange(20)
    return t, A*np.cos(f*t+phi) + n


def model_exp(samp, data, t):
    return samp[0]*np.cos(samp[1]*t+samp[2])


def main():
    np.random.seed(10)
    Nsteps = 200000
    sigprop = 0.09
    sig = 1
    D = 3
    t, data = get_data()
    priorrange = np.array([[0,5],[0,np.pi],[0,np.pi]])
    test_model = mp.Model(model_exp, data, sig, D, mp.res_norm, mp.liklie_norm)
    mc = mp.MarkChain(test_model, D, priorrange, sigprop)
    mc.run(Nsteps, data, t)
    fig, axs = plt.subplots(nrows=1, ncols=3)
    ax = axs[0]
    ax.plot(mc.states[0,:,0],mc.states[0,:,1], 'rx')
    ax = axs[1]
    ax.plot(mc.states[0,:,0],mc.states[0,:,2], 'rx')
    ax = axs[2]
    ax.plot(mc.states[0,:,2],mc.states[0,:,1], 'rx')

    plt.show()






if __name__ == "__main__":
    main()
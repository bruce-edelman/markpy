import numpy as np
import matplotlib.pyplot as plt
import markpy.sampler as mp


def get_data():
    A = 2.
    f = 4. * 2. * np.pi/20
    phi = 1.
    #np.random.seed(10)
    n = np.random.normal(0., 1, 20)
    t = np.arange(20)
    return t, A*np.cos(f*t+phi) + n

def get_data_two():
    A1 = 2.
    A2 = 3.4
    f1 = 4. * 2. * np.pi/20
    f2 = 3.6 * 1.3 *np.pi/20
    phi1 = 1.
    phi2 = 2.7
    n = np.random.normal(0., 1, 30)
    t = np.arange(30)
    return t, A1*np.cos(f1*t+phi1)+A2*np.sin(f2*t+phi2)+n


def model_exp(samp, data, t):
    return samp[0]*np.cos(samp[1]*t+samp[2])

def model_two(samp, data, t):
    return samp[0]*np.cos(samp[1]*t+samp[2])+samp[3]*np.sin(samp[4]*t+samp[5])

def main():
    #np.random.seed(10)
    Nsteps = 50000
    sigprop = 0.11
    sig = 1
    D = 6
    t, data = get_data_two()
    mean = 0.5
    params = ['Amp', 'Freq', 'Phase', 'Amp2', 'Freq2', 'Phase2']
    priorrange = np.array([[0,5],[0,np.pi],[0,np.pi],[0,5],[0,np.pi],[0,np.pi]])


    liklie = mp.Liklie_Norm(model_two, mean, sig, data)
    test_model = mp.Model(model_two, data, sig, D, params, liklie)
    mc = mp.MarkChain(test_model, D, priorrange, sigprop)
    mc.run(Nsteps, data, t)
    plot_chains(mc.states)
    samps = mc.get_burn_samps()
    plot_chains(samps)
    plot_signal(mc, t, data)



def plot_chains(chain):
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
    if mc.is_burned_in:
         chain = mc.get_burn_samps()
    else:
         return
    percentile5 = np.zeros(len(t))
    percentile95 = np.zeros(len(t))
    for time in t:
        d = model_two(chain, data, time)
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
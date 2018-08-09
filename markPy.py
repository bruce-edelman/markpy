import matplotlib.pyplot as plt
import math
import random
import scipy.stats
import numpy as np



class state(object):
    def __init__(self, x, prior, likely):
        self.x = x
        self.prior = prior
        self.likely = likely

    def get_prior(self):
        return self.prior.model.pdf(self.x)

    def get_likely(self):
        return self.likely.model.pdf(self.likely.data).prod()

    def get_prob(self):
        return  self.prior.model.pdf(self.x)*self.likely.model.pdf(self.likely.data).prod()

    def get_logp(self):
        return math.log(self.prior.model.pdf(self.x)*self.likely.model.pdf(self.likely.data).prod())


class chain(object):
    def __init__(self, curstate, proposal):
        self.current = curstate
        self.propose = proposal
        self.states = [curstate]

    def met_hast_step(self):
        new =  self.propose(self.current)
        logpn, logpc = new.get_logp(), self.current.get_logp()
        if (logpn > logpc) or (math.log(random.random()) < logpn - logpc):
            self.current = new
        self.states.append(self.current)

    def run(self, n):
        for i in range(n):
            self.met_hast_step()


class prior(object):
    def __init__(self, model, mu, sd):
        self.mu = mu
        self.sd = sd
        self.model = model
        self.pdf = model.pdf


class likliehood(object):
    def __init__(self, model, mu, sd, data):
        self.mu = mu
        self.sd = sd
        self.model = model
        self.pdf = model.pdf
        self.data = data

def randomwalk(s):
    dx = random.random() - 0.5
    return state(s.x + dx, s.prior, s.likely)

def main():
    data = np.random.randn(30)
    plt.hist(data)
    plt.show()
    plt.figure()
    pr = prior(scipy.stats.norm(0, 1), 0, 1)
    li = likliehood(scipy.stats.norm(0, 1), 0, 1, data)
    c = chain(state(0, pr, li), randomwalk)
    c.run(1000)
    plt.plot([s.x for s in c.states])
    plt.show()
    plt.figure()
    plt.hist([s.x for s in c.states])
    plt.show()



if __name__ == "__main__":
    main()

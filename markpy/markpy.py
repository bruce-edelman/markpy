import numpy as np


class MarkChain(object):
    def __init__(self, PDF, d, priorrange, sigprop):
        self.oldsamp = np.array([np.random.uniform(priorrange[i][0],priorrange[i][1]) for i in range(d)])
        self.acc = 0
        self.dim = d
        self.sigmaprop = sigprop
        self.post = PDF
        self.priorrange = priorrange
        self.states = np.array([[self.oldsamp]])
        self.model = PDF

    def step(self, *kargs):
        newsamp = self.proposedStep()
        acc, newsamp = self.hastingsRatio(newsamp, *kargs)

        self.states = np.append(self.states, [[newsamp]], axis=1)
        if acc:
            self.acc += 1
        self.oldsamp = newsamp

    def proposedStep(self):
        return self.oldsamp+np.random.normal(0.,self.sigmaprop,self.dim)

    def AcceptenceRatio(self):
        return 1.*self.acc/self.N

    def run(self, N, *kargs):
        self.N = N
        for i in range(N):
            self.step(*kargs)

    def hastingsRatio(self, newsamp, *kargs):
        if not ((np.array([p1-p2 for p1,p2 in zip(newsamp, np.transpose(self.priorrange)[:][0])])>0).all()
                and (np.array([p2-p1 for p1,p2 in zip(newsamp, np.transpose(self.priorrange)[:][1])])>0).all()):
            acc = False
            return acc, self.oldsamp
        newp = self.model.get_posterior(newsamp, *kargs)
        oldp = self.model.get_posterior(self.oldsamp, *kargs)

        if newp >= oldp:
            acc = True
            return acc, newsamp
        else:
            prob = newp/oldp
            acc = np.random.choice([True, False], p=[prob,1.-prob])
            return acc, acc*newsamp + (1. - acc)*self.oldsamp


class Model(object):
    def __init__(self, model, d, sig, D, res, lik, prior=1):
        self.data = d
        self.model = model
        self.dim = D
        self.sig = sig
        self.res = res
        self.lik = lik
        self.prior = prior

    def get_posterior(self, samp, *kargs):
        resid = self.res(self.data, self.model, samp, *kargs)
        likliehood = self.lik(self.sig, 0.5, resid)
        return self.prior*likliehood

def res_norm(data, model, samp, *kargs):
    return (data - model(samp, *kargs))**2


def liklie_norm(sig, mean, res):
    return np.exp(-mean*(res.sum()/sig**2))



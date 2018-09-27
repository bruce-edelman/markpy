import markpy
import matplotlib.pyplot as plt
import numpy as np
import datetime, time
import kombine
from scipy.stats import multivariate_normal as mvn

ndim = 2

class Target(object):
    def __init__(self, cov):
        self.cov = cov
        self.ndim = self.cov[0].shape

    def logpdf(self, x):
        return mvn.logpdf(x, mean=np.zeros(self.ndim), cov=self.cov)

    def __call__(self, x):
        return self.logpdf(x)


A = np.random.rand(ndim,ndim)
cov = A + A.T + ndim*np.eye(ndim)
lnpdf = Target(cov)

start_time = time.perf_counter()
nwalkers = 500
sampler = kombine.Sampler(nwalkers, ndim, lnpdf, processes=1)
p0 = np.random.uniform(-10, 10, size=(nwalkers, ndim))
p, post, q = sampler.burnin(p0)
Nsteps = 1000
total = Nsteps*nwalkers*ndim
p, post, q = sampler.run_mcmc(Nsteps)

time_elapsed = round(time.perf_counter()-start_time, 2)
print("KOMBINE SAMPLER TEST - 2D Gaussian")
print("Time Elapsed: %.2f" % time_elapsed)
print("Average Step/sec: %.3f\n" % float(total/time_elapsed))

fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(15,3))
ax1.plot(sampler.acceptance_fraction, 'k', alpha=.5, label="Mean Acceptence Rate")
for p, ax in zip(range(ndim), [ax2, ax3]):
    ax.plot(sampler.chain[..., p], alpha=.1)
ax1.legend(loc='lower right')
plt.show()



start_time = time.perf_counter()
means = np.zeros([ndim])
sigs = np.array([cov[0,0],cov[1,1]])
stats = np.zeros([2, ndim])
stats[0,:] = means
stats[1,:] = sigs
params = ['x1', 'x2']
sigmaprop = .1
norm_model = markpy.NormModelAnalytic(params, cov, means, prior_stats=stats)
mc = markpy.ParallelMarkChain(100, norm_model, ndim, sigmaprop)
Nsteps = 5000
c = mc.run(Nsteps, progress=True, thin=10)
time_elapsed = round(time.perf_counter()-start_time, 2)
print("MARKPY SAMPLER TEST - 2D Gaussian")
print("Time Elapsed: %.2f" % time_elapsed)
print("Average Step/sec: %.3f\n" % float(total*10/time_elapsed))
fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(15,3))
ax1.plot(mc.AcceptenceFraction, 'k', alpha=.5, label="Mean Acceptence Rate")
for p, ax in zip(range(ndim), [ax2, ax3]):
    for walk in range(100):
        ax.plot(c[:, p, walk], alpha=.1)
ax1.legend(loc='lower right')
plt.show()

# markpy
Simple mcmc sampler created for bayesian inference and to learn the basics of mcmc sampling and practice OOP skills in python.

Details of how to run the sampler shown in various test files shown.

The sampler lets you create or use a pre-made model (likliehood) to use in sampling along with choosing what mcmc smapling algorithm to use. As of now only ones implemented are the generic Metropolis Hastings Algorithm and Gibbs Sampling (requires the problem to be multi-dimensional). 

The easiest way to generate a Markov Chain is to folloow test_analyitc.py and use the method MarkChain.run(Nsteps) or ParallelMarkChain.run(Nsteps) to generate samples in a numpy array of shape (nsteps, ndimensions, nwalkers) with nwalkers the number of indenedpent chains using. 



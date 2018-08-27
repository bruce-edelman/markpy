# markpy
simple free time project creating mcmc python package for bayesian inference


Example of how to use it on a simple 3-dimensional wave-model is shown in /markpy/test.py

Need to define a Model class and MarkChain class
then use MarkChain.run(Nsteps) and then You can access data by MarkChain.states

which is a (N.dimensions x N.samples) dimensional array for the markov chain

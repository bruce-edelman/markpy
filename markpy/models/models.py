class BaseModel(object):
        pass

class Model(object):
        """"
        Model class is an object that stores information about the model we choose to sample
        takes in three different fcts, model, res, lik and prior to initilialize. These fcts can be grabbed from markpy or
        programmed in when using markpy
        """

        def __init__(self, model, d, sig, D, res, lik, prior=1):
            # this Model class has parameters:
            # model - the model function of the problem we want to sample
            # d is the data we are inferring from
            # sig is the sigma of the model,
            # D is the dimension of the model
            # res is a function that calculates the residual,
            # lik is a function that calculates the liklihood of the model
            # prior is set to uniform (prior =1) but we can adjust this if wantegitd
            self.data = d  # data
            self.model = model  # primary model using
            self.dim = D  # dimension of model
            self.sig = sig  # sigma of model
            self.res = res  # fct to calcuate the residual
            self.lik = lik  # likliehood fct
            self.prior = prior  # prior fct (default to uniform prior=1)

        def get_posterior(self, samp, *kargs):
            # function that returns the posterior for the model (used in sampling)
            resid = self.res(self.data, self.model, samp, *kargs)  # calc the resid with given fct
            likliehood = self.lik(self.sig, 0.5, resid)  # calc the likliehood
            return self.prior * likliehood
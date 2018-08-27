import numpy as np
#function lists
#function types, res has parameters:
# data, model, samp. *kargs

# lik_type has parameters:
# sig, mean, res


def res_norm(data, model, samp, *kargs):
    # function of Res_type, normal
    return (data - model(samp, *kargs))**2


def liklie_norm(sig, mean, res):
    # function of Lik_type, normal,
    return np.exp(-mean*(res.sum()/sig**2))

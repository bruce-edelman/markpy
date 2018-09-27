import numpy as np

list1 = np.random.normal(0., 1., 445)
list2 = np.random.normal(0., 1., 452)


def thin(list, nbins=30):
    nlen = list.shape[0]
    binlen = int(nlen/nbins)
    binstarts = np.arange(0, nlen, binlen+1)
    out = np.zeros([nbins])
    ct = 0
    for i in binstarts:
        binend = i+binlen
        if binend > nlen-1:
            binend = nlen-1
        out[ct] = np.mean(list[i:binend])
        ct += 1

    return out

print(thin(list1), len(thin(list1)))
print(thin(list2), len(thin(list2)))
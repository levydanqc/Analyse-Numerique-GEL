import numpy as np


def pointfixe(fonction_pointfixe, x0, nmax, tolr):
    iterations = np.array([x0])
    iterations = np.append(iterations, fonction_pointfixe(x0))
    n = 1
    eps = np.finfo(np.float64).eps
    err_rel = np.abs(iterations[n]-iterations[n-1])/(np.abs(iterations[n])+eps)
    while (err_rel >= tolr and n < nmax):
        iterations = np.append(iterations, fonction_pointfixe(iterations[n]))
        err_rel = np.abs(iterations[n+1]-iterations[n]) / \
            (np.abs(iterations[n+1])+eps)
        n += 1
    return iterations

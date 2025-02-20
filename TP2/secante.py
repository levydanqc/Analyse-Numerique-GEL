import numpy as np


def secante(f, x0, x1, N, tol):
    x = np.zeros(N+1, dtype=float)
    x[0] = x0
    x[1] = x1

    for n in range(2, N+1):
        denom = f(x[n-1]) - f(x[n-2])
        if denom == 0:
            x = x[:n]
            break
        x[n] = x[n-1] - ((f(x[n-1]) * (x[n-1] - x[n-2])) / denom)

        if abs(x[n] - x[n-1]) < tol:
            x = x[:n+1]
            break
    return x

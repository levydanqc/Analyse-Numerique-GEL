import numpy as np

import matplotlib.pyplot as plt
from suiteSn import suiteSn

# Question 1
a = np.full((6, 1), 1)
b = np.arange(1, 7)
c = a.reshape(1, 6)
d = c * 17

I = np.identity(6)
J = np.full((6, 6), 1)
K = np.diag(b)

L = 55 * I - J + 2 * np.dot(a, c)
M = K.copy()
M[:, 0] = a.flatten()

dd = np.linalg.det(M)
x = np.linalg.solve(M, a)
N = np.linalg.solve(M, M.T)

# Affichage des résultats
print("a =\n", a)
print("b =\n", b)
print("c =\n", c)
print("d =\n", d)
print("I =\n", I)
print("J =\n", J)
print("K =\n", K)
print("L =\n", L)
print("M =\n", M)
print("Déterminant de M (dd) =", dd)
print("Solution de Mx = a (x) =\n", x)
print("Solution de MN = M' (N) =\n", N)

# Affichage de la Figure 1
plt.matshow(N)
plt.title("Matrice N")
plt.show()

# Définition de la fonction f(x)


def f(x):
    return -x**2 / 2 + np.exp(x) + np.sin(x)


x_vals = np.linspace(0, 1, 101)  # Points de l'intervalle [0,1]
plt.plot(x_vals, f(x_vals))
plt.title("Graphe de f(x) = -x^2/2 + e^x + sin(x)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.show()

# Question 2
n_max = 19
S_vals = suiteSn(n_max)
plt.plot(range(n_max + 1), S_vals, marker='o')
plt.title("Évolution de S_n en fonction de n")
plt.xlabel("n")
plt.ylabel("S_n")
plt.grid(True)
plt.show()


# Question 3
def df_exact(x):
    return -x + np.exp(x) + np.cos(x)


def D(x, h):
    return (f(x + h) - f(x)) / h


h_values = np.logspace(-1, -12, 12)  # 10^-1 à 10^-12
errors = np.abs(df_exact(0) - D(0, h_values))

plt.loglog(h_values, errors, marker='o')
plt.xlabel("h")
plt.ylabel("Erreur |f'(0) - D(0,h)|")
plt.title("Erreur en fonction de h")
plt.grid(True, which="both", linestyle="--")
plt.show()

# Question 4


def taylor_p2(x):
    return 1 + 2*x


x_values = np.logspace(-1, -6, 6)  # 10^-1 à 10^-6
errors = np.abs(f(x_values) - taylor_p2(x_values))

plt.plot(x_values, errors, marker='o')
plt.xlabel("x")
plt.ylabel("Erreur |f(x) - p2(x)|")
plt.title("Erreur de l'approxlimation de Taylor en fonction de x")
plt.grid(True, which="both", linestyle="--")
plt.show()

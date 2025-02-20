import numpy as np
import matplotlib.pyplot as plt
from pointfixe import pointfixe
from secante import secante


def f(x):
    return (x + 1) * (x - 1)**2


def g(x):
    return x - f(x) / 5


def df(x):
    return 3 * x**2 - 2 * x - 1


def g_newton(x):
    return x - f(x) / df(x)


def g_steff(x):
    y = f(x)
    z = f(y)
    return x - ((y - x)**2 / (z - 2*y + x))


N = 50
tol = 1e-7
x0_r1 = -1.5
x0_r2 = 1.5

# 1. Méthode du point fixe
x_r1 = pointfixe(g, x0_r1, N, tol)
x_r2 = pointfixe(g, x0_r2, N, tol)

E_r1 = np.abs(x_r1 - (-1))
E_r2 = np.abs(x_r2 - 1)

plt.figure(1)
plt.semilogy(range(len(E_r1)), E_r1, 'o-', label='Point fixe  (r1)')
plt.semilogy(range(len(E_r2)), E_r2, 'o-', label='Point fixe  (r2)')
plt.xlabel('n')
plt.ylabel('|x_n - r|')
plt.title('Figure 1: Erreur Point Fixe')
plt.legend()

plt.figure(2)
plt.plot(range(len(E_r1) - 1), E_r1[1:] /
         E_r1[:-1], 'o-', label='Rapport E_{n+1}/E_n  (r1)')
plt.plot(range(len(E_r2) - 1), E_r2[1:] /
         E_r2[:-1], 'o-', label='Rapport E_{n+1}/E_n  (r2)')
plt.xlabel('n')
plt.ylabel('E_{n+1}/E_n')
plt.title('Figure 2: Rapport Erreurs Point Fixe')
plt.legend()

# 2. Newton
x_r1_newton = pointfixe(g_newton, x0_r1, N, tol)
x_r2_newton = pointfixe(g_newton, x0_r2, N, tol)

E_r1_newton = np.abs(x_r1_newton - (-1))
E_r2_newton = np.abs(x_r2_newton - 1)

plt.figure(3)
plt.semilogy(range(len(E_r1_newton)), E_r1_newton, 'o-', label='Newton (r1)')
plt.semilogy(range(len(E_r2_newton)), E_r2_newton, 'o-', label='Newton (r2)')
plt.xlabel('n')
plt.ylabel('|x_n - r|')
plt.title('Figure 3: Erreur Newton')
plt.legend()

plt.figure(4)
plt.plot(range(len(E_r1_newton) - 1),
         E_r1_newton[1:] / E_r1_newton[:-1]**2, 'o-', label='Rapport E_{n+1}/E_n^2  (r1)')
plt.xlabel('n')
plt.ylabel('E_{n+1}/E_n^2')
plt.title('Figure 4: Rapport Erreurs Newton (r1)')
plt.legend()

plt.figure(5)
plt.plot(range(len(E_r2_newton) - 1),
         E_r2_newton[1:] / E_r2_newton[:-1], 'o-', label='Rapport E_{n+1}/E_n  (r2)')
plt.xlabel('n')
plt.ylabel('E_{n+1}/E_n')
plt.title('Figure 5: Rapport Erreurs Newton (r2)')
plt.legend()

# 3. Steffenson
x_r1_steff = pointfixe(g_steff, x0_r1, N, tol)
E_r1_steff = np.abs(x_r1_steff - (-1))

plt.figure(6)
plt.semilogy(range(len(E_r1_steff)), E_r1_steff, 'o-', label='Steffenson (r1)')
plt.xlabel('n')
plt.ylabel('|x_n - r|')
plt.title('Figure 6: Erreur Steffenson r1')
plt.legend()

plt.figure(7)
plt.plot(range(len(E_r1_steff) - 1),
         E_r1_steff[1:] / E_r1_steff[:-1]**2, 'o-', label='Rapport E_{n+1}/E_n^2  (r1)')
plt.xlabel('n')
plt.ylabel('E_{n+1}/E_n^2')
plt.title('Figure 7: Rapport Erreurs Steffenson (r1)')
plt.legend()

# 4. Sécante
x_r1_secante = secante(f, -1.5, -1.25, N, tol)
E_r1_secante = np.abs(x_r1_secante - (-1))

print(x_r1_secante)

plt.figure(8)
plt.semilogy(range(len(E_r1_secante)), E_r1_secante,
             'o-', label='Sécante (r1)')
plt.xlabel('n')
plt.ylabel('|x_n - r|')
plt.title('Figure 8: Erreur Sécante (r1)')
plt.legend()

phi = (1 + np.sqrt(5)) / 2
plt.figure(9)
plt.plot(range(len(E_r1_secante) - 1),
         E_r1_secante[1:] / E_r1_secante[:-1]**phi, 'o-', label='Rapport E_{n+1}/E_n^phi  (r1)')
plt.xlabel('n')
plt.ylabel('E_{n+1}/E_n^phi')
plt.title('Figure 9: Rapport Erreurs Sécante (r1)')
plt.legend()


plt.show()

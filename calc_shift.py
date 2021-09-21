import numpy as np
import pylab
from plot_curve import PCurve

A = int(2 * 3**3 * 10000)
gamma = int(2.8e-5 * 1e18)
D = 3 * 10**18
N = 3


def shift(A, gamma):
    A = int(A)
    gamma = int(gamma)
    c = PCurve(A, gamma, D, N)
    inv1 = c.get_xcp()
    c.x = [2 * 10**18, 10**36 // (2 * 10**18)]
    inv2 = c.get_xcp()
    return inv2 / inv1


print(shift(A, gamma))

_A = np.logspace(np.log10(0.1), np.log10(30), 100) * A
pylab.plot(np.log(_A), np.log([shift(a, gamma) - 1 for a in _A]))
pylab.show()

_gamma = np.logspace(np.log10(0.01), np.log10(10), 100) * gamma
pylab.plot(np.log(_gamma), np.log([shift(A, g) - 1 for g in _gamma]))
pylab.show()

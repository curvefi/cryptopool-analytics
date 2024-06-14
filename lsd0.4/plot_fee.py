#!/usr/bin/env python3

import numpy as np
from simulation_int_many import Curve


N = 2
A = int(500 * N**N * 10000)
gamma = int(0.02 * 1e18)

fee_gamma = 9e-2
mid_fee = 0.03  # %
out_fee = 0.15   # %

N_POINTS = 5000


def f_fees(A, gamma):
    D = N * 10**18
    c = Curve(A, gamma, D, N)
    X = 1e18 * np.linspace(0.5, 1.5, N_POINTS)
    Y = np.array([c.y(int(x), 0, 1) for x in X])
    p = (X[1:] - X[:-1]) / (Y[:-1] - Y[1:]) - 1
    f = fee_gamma / (fee_gamma + 1 - X*Y / (D / N) ** N)
    fees = mid_fee * f + out_fee * (1 - f)
    return p * 100, fees[1:]


if __name__ == '__main__':
    import pylab
    pylab.plot(*f_fees(A, gamma))
    pylab.plot([0, 1], [0, 1], '--', c='gray')
    pylab.plot([-1, 0], [1, 0], '--', c='gray')
    pylab.grid()
    pylab.show()

#!/usr/bin/env python3

import numpy as np
from scipy.interpolate import interp1d

from simulation_int_many import Curve, geometric_mean


A = 4.525
gamma = 54.7


D = 2 * 10**18
N = 2
dp = 1e-8
dx = 1e-5


class PCurve(Curve):
    def get_xcp(self):
        # First calculate the ideal balance
        # Then calculate, what the constant-product would be
        D = self.D()
        N = len(self.x)
        X = [D * 10**18 // (N * p) for p in self.p]

        return geometric_mean(X)


def profit_deriv_calc(c, p):
    p0 = int((1 + p) * 1e18)
    p1 = int((1 + p + dp) * 1e18)
    c.p[0] = p0
    P0 = c.get_xcp()
    c.p[0] = p1
    P1 = c.get_xcp()
    return (P1 - P0) / P0 * p0 / (p1 - p0)


def f_profit_deriv(A, gamma):
    c = PCurve(A, gamma, D, N)
    d_price = np.logspace(np.log10(0.1), np.log10(10), 6000)
    d_profit = [profit_deriv_calc(c, p) for p in d_price]
    return interp1d(d_price, d_profit, fill_value='extrapolate')


def f_liq_density(A, gamma):
    c = PCurve(A, gamma, D, N)
    prices = np.logspace(np.log10(0.2), np.log10(1.8), 6000)
    dens = []
    derivs = []
    for p in prices:
        xx = np.array([1 - dx, 1, 1 + dx]) * p * 1e18
        yy = np.array([c.y(int(x), 0, 1) for x in xx])
        d2y = (yy[2] + yy[0] - 2 * yy[1]) / ((xx[2] - xx[0]) / 2)**2
        deriv = -0.5 * ((xx[2] - xx[1]) / (yy[2] - yy[1]) + (xx[1] - xx[0]) / (yy[1] - yy[0]))
        dens.append(2 / (xx[1] * d2y))
        derivs.append(deriv)
    return np.array(derivs), np.array(dens)


if __name__ == '__main__':
    import pylab
    p, dens = f_liq_density(int(A * 2**2 * 10000), int(gamma * 1e18))
    pylab.loglog(p, dens)
    pylab.xlabel('Price')
    pylab.ylabel('Liquidity density (xyk units)')

    pylab.grid()
    pylab.tight_layout()
    pylab.show()

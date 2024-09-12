#!/usr/bin/env python3

import numpy as np
from simulation_int_many import Curve, geometric_mean


A = 85.78
gamma = 0.0153


D = 2 * 10**18
N = 2


class PCurve(Curve):
    def get_xcp(self):
        # First calculate the ideal balance
        # Then calculate, what the constant-product would be
        D = self.D()
        N = len(self.x)
        X = [D * 10**18 // (N * p) for p in self.p]

        return geometric_mean(X)


def f_impact(fracs):
    c = PCurve(int(A * 10000 * N**N), int(gamma * 1e18), D, N)
    y0 = c.x[1]
    x0 = c.x[0]
    x = x0 * (1 + fracs)
    y = np.array([c.y(int(_x), 0, 1) for _x in x])
    impacts = x0 * fracs / (y0 - y) - 1
    return impacts


if __name__ == '__main__':
    import pylab

    traded_frac = np.linspace(1e-5, 0.1, 500)
    fracs = traded_frac * 2
    impact = f_impact(fracs)

    pylab.plot(traded_frac * 100, impact * 100)

    pylab.xlabel('Fraction of TVL swapped (%)')
    pylab.ylabel('Price impact (%)')

    pylab.grid()
    pylab.tight_layout()
    pylab.show()

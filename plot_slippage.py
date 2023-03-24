import numpy as np
from simulation_int_many import Curve, geometric_mean


N = 3
A = int(1000 * N**N * 10000)
gamma = int(4e-4 * 1e18)
D = 1000 * 10**18
dp = 1e-8
dx = 1e-6


class PCurve(Curve):
    def get_xcp(self):
        # First calculate the ideal balance
        # Then calculate, what the constant-product would be
        D = self.D()
        N = len(self.x)
        X = [D * 10**18 // (N * p) for p in self.p]

        return geometric_mean(X)


def f_slippage():
    D = 3 * 10**18
    c = PCurve(A, gamma, D, N)
    p = []
    amounts = []
    for dDD in [1e-8] + list(np.logspace(-5, -0.5)):
        dx = int(D * dDD)
        x0 = c.x[0]
        y0 = c.x[1]
        y = c.y(x0 + dx, 0, 1)
        amounts.append(dDD)
        p.append(dx / (y0 - y))
    p = np.array(p)
    p = (p - p[0]) / p[0]
    return amounts[1:], p[1:]


if __name__ == '__main__':
    import pylab
    amounts, p = f_slippage()
    pylab.loglog(amounts, p)
    pylab.xlabel('trade_size / pool_size')
    pylab.ylabel('Relative price impact')
    pylab.show()

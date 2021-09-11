from simulate_apy import Simulator
from multiprocessing import Pool
import numpy as np

A = 2 * 3**3 * 10000
gammas = np.logspace(np.log10(5e-7), np.log10(2e-3), 20)

sim = Simulator()


def calc(gamma):
    res = sim.simulate(A=int(A), gamma=int(gamma * 1e18))[1]
    print(f'A={A}, gamma={gamma:.3e} -> {res:.3f}')
    return res


pool = Pool(4)

if __name__ == '__main__':
    import pylab

    lds = pool.map(calc, gammas)

    pylab.semilogx(gammas, lds)
    pylab.show()

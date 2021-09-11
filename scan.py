from simulate_apy import Simulator
from multiprocessing import Pool
import numpy as np

As = [2, 10, 50]
gammas = np.logspace(np.log10(5e-7), np.log10(2e-3), 20)

sim = Simulator()


def calc(params):
    A, gamma = params
    res = sim.simulate(A=int(A * 3**3 * 10000), gamma=int(gamma * 1e18))[1]
    print(f'A={A:.1f}, gamma={gamma:.3e} -> {res:.3f}')
    return res


pool = Pool(4)

if __name__ == '__main__':
    import pylab

    for A in As:
        lds = pool.map(calc, [(A, gamma) for gamma in gammas])
        pylab.semilogx(gammas, lds, label=f'A={A:.1f}')

    pylab.legend()
    pylab.show()

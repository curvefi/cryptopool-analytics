import numpy as np
from scipy.optimize import minimize
from plot_curve import f_profit_deriv


X = np.linspace(-20e-2, 20e-2, 1000)
initial_A = 2 * 3**3 * 10000
initial_gamma = int(2.8e-5 * 1e18)

initial_f = f_profit_deriv(initial_A, initial_gamma)
initial_y = initial_f(X)
norm = initial_y.max()**2 * len(X)


def get_gamma(A):
    def g(gamma):
        print('Calculating for', A / (3**3 * 10000), gamma)
        f = f_profit_deriv(int(A), int(gamma * 1e18))
        y = f(X)
        return ((initial_y - y) ** 2).sum() / norm
    return minimize(g, (initial_gamma / 1e18), method='TNC', options={'eps': 1e-8}, bounds=[(1e-9, 0.1)]).x[0]


if __name__ == '__main__':
    print(get_gamma(0.6 * 3**3 * 10000))

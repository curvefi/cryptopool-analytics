#!/usr/bin/env python3

from math import sqrt, log10
import numpy as np

steps = 5000
n = 2


def model(p0, y, d):
    D = p0**2 * y**2 - 4 * p0 * y * d * (n / (2 * n - 1))**2
    x0 = (p0 * y + sqrt(D)) / (2 * (n / (2 * n - 1))**2)
    y_opt = n / (2 * n - 1) * x0 / p0
    d_opt = (n - 1) / n * p0 * y_opt
    return y_opt, d_opt


y = 2.0
d = 1.0

for p in np.logspace(log10(1), log10(2), steps):
    y, d = model(p, y, d)
    print(p, y)

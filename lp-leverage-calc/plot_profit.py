#!/usr/bin/env python3

import lzma
import json
import pylab

import numpy as np

with lzma.open("detailed-output.json.xz", "r") as f:
    data = json.load(f)

t = np.array([d['t'] for d in data[::1000]])
t -= t[0]
profit = np.array([d['profit'] for d in data[::1000]])
profit -= t * 0.05 / (365 * 86400)

pylab.plot(t, profit)
pylab.show()

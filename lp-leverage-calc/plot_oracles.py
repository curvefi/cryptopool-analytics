#!/usr/bin/env python3

import lzma
import json
import pylab

import numpy as np

with lzma.open("detailed-output.json.xz", "r") as f:
    data = json.load(f)

t = np.array([d['t'] for d in data[::1000]])
t = (t - t[0]) / (86400)
oracle1 = np.array([d['price_oracle'] for d in data[::1000]])
oracle2 = np.array([d['price_scale'] for d in data[::1000]])

pylab.plot(t, oracle1)
pylab.plot(t, oracle2)
pylab.show()

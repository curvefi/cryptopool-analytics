import json
import numpy as np
import pylab

FEE = 0.003 * 2/3

with open('trades.json', 'r') as f:
    data = json.load(f)

pre_profits = []
for pair, d in data['pricevol'].items():
    for obj in d:
        pre_profits += [
            (obj['t'], FEE * obj['volume'] / (data['balances'][pair][str(obj['block'])]['balance1'] * 2))
        ]
pre_profits = sorted(pre_profits)
xcp = 1.0
times = []
profits = []
for t, p in pre_profits:
    xcp += p
    times.append(t)
    profits.append(xcp)

# pylab.plot((np.array(times) - times[0]) / 86400, profits)
pylab.plot(np.array(times), profits)
pylab.show()

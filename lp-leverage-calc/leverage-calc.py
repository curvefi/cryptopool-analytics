#!/usr/bin/env python3

import json
import pylab
from math import sqrt, log
import numpy as np

fname = 'btcusdt-2023-2024.json'

rebalance_fee = 0.0003

with open(fname, 'r') as f:
    data = json.load(f)

lp_data = []
t_data = []

for row in data:
    r = [int(row[0]) // 1000] + [float(r) for r in row[1:6]]
    t_data.append(r[:])
    r = [r[0]] + [sqrt(x) for x in r[1:5]] + [sqrt(r[5])]
    lp_data.append(r)


def get_apy(rebalance_threshold):
    tokens = 1.0
    usd = t_data[0][1] * tokens
    debt = usd
    lp_tokens = usd * 2 / lp_data[0][1]

    previous_price = lp_data[0][1]

    for row_t, row_lp in zip(t_data, lp_data):
        new_lp = row_lp[4]
        new_t = row_t[4]

        # Rebalance
        if abs(log(new_lp / previous_price)) >= rebalance_threshold:
            usd_value = lp_tokens * new_lp - debt
            previous_debt = debt
            debt = usd_value
            lp_tokens = usd_value * 2 / new_lp * (1 - abs(previous_debt - debt) / debt * rebalance_fee)
            previous_price = new_lp

        usd_value = lp_tokens * new_lp - debt
        btc_value = usd_value / new_t

    apy = btc_value ** (1 / ((t_data[-1][0] - t_data[0][0]) / (365 * 86400)))
    print(rebalance_threshold, apy)
    return apy


freq = np.logspace(np.log10(0.001), np.log10(0.8), 100)
apys = np.array([get_apy(f) for f in freq])

pylab.semilogx(freq, 1 - apys)

pylab.xlabel('Rebalance threshold')
pylab.ylabel('Loss in APY')
pylab.tight_layout()
pylab.show()

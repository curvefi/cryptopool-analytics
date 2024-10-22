#!/usr/bin/env python3

import json
import numpy as np
from math import sqrt, log10
from datetime import datetime


pool = None
price_data = None


class AMM:
    def __init__(self, p, fee):
        self.X = [p, 1]
        self.fee = fee

    def get_invariant(self):
        return self.X[0] * self.X[1]

    def get_dy(self, i, j, in_amount):
        return (self.X[j] - self.X[0] * self.X[1] / (self.X[i] + in_amount)) * (1 - self.fee)

    def exchange(self, i, j, amount):
        out = self.get_dy(i, j, amount)
        self.X[i] += amount
        self.X[j] -= out
        return out

    def get_p(self):
        return self.X[0] / self.X[1]

    def trade_to_price(self, p):
        initial_price = self.get_p()

        if p > initial_price:
            # Decrease debt + decrease collateral
            p *= (1 - self.fee)
            if p <= initial_price:
                return

        elif p < initial_price:
            p /= (1 - self.fee)
            if p >= initial_price:
                return

        else:
            return

        inv = self.X[0] * self.X[1]
        x_after = sqrt(inv * p)
        y_after = inv / x_after

        if p > initial_price:
            x_after += (x_after - self.X[0]) * self.fee  # Buy collateral -> more USD in
        else:
            y_after += (y_after - self.X[1]) * self.fee  # Dupm more collateral -> more collateral in

        self.X = [x_after, y_after]

    def get_value(self, p):
        total = self.X[0] + self.X[1] * p
        return sqrt((total / 2) * (total / (2 * p)))


class Simulator:
    def __init__(self, filename, ext_fee, add_reverse=False,
                 log=False, verbose=False, func=(lambda x: x)):
        """
        filename - OHLC data in the same format as Binance returns
        ext_fee - Fee which arb trader pays to external platforms
        add_reverse - Attach the same data with the time reversed
        func - postprocessing function for the price
        """
        self.filename = filename
        self.ext_fee = ext_fee
        self.add_reverse = add_reverse
        self.func = func
        self.log = log
        self.verbose = verbose
        self.load_prices()

    def load_prices(self):
        global price_data

        if self.filename.endswith('.gz'):
            import gzip
            with gzip.open(self.filename, "r") as f:
                data = json.load(f)
        else:
            with open(self.filename, "r") as f:
                data = json.load(f)

        # timestamp, OHLC, vol
        unfiltered_data = [[int(d[0])] + [self.func(float(x)) for x in d[1:6]] for d in data]
        data = []
        prev_time = 0
        for d in unfiltered_data:
            if d[0] >= prev_time:
                data.append(d)
                prev_time = d[0]
        if self.add_reverse:
            t0 = data[-1][0]
            data += [[t0 + (t0 - d[0])] + d[1:] for d in data[::-1]]
        price_data = data

    def single_run(self, fee):
        # Data for prices
        data = price_data
        amm = AMM(data[0][1], fee)
        initial_value = amm.get_value(data[0][1])

        losses = []

        t_start = data[0][0]
        t_end = data[-1][0]

        for t, o, high, low, c, vol in data:
            # max_price = amm.p_up(amm.max_band)
            # min_price = amm.p_down(amm.min_band)
            high = high * (1 - self.ext_fee)
            low = low * (1 + self.ext_fee)
            if high > amm.get_p():
                amm.trade_to_price(high)

            if low < amm.get_p():
                amm.trade_to_price(low)

            if self.log or self.verbose:
                d = datetime.fromtimestamp(t//1000).strftime("%Y/%m/%d %H:%M")
                current_value = amm.get_value()
                loss = current_value / initial_value * 100
                if self.log:
                    print(f'{d}\t{o:.2f}\t{amm.get_p():.2f}\t\t{loss:.2f}%')
                if self.verbose:
                    losses.append([t//1000, loss / 100])

        if losses:
            self.losses = losses

        current_value = amm.get_value(c)
        profit = (current_value / initial_value) ** ((365 * 86400 * 1000) / (t_end - t_start)) - 1
        return profit


if __name__ == '__main__':
    simulator = Simulator(
            filename='btcusdt-2023-2024.json',
            ext_fee=0.0003, add_reverse=True,
            log=False, verbose=False)

    for fee in np.logspace(log10(0.0005), log10(0.5), 30):
        profit = simulator.single_run(fee=fee)
        print(fee, profit)

#!/usr/bin/env python3

import json
import numpy as np
from math import sqrt, log10
from datetime import datetime


pool = None
price_data = None


class AMM:
    def __init__(self, collateral, leverage, fee, oracle):
        self.collateral = collateral  # y
        self.leverage = leverage
        self.fee = fee
        self.p_oracle = oracle
        self.debt = collateral * oracle * (leverage - 1) / leverage

    def set_p_oracle(self, p):
        self.p_oracle = p

    def get_x0(self):
        lev_ratio = (self.leverage / (2 * self.leverage - 1))**2
        D = self.p_oracle**2 * self.collateral**2 - 4 * self.p_oracle * self.collateral * self.debt * lev_ratio
        return (self.p_oracle * self.collateral + sqrt(D)) / (2 * lev_ratio)

    def get_invariant(self):
        x0 = self.get_x0()
        return self.collateral * (x0 - self.debt)

    def get_dy(self, i, j, in_amount):
        x0 = self.get_x0()
        y_initial = self.collateral
        x_initial = x0 - self.debt
        if i == 0:
            x = x_initial + in_amount
            y = x_initial * y_initial / x
            return (y_initial - y) * (1 - self.fee)
        else:
            y = y_initial + in_amount
            x = x_initial * y_initial / y
            return (x_initial - x) * (1 - self.fee)

    def exchange(self, i, j, amount):
        x0 = self.get_x0()
        y_initial = self.collateral
        x_initial = x0 - self.debt
        if i == 0:
            x = x_initial + amount
            y = x_initial * y_initial / x
            dy = (y_initial - y) * (1 - self.fee)
            self.debt -= amount
            self.collateral -= dy
            return dy
        else:
            y = y_initial + amount
            x = x_initial * y_initial / y
            dx = (x_initial - x) * (1 - self.fee)
            self.collateral += amount
            self.debt += dx
            return dx

    def get_p(self):
        x0 = self.get_x0()
        return (x0 - self.debt) / self.collateral

    def trade_to_price(self, p):
        x0 = self.get_x0()
        initial_price = (x0 - self.debt) / self.collateral

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

        inv = (x0 - self.debt) * self.collateral
        x_after = sqrt(inv * p)
        y_after = x_after / p

        if p > initial_price:
            x_after += (x_after - (x0 - self.debt)) * self.fee  # Buy collateral -> more USD in
        else:
            y_after += (y_after - self.collateral) * self.fee  # Dupm more collateral -> more collateral in

        self.collateral = y_after
        self.debt = x0 - x_after

    def get_value(self):
        return self.collateral * self.p_oracle - self.debt


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
        self.ema_time = 0
        self.emas = []
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

    def update_emas(self, Texp):
        if self.ema_time != Texp:
            self.ema_time = Texp
            self.emas = []
            ema = price_data[0][1]
            ema_t = price_data[0][0]
            for t, _, high, low, _, _ in price_data:
                ema_mul = 2 ** (- (t - ema_t) / (1000 * Texp))
                ema = ema * ema_mul + (low + high) / 2 * (1 - ema_mul)
                ema_t = t
                self.emas.append(ema)

    def single_run(self, fee, Texp, leverage):
        self.update_emas(Texp)

        # Data for prices
        data = price_data
        emas = self.emas
        initial_price = emas[0]
        amm = AMM(collateral=(1.0 * leverage), leverage=leverage, fee=fee, oracle=initial_price)
        initial_value = amm.get_value() / initial_price**leverage

        losses = []

        t_start = data[0][0]
        t_end = data[-1][0]

        for (t, o, high, low, c, vol), ema in zip(data, emas):
            amm.set_p_oracle(ema)
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
                current_value = amm.get_value() / ema**leverage
                loss = current_value / initial_value * 100
                if self.log:
                    print(f'{d}\t{o:.2f}\t{ema:.2f}\t{amm.get_p():.2f}\t\t{loss:.2f}%')
                if self.verbose:
                    losses.append([t//1000, loss / 100])

        if losses:
            self.losses = losses

        current_value = amm.get_value() / ema**leverage
        loss = 1 - (current_value / initial_value) ** ((365 * 86400 * 1000) / (t_end - t_start))
        return loss


if __name__ == '__main__':
    simulator = Simulator(
            filename='btcusdt-2023-2024.json',
            ext_fee=0.0002, add_reverse=True,
            log=False, verbose=False, func=sqrt)

    fees = []
    losses = []

    for fee in np.logspace(log10(0.0005), log10(0.2), 30):
        fees.append(fee)
        loss = simulator.single_run(fee=fee, Texp=866, leverage=2)
        losses.append(loss)
        print(fee, loss)

    import pylab
    pylab.semilogx(fees, losses)
    pylab.xlabel('Rebalancing AMM fee')
    pylab.ylabel('Loss in APY')
    pylab.tight_layout()
    pylab.show()

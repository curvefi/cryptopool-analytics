import gzip
import json
import numpy as np
from scipy.interpolate import interp1d
from plot_curve import f_profit_deriv, f_liq_density

FEE = 0.003 * 2/3

# not feasible gas-wise if lower
# makes sense to do like gas / fee * 2 or thereabouts
USD_TRADE_LIMIT = 50000  # USD


def load():
    with gzip.open('trades.json.gz', 'r') as f:
        data = json.load(f)
    TRUNC_BLOCK = 12580000

    pre_profits = []
    pre_prices = []
    price_0 = {}  # eth-usdt, eth-wbtc, wbtc-usdc
    btc2usd = None
    for pair in ['eth-usdt', 'wbtc-usdc', 'eth-wbtc']:
        d = data['pricevol'][pair]
        _t = []
        _price = []
        for obj in d:
            if obj['block'] < TRUNC_BLOCK:
                continue
            if pair not in price_0:
                price_0[pair] = obj['p']
            low_vol = False

            # This handling of truncating small trades is a bit dirty but works
            if pair == 'eth-usdt':
                if obj['volume'] < USD_TRADE_LIMIT:
                    low_vol = True
            elif pair == 'eth-wbtc':
                v = btc2usd(obj['t']) * obj['volume']
                if v < USD_TRADE_LIMIT:
                    low_vol = True
            elif pair == 'wbtc-usdc':
                if obj['volume'] < USD_TRADE_LIMIT:
                    low_vol = True

            if low_vol:
                pre_profits += [(obj['t'], 0.0)]
            else:
                pre_profits += [
                    (obj['t'], FEE * obj['volume'] / (data['balances'][pair][str(obj['block'])]['balance1'] * 2))
                ]
            pre_prices += [
                (obj['t'], pair, obj['p'])
            ]
            _t.append(obj['t'])
            _price.append(obj['p'])

        if pair == 'wbtc-usdc':
            btc2usd = interp1d(_t, _price, fill_value='extrapolate')

    pre_profits = sorted(pre_profits)
    pre_prices = sorted(pre_prices)
    prices = []
    p_vec = [price_0['wbtc-usdc'], price_0['eth-usdt']]
    for t, pair, pp in pre_prices:
        p_vec = p_vec[:]
        if pair == 'wbtc-usdc':
            p_vec[0] = pp
        elif pair == 'eth-usdt':
            p_vec[1] = pp
        else:  # eth-wbtc
            p_vec[1] = p_vec[0] * pp
        prices.append(p_vec)
    xcp = 1.0
    times = []
    profits = []
    for t, p in pre_profits:
        xcp += p
        times.append(t)
        profits.append(xcp)

    return np.array(times), np.array(profits), np.array(prices)


def ema(times, prices, T=600):
    p0 = prices[0]
    muls = np.exp(-(times[1:] - times[:-1]) / T)
    result = [p0]
    for mul, p in zip(muls, prices):
        result.append(result[-1] * mul + p * (1 - mul))
    return np.array(result)


class Simulator:
    def __init__(self, step=0.002):
        self.times, self.profits, self.prices = load()
        self.emas = ema(self.times, self.prices)
        self.step = step
        self.allowed_extra_profit = 2e-6

    def simulate(self, A, gamma):
        f_profit = f_profit_deriv(A, gamma)
        f_ld = f_liq_density(A, gamma)
        p0 = self.prices[0]
        p = [p0]
        _profit = self.profits[0]
        lds = []

        # for t, profit, ema in zip(self.times[1:], self.profits, self.emas):
        for i in range(1, len(self.times)):
            _profit += self.profits[i] - self.profits[i - 1]
            pvec = np.log(self.emas[i] / p[-1])
            delta = (pvec ** 2).sum() ** .5
            lds.append(f_ld(delta))
            if delta < self.step or self.profits[i] == self.profits[i - 1]:
                p.append(p[-1])
            else:
                d_profit = abs(f_profit(delta) * self.step * _profit)
                if _profit - d_profit - 1 > (self.profits[i] - 1) / 2 + self.allowed_extra_profit:
                    _profit -= d_profit
                    new_p = np.exp(self.step / delta * pvec) * p[-1]
                    p.append(new_p)
                else:
                    p.append(p[-1])

        dt = (self.times[1:] - self.times[:-1])

        # Normal mean
        lds = np.array(lds) * dt
        mean = np.sum(lds) / (self.times[-1] - self.times[0])

        # Inverse mean
        # lds = (1 / np.array(lds)) * dt
        # mean = 1 / (np.sum(lds) / (self.times[-1] - self.times[0]))

        # Time-median LD
        # lds = np.array(lds)
        # ix = lds.argsort()
        # mean = lds[ix][np.cumsum(dt[ix]).searchsorted((self.times[-1] - self.times[0]) / 2)]

        return np.array(p), mean, lds


if __name__ == '__main__':
    import pylab

    s = Simulator()

    A0 = 0.2
    gamma0 = 3e-4

    # for mul in 2 ** np.arange(10):
    #     for gamma_mul in np.logspace(np.log10(0.1), np.log10(10), 5):
    #         A = int(A0 * mul * 3**3 * 10000)
    #         gamma = int(gamma0 * 0.5 ** (10/mul) * gamma_mul * 1e18)
    #         _, ld, _ = s.simulate(A=A, gamma=gamma)
    #         print(A / (3**3 * 10000), gamma / 1e18, ld)
    #     print()

    t = s.times
    p, ld, lds = s.simulate(A=int(2 * 3**3 * 10000), gamma=int(2.1e-5 * 1e18))
    print(ld)

    pylab.plot((t - t[0]) / 3600, s.emas[:, 1], color='gray')
    pylab.plot((t - t[0]) / 3600, p[:, 1], color='blue')
    # pylab.plot((t - t[0])[1:] / 3600, lds)

    pylab.show()

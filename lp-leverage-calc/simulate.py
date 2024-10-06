from math import sqrt


class AMM:
    def __init__(self, collateral, debt, leverage, fee, oracle):
        self.collateral = collateral  # y
        self.debt = debt
        self.leverage = leverage
        self.fee = fee
        self.p_oracle = oracle

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

        self.collateral = y_after
        self.debt = x0 - x_after


if __name__ == '__main__':
    from math import log10
    import numpy as np

    steps = 5000

    amm = AMM(collateral=2.0, debt=1.0, leverage=2, fee=0.000, oracle=1.0)

    for p in np.logspace(log10(1), log10(2), steps):
        amm.set_p_oracle(p)
        amm.trade_to_price(p)
        print(p, amm.collateral, amm.debt)

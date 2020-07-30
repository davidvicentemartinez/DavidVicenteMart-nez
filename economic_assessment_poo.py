import pandas as pd
import numpy as np

from numpy_financial import pmt, ipmt, ppmt, npv, irr
from enum import Enum

data_factors = np.array([[0.3, 0.5, 0.6],
                         [0.8, 0.6, 0.2],
                         [0.3, 0.3, 0.2],
                         [0.2, 0.2, 0.15],
                         [0.3, 0.3, 0.2],
                         [0.2, 0.2, 0.1],
                         [0.1, 0.1, 0.05],
                         [0.3, 0.4, 0.4],
                         [0.35, 0.25, 0.2],
                         [0.1, 0.1, 0.1]])

Capital_factors = pd.DataFrame(data_factors,
                               index=["fer", "fp", "fi", "fel", "fc", "fs", "fl", "OS", "D&E", "X"],
                               columns=["Fluids", "Fluids-Solids", "Solids"])


class Category(Enum):
    fluids = "Fluids"
    fluids_solids = "Fluids-Solids"
    solids = "Solids"


class Equipment:
    def __init__(self, category, fm=1, installed=True):
        assert type(installed) == bool
        assert isinstance(category, Category)
        self.category = category.value
        self.fm = fm
        self.installed = True

    def william(self, cost_ref, cap, cap_ref, n):
        assert n <= 1
        assert n >= 0.6

        return cost_ref * pow(cap / cap_ref, n)

    def lang(self, C, Capital_factors):
        t = self.category
        C *= ((1 + Capital_factors.loc["fp"][t]) * self.fm + (
                Capital_factors.loc["fer"][t] + Capital_factors.loc["fel"][t]
                + Capital_factors.loc["fi"][t] + Capital_factors.loc["fc"][t]
                + Capital_factors.loc["fs"][t] + Capital_factors.loc["fl"][t]))
        return C


class Boiler(Equipment):
    def __init__(self, Q, p, category=Category.fluids, fm=1, installed=True):
        super().__init__(category, fm, installed)
        self.Q = Q
        self.p = p

    def boiler(self):
        if self.Q < 5000 or self.Q > 800000:
            print(
                f"    - WARNING: boiler vapor production out of method bounds, 5000 < Q < 800000. Results may not be accurate.")
        if self.p < 10 or self.p > 70:
            print(f"    - WARNING: boiler pressure out of method bounds, 10 < p < 70. Results may not be accurate.")
        if self.Q < 20000:
            C = 106000 + 8.7 * self.Q
        elif self.Q < 200000:
            if self.p < 15:
                C = 110000 + 4.5 * self.Q ** 0.9
            elif self.p < 40:
                C = 106000 + 8.7 * self.Q
            else:
                C = 110000 + 4.5 * self.Q ** 0.9
        else:
            C = 110000 + 4.5 * Q ** 0.9

        if self.installed:
            C = self.lang(C, Capital_factors)

        return C


class Pump(Equipment):
    def __init__(self, Q, category=Category.fluids, fm=1, installed=True):
        super().__init__(category, fm, installed)
        self.Q = Q

    def pump(self):
        if self.Q < 0.2 or self.Q > 126:
            print(f"    - WARNING: pump caudal out of method bounds, 0.2 < Q (L/s) < 126. Results may not be accurate.")

        C = 6900 + 206 * self.Q ** 0.9

        if self.installed:
            C = self.lang(C, Capital_factors)

        return C


class SteamTurbine(Equipment):
    def __init__(self, kw, category=Category.fluids, fm=1, installed=True):
        super().__init__(category, fm, installed)
        self.kw = kw

    def steam_turbine(self):
        if self.kw < 100 or self.kw > 20000:
            print(
                f"    - WARNING: steam turbine power out of method bounds, 100 < kW < 20000. Results may not be accurate.")

        C = -12000 + 1630 * self.kw ** 0.75

        if self.installed:
            C = self.lang(C, Capital_factors)

        return C


class Loan:
    def __init__(self, quantity, interest, years):
        self.quantity = quantity
        self.interest = interest
        self.years = years

    def loan_payment(self):
        assert self.quantity > 0
        assert self.interest >= 0 and self.interest <= 1
        assert self.years > 1
        return pmt(self.interest, self.years, self.quantity)

    def loan_interest(self):
        assert self.quantity > 0
        assert self.interest >= 0 and self.interest <= 1
        assert self.years > 1
        return ipmt(self.interest, np.arange(self.years) + 1, self.years, self.quantity)

    def loan_principal(self):
        assert self.quantity > 0
        assert self.interest >= 0 and self.interest <= 1
        assert self.years > 1
        return ppmt(self.interest, np.arange(self.years) + 1, self.years, self.quantity)


class EconomicAnalysis:
    def __init__(self, capex, water, salaries, loan=None, capacity_factor=1):
        self.capex = capex
        self.capacity_factor = capacity_factor
        self.water = water
        self.salaries = salaries
        self.loan = loan


if __name__ == '__main__':
    print(Capital_factors)

    boiler = Boiler(10000, 70).boiler()
    turbine = SteamTurbine(1500).steam_turbine()
    condenser = Equipment(Category.fluids).william(400000, 10000, 15000, 0.8)
    pump = Pump(2.84).pump()

    print(boiler + turbine + condenser + pump)

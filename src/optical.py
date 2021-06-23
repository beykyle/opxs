#! /usr/bin/python3

import numpy as np
from matplotlib import pyplot as plt
from nuclide import Nuclide

"""
This module provides simple optical potentials
"""

class simpleWoodSaxon:
    def __init__(self, V1 : float, V2 : float, V3: float, nuc : Nuclide):
        self.coeffs = np.array([V1, V2, V3])
        self.nuc = nuc

    def V(self, E : float, r : float):
        V0 = self.coeffs[0] - self.coeffs[1] * E \
           - (1 - 2 * self.nuc.Z / self.nuc.A) * self.coeffs[2]
        return -V0 / ( 1 + np.exp((r - self.nuc.R)/self.nuc.a))

    def plot(self,E,r):
        V = self.V(E,r)
        plt.plot(r, V)
        plt.xlabel("$r$ [fm]")
        plt.ylabel(r"$V(r)$ [MeV]")
        plt.tight_layout()
        plt.show()

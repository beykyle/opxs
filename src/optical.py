#! /usr/bin/python3

import numpy as np

"""
This module provides simple optical potentials
"""

class simpleWoodSaxon:
    def __init__(V1 : float, V2 : float, V3: float, nuc : Nuclide):
        self.V = np.array([V1, V2, V3])
        self.nuc = nucs

    def V(E : float, r : float):
        V0 = self.V[0] - self.V[1] * E - (1 - 2 * self.nuc.Z / self.nuc.A) * self.V[2]
        return -V0 / ( 1 + np.exp((r - self.nuc.R)/self.nuc.a)



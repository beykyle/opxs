#! /usr/bin/python3

import numpy as np

"""
This module provides simple data structures from storing nuclear and projectile information used in optical calculations
"""

amu2eV = 9.3149102E8          # eV/(amu * c^2)
hbar   = 4.136E-15/(2*np.pi)  # eV s
c      = 2.998E23             # fm/s

class Projectile:
    def __init__(self, A : int):
        self.A = A # mass in amu

class Nuclide:
    def __init__(self, A : int , Z : int, r0=1.17, a=0.75):
        self.A = A
        self.Z = Z
        self.R = r0 * np.power(A,1./3.) # nuclear radisu [fm]
        self.a = a # nuclear skin depth [fm]

def reducedMass_eV(proj : Projectile, target : Nuclide):
    return (proj.A * target.A)/(proj.A + target.A) * amu2eV

def reducedh2m(proj : Projectile, target : Nuclide):
    m_reduced = reducedMass_eV(proj,target)
    return hbar*hbar/(2*m_reduced) * c * c

#! /usr/bin/python3

import numpy as np
import sys
sys.path.append("/home/beykyle/umich/nuclide-data")
import nuclide_data



"""
This module provides simple data structures from storing nuclear and projectile information used in optical calculations
"""

amu2eV = 9.3149102E8          # eV/(amu * c^2)
hbar   = 4.136E-15/(2*np.pi)  # eV s
c      = 2.998E23             # fm/s
e      = 1.439964548          # MeV-fm

class Projectile():
    def __init__(self, A : int, Z = 0):
        self.A   = A
        self.Z   = Z
        self.m   = Nuclide(A,Z).mass()
        self.Jpi = Nuclide(A,Z).spin()

class Neutron(Projectile):
    def __init__(self):
        self.A    = 1
        self.Z    = 0
        self.m    = 1.008665
        self.Jpi  = 1/2,True

class Proton(Projectile):
    def __init__(self):
        self.A    = 0
        self.Z    = 1
        self.m    = 1.0072764
        self.Jpi  = 1/2,True

class Nuclide(Projectile):
    def __init__(self, A : int , Z : int, r0=1.17, a=0.75):
        self.A = A
        self.Z = Z
        self.m   = self.mass()
        self.Jpi = self.spin()
        self.R = r0 * np.power(A,1./3.) # nuclear radius [fm]
        self.a = a                      # nuclear skin depth [fm]

    """ Semi-empirical binding energy in Mev """
    def binding(self, Av=15.5, As=16.8, Ac=0.72, Asym=23, Ap=34):
        # default values from Krane, 1988
        N = self.A + self.Z
        delta = Ap * self.A**(3./4.)
        if (self.A%2 != 0):
            delta = 0
        elif ((N%2 != 0) and (self.Z%2 != 0)):
            delta *= -1.

        return   Av   * self.A \
               - As   * self.A**(2./3.) \
               - Ac   * self.Z * (self.Z - 1) * self.A**(1./3.) \
               - Asym * (self.A - 2*self.Z)**2/self.A \
               + delta

    """ mass in amu """
    def mass(self):
        return Proton().m * self.Z + Neutron().m * self.A - self.binding() * 1E6 / amu2eV

    def compound(self, proj : Projectile):
        return Nuclide(self.A + proj.A, self.Z + proj.Z)

    """ returns (J,parity) in ground state from shell model. pi = (True,False) for (even,odd) """
    def spin(self):
        ## TODO pull in nuclide_data
        return 0, True

def reducedMass_eV(proj : Projectile, target : Nuclide):
    return (proj.m * target.m)/(proj.m + target.m) * amu2eV

def reducedh2m_fm2(proj : Projectile, target : Nuclide):
    m_reduced = reducedMass_eV(proj,target)
    return hbar*hbar/(2*m_reduced) * c * c

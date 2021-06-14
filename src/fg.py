#! /usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

"""
This module numerically solves the radial Schroedinger's Equation in an arbitrary central potential using a finite differencing scheme

TODO:
    - test with 0 potential for Bessel function
    - test with real potentials
    - extend to complex potentials
    - test with Koning-Delaroche optical model potentials
"""

def solve(l : int, E : float, mh2 : float, V : np.array, r : np.array):
    grid_size = V.size
    assert(r.size == grid_size)
    deltaR = r[1:] - r[:-1]
    h = deltaR[0]

    # set up potentials
    w      = (E - V) * mh2  - l * (l + 1) / r**2
    w[0] = 0 # TODO numerical trick

    # set up wavefuncton
    u = np.zeros(grid_size)
    u[0] = 0
    u[1] = 1

    # finite difference: Fox-Goodwin scheme O(deltaR^4)
    k = h**2/12
    for i in range(1,grid_size-1):
        u[i+1] = (2*u[i] - u[i-1] - k * (10 * w[i] * u[i] + w[i-1] * u[i-1])) /\
                   (1 + k * w[i+1] )

    return u / np.sqrt(np.sum(u*u))


def AlphaTest():
    grid_sz = 10000
    Rmax    = 6 #fm
    mph2    = 1.0/10.375E6 # 2 * alpha_mass/hbar^2 [eV fm^2]
    r    = np.linspace(0,Rmax,grid_sz)
    V0      = 122.694E6 # eV
    beta    = 0.22 #fm
    V    = -V0 * np.exp(-beta * r * r)
    # plot potential
    plt.plot(r, V)
    plt.xlabel("$r$ [fm]")
    plt.ylabel(r"$V(r)$ [eV]")
    plt.tight_layout()
    plt.show()
    l = 1 # angular momentum quantum number
    u    = np.zeros(grid_sz)
    E    = -76.9036145E6
    u    = solve(l,0,mph2,V,r)
    print("Found bound state! Energy: {:1.6e} eV".format(E))
    plt.plot(r, u, label=r"$\psi_{}(r)$".format(l))
    plt.plot(r, u*u/np.max(u), label=r"$|\psi_{}(r)|^2$".format(l))
    plt.xlabel("$r$ [fm]")
    plt.legend()
    plt.tight_layout()
    plt.show()

def BesselTest():
    pass

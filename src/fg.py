#! /usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from scipy import special as sc
from nuclide import Nuclide, reducedh2m

"""
This module numerically solves the radial Schroedinger's Equation in an arbitrary central potential using a finite differencing scheme

TODO:
    - test with 0 potential for Bessel function
    - test with real potentials
    - extend to complex potentials
    - test with Koning-Delaroche optical model potentials
"""

def solve(l : int, E : float, h2m : float, V : np.array, r : np.array, u : np.array):
    grid_size = V.size
    assert(r.size == grid_size)
    assert(u.size == grid_size)
    deltaR = r[1:] - r[:-1]
    h = deltaR[0]

    # set up potentials
    w      = (E - V) / h2m  - l * (l + 1) / r**2
    w[0] = 0 # TODO numerical trick

    # finite difference: Fox-Goodwin scheme O(deltaR^4)
    k = h**2/12
    for i in range(1,grid_size-1):
        u[i+1] = (2*u[i] - u[i-1] - k * (10 * w[i] * u[i] + w[i-1] * u[i-1])) /\
                   (1 + k * w[i+1] )

    return u / np.sqrt(np.sum(u*u))

def plotPotential(r,V):
    # plot potential
    plt.plot(r, V)
    plt.xlabel("$r$ [fm]")
    plt.ylabel(r"$V(r)$ [eV]")
    plt.tight_layout()
    plt.show()

def VAlpha(r , V0=122.694E6, beta=0.22):
    return -V0 * np.exp(-beta * r * r)

def AlphaTest():
    grid_sz = 10000
    Rmax    = 6.00 # fm
    r       = np.linspace(0,Rmax,grid_sz)
    proj    = Nuclide(4,2)
    targ    = Nuclide(4,2)
    h2m     = reducedh2m(proj, targ)
    h2m     = 10.375E6
    V0      = 122.694E6 # eV
    beta    = 0.22 #fm
    V       = VAlpha(r)

    # set up wavefuncton
    u = np.zeros(grid_sz)
    u[0] = 0 # repulsive
    u[1] = 1


    # solve for multiple angular momenta
    for l in range(0,1):
        E = -76.9036145E6 #TODO
        u = solve(l,E,h2m,V,r,u)
        plt.plot(r, u, label=r"$\psi_{}(r)$".format(l))

    # plot resulting wavefunction
    #plt.plot(r, u*u/np.max(u), label=r"$|\psi_{}(r)|^2$".format(l))
    plt.xlabel("$r$ [fm]")
    plt.legend()
    plt.tight_layout()
    plt.show()

def BesselTest():
    grid_sz = 1000
    Rmax    = 6.00 # fm
    r       = np.linspace(0,Rmax,grid_sz)
    proj    = Nuclide(4,2)
    targ    = Nuclide(4,2)
    h2m     = reducedh2m(proj, targ)
    V0      = 1E15 # potential well w/ depth 1E9 eV
    V       = np.zeros(grid_sz)
    #V[-1]   = V0

    #plotPotential(r,V)
    # set up wavefuncton
    u = np.zeros(grid_sz)
    u[0] = 1
    u[1] = 1

    # solve for multiple angular momenta
    lMax  = 1
    nperl = 2
    jn_zeros_sph = { 0 : [3.14, 6.28], 1 : [4.49, 7.73] }
    for l in range(0,lMax+1):
        allowed_k = np.array(jn_zeros_sph[l])/Rmax
        for n, k in enumerate(allowed_k):
            E =  h2m * k**2
            u = solve(l,E,h2m,V,r,u)
            plt.plot(r, u, label=r"$| {}{} \rangle$".format(n+1,l))

    # plot resulting wavefunction
    #plt.plot(r, u*u/np.max(u), label=r"$|\psi_{}(r)|^2$".format(l))
    plt.xlabel("$r$ [fm]")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Running 0 potential for Bessel function solutions")
    BesselTest()
    print("Running alpha-alpha reaction example from TPOTPC, ch. 13")
    AlphaTest()

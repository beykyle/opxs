#! /usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from scipy import special as sc
from nuclide import Nuclide, Projectile, reducedh2m_fm2
from fg import solve
import optical

"""
This module calculates cross sections for neutral particles incident on spherically symmmetric potentials
"""

"""
Calculates a non-relativistic differential scattering cross section [fm^2] for a
neutral projectile incident on a spherically-symmetric Woods-Saxon optical potential
@param E_inc  incident projectile energy in COM frame [MeV]
@param mu     grid of cos(theta) in the COM frame over which to determine xs (on [-1,1])
@param pot    potential function for target-projectile pair, as a function of the distance
              between them. Object should be callable with a distance in fm. Should return a
              value in MeV.
"""
def neutralProjXS(target : Nuclide, proj : Projectile, pot,
                  E_inc : float, mu : np.array,
                  grid_size = 10000 , lmax = 30):
    assert(proj.Z == 0)
    # parameters and dimensional analysis
    h2m = reducedh2m_fm2(proj, target)/1E6
    print(h2m)
    k   = np.sqrt(E_inc/h2m)

    # set up radial and potential for the internal region
    Rmax = target.R
    r  = np.linspace(0,3*Rmax,grid_size)
    dr = r[-1] - r[-2]
    V  = pot(r)

    # for each angular momentum eigenstate
    u = np.zeros(grid_size)
    u[1] = 1
    delta = np.zeros(lmax+1)
    sig_s = 0
    for l in range(0,lmax+1):
        # solve Schroedinger's equation in the internal region
        u = solve(l, E_inc, h2m, V, r, u)
        # stitch together internal wvfxn with external analytic solution
        A1 = (Rmax - dr) * sc.spherical_jn(l, k * (Rmax -dr)) * u[-1] \
           - Rmax * sc.spherical_jn(l, k * Rmax) * u[-2]
        A2 = (Rmax - dr) * sc.spherical_yn(l, k * (Rmax -dr)) * u[-1] \
           - Rmax * sc.spherical_yn(l, k * Rmax) * u[-2]
        # calculate phase shift
        delta[l]  = np.arctan2(A1,A2)
        sig_s += (2*l +1) * np.sin(delta[l])**2


    # matrix element S_l = f_l(mu) is the scattering matrix element
    # for angular momentum quantum number l in a partial wave expansion
    # sum series expansion over l in {0,...,lmax}
    Refmu = 1/(2*k) * np.polynomial.legendre.legval(mu, (2*l+1)*np.sin(2 * delta) )
    Imfmu = 1/(k)   * np.polynomial.legendre.legval(mu, (2*l+1)*np.sin(delta)**2  )

    # return complex square of matrix element, and optical total xs
    return Refmu**2 + Imfmu**2, 4*np.pi/k**2 * sig_s

def test_dSigdmu():

    E_inc = 2.776 # MeV
    A     = 56
    Z     = 26
    lmax  = 5

    target    = Nuclide(A,Z)
    neutron   = Projectile(1.008665)
    potential = optical.simpleWoodSaxon(56.3, 0.32, 24.0, target)
    mu, w     = np.polynomial.legendre.leggauss(500)
    #potential.plot(E_inc, np.linspace(0,3*target.R,500))

    # run xs calculation
    dSigdMu, sig = neutralProjXS(
            target, neutron,
            lambda r : potential.V(E_inc,r),
            E_inc, mu, lmax=lmax)

    sig *= 0.01
    # plot result
    plt.semilogy(mu,0.01*dSigdMu, label=r"$n + Fe_{56}^{26} \rightarrow n + Fe_{56}^{26}$  "
            + ", E={E:.4f} [MeV]".format(E=E_inc))
    plt.xlabel(r"$\mu$")
    plt.ylabel(r"$\frac{d\sigma}{d\mu}$ [b]")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("Total xs series sum:     {} [b]".format(sig))
    print("Total xs G_L inegration: {} [b]".format(np.dot(w,dSigdMu)))


if __name__ == "__main__":
    test_dSigdmu()

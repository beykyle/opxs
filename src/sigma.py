#! /usr/bin/python3

import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy import special as sc
from nuclide import Nuclide, Projectile, reducedh2m_fm2
from fg import solve
import optical

#debug
import pdb

"""
This module calculates cross sections for neutral particles incident on spherically symmmetric potentials
"""
def plotDiffXS(dSigdMu,mu, label):
    plt.semilogy(mu,dSigdMu, label=label)
    plt.xlabel(r"$\cos{\left( \theta \right)}$")
    plt.ylabel(r"$\frac{d\sigma}{d\mu}$ [mb]")
    plt.legend()
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.show()

def plotDiffXSDeg(dSigdMu,mu, label):
    plt.semilogy(np.arccos(mu)*180/(np.pi),dSigdMu, label=label)
    plt.xlabel(r"$\theta^\degree$")
    plt.ylabel(r"$\frac{d\sigma}{d\mu}$ [mb]")
    plt.legend()
    plt.tight_layout()
    plt.show()

"""
Calculates a non-relativistic differential scattering cross section [fm^2] for a
neutral projectile incident on a spherically-symmetric Woods-Saxon optical potential
@param E_inc  incident projectile energy in COM frame [MeV]
@param mu     grid of cos(theta) in the COM frame over which to determine xs (on [-1,1]).
@param w      quadrature weights for the grid over mu
@param pot    potential function for target-projectile pair, as a function of the distance
              between them. Object should be callable with a distance in fm. Should return a
              value in MeV.
"""
def neutralProjXS(target : Nuclide, proj : Projectile, pot,
        E_inc : float, mu : np.array, w : np.array,
                  grid_size = 1000 , lmax = 30):

    # neutral particles only
    assert(proj.Z == 0)

    # parameters and dimensional analysis
    h2m = reducedh2m_fm2(proj, target)/1E6 # MeV fm^2
    k   = np.sqrt(E_inc/h2m) # 1/fm

    # set up radial and potential grids for the internal region
    Rmax = 3*target.R
    r  = np.linspace(0,Rmax,grid_size)
    dr = r[-1] - r[-2]
    V  = pot(r)

    # radial grid for internal wvfxn
    u = np.zeros(grid_size)
    u[1] = 1

    # grid over Legendre moments
    delta  = np.zeros(lmax+1)
    l_grid = np.arange(0,lmax+1,1)

    # for each angular momentum eigenstate
    sig_s = 0 # tally up contributions from each eigenstate
    for l in l_grid:
        # solve Schroedinger's equation in the internal region
        u = solve(l, E_inc, h2m, V, r, u)
        # stitch together internal wvfxn with external analytic solution
        A1 = (Rmax - dr) * sc.spherical_jn(l, k * (Rmax - dr)) * u[-1] \
           - Rmax * sc.spherical_jn(l, k * Rmax) * u[-2]
        A2 = (Rmax - dr) * sc.spherical_yn(l, k * (Rmax - dr)) * u[-1] \
           - Rmax * sc.spherical_yn(l, k * Rmax) * u[-2]
        # calculate phase shift
        delta[l]  = np.arctan2(A1,A2)
        sig_s += (2*l +1) * np.sin(delta[l])**2


    # partial wave expansion gives us the matrix element S(mu) as a Legendre series
    Refmu = 1/(2*k) * np.polynomial.legendre.legval(mu, (2 * l_grid + 1)*np.sin(2 * delta) )
    Imfmu = 1/(k)   * np.polynomial.legendre.legval(mu, (2 * l_grid + 1)*np.sin(delta)**2  )

    # dSigma/dmu = complex square of matrix element
    # factor fo 2pi comes fro integrating over azimuth (symmetric)
    Smu       = 2*np.pi*( Refmu**2 + Imfmu**2)

    # total xs (Gauss-Legendre integration over Smu)
    sig_s_GL  = np.dot(Smu,w)

    # total xs (Griffths 11.48 - direct series sum)
    sig_s_SS  = 4*np.pi/k**2 * sig_s

    # total xs (TPOPC 19.21, Optical theorem)
    # extrapolate Imfmu to mu=1 to get forward scattering amplitude
    slope =  (Imfmu[-1] - Imfmu[-2])/(mu[-1] - mu[-2])
    sig_s_opt = 4*np.pi/k * (Imfmu[-1] + slope * (1 - mu[-1]))

    return Smu, sig_s_GL, sig_s_SS, sig_s_opt


def test_dSigdmu(E):

    # parameters
    E_inc = E # MeV
    A     = 56
    Z     = 26
    lmax  = 30

    target    = Nuclide(A,Z)
    neutron   = Projectile(1.008665)
    potential = optical.simpleWoodSaxon(56.3, 0.32, 24.0, target)
    mu, w     = np.polynomial.legendre.leggauss(500)
    #potential.plot(E_inc, np.linspace(0,3*target.R,500))

    # run xs calculation
    dSigdMu, sig_GL, sig_ss, sig_opt = neutralProjXS(
            target, neutron,
            lambda r : potential.V(E_inc,r),
            E_inc, mu, w, lmax=lmax)

    # convert xs to milibarns
    sig_ss   *= 10
    sig_opt  *= 10
    sig_GL   *= 10
    dSigdMu  *= 10

    # plotting and output
    #plotDiffXS(dSigdMu,mu,
    #        r"$n + Fe_{56}^{26} \rightarrow n + Fe_{56}^{26}$  "
    #        + ", E={E:1.3e} [MeV]".format(E=E_inc))

    plotDiffXSDeg(dSigdMu,mu,
            r"$n + Fe_{56}^{26} \rightarrow n + Fe_{56}^{26}$  "
            + ", E={E:1.3e} [MeV]".format(E=E_inc))

    print("\nFor incident neutron energy: {:1.3e}".format(E_inc))
    print("Total xs series sum:     {:1.4e} [mb]".format(sig_ss))
    print("Total xs optical:        {:1.4e} [mb]".format(sig_opt))
    print("Total xs G_L inegration: {:1.4e} [mb]\n".format(sig_GL))

def test_SigE():
    A     = 56
    Z     = 26
    lmax  = 58

    Egrid_sz = 500
    Egrid    = np.logspace(1,2.5,Egrid_sz)

    target    = Nuclide(A,Z)
    neutron   = Projectile(1.008665)
    potential = optical.simpleWoodSaxon(56.3, 0.32, 24.0, target)
    mu, w     = np.polynomial.legendre.leggauss(200)

    sig_ss_E  = np.zeros(Egrid_sz)
    sig_GL_E  = np.zeros(Egrid_sz)
    sig_opt_E = np.zeros(Egrid_sz)

    # calculate total xs over energy
    for i, E in enumerate(Egrid):
        # run xs calculation
        dSigdMu, sig_GL, sig_ss, sig_opt = neutralProjXS(
                target, neutron,
                lambda r : potential.V(E,r),
                E, mu, w, lmax=lmax)
        sig_ss_E[i]  = sig_ss*10
        sig_opt_E[i] = sig_opt*10
        sig_GL_E[i]  = sig_GL*10

    #plt.loglog(Egrid, sig_ss_E, label="Series Sum")
    plt.loglog(Egrid, sig_opt_E, label="Optical Theorem")
    #plt.loglog(Egrid, sig_GL_E, label="Gauss-Legendre")
    plt.ylabel(r"$\sigma$ [mb]")
    plt.xlabel(r"$E$ [MeV] - COM Frame")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # reproduce figs form TPOPC ch.19
    test_dSigdmu(2.776) # 19.10
    test_dSigdmu(10)    # 19.8
    test_dSigdmu(50)    # 19.9
    test_dSigdmu(1.7091E2)    # weird dip in xs(E)
    test_SigE()

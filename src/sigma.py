#! /usr/bin/python3

import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy import special as sc
from nuclide import *
from fg import solve
import optical
from fractions import Fraction
from sympy.physics.quantum.spin import *

#debug
import pdb

"""
This module calculates cross sections for neutral particles incident on spherically symmmetric potentials
"""

""" assumes mu in [-1,1] """
def getSrPerBin(mu):
    sr_per_bin = np.zeros(mu.shape)
    sr_per_bin[0]  = 2*np.pi*(mu[0] +1)
    sr_per_bin[1:] = 2*np.pi*(mu[1:] - mu[:-1])
    return sr_per_bin

def plotDiffXS(dSigdMu,mu, label):
    plt.semilogy(mu, dSigdMu/getSrPerBin(mu), label=label)
    plt.xlabel(r"$\cos{\left( \theta \right)}$")
    plt.ylabel(r"$\frac{d\sigma}{d\mu}$ [mb/sr]")
    plt.legend()
    plt.gca().invert_xaxis()

def plotDiffXSDeg(dSigdMu,mu, label):
    sr_per_bin = getSrPerBin(mu)
    jacobian   = np.sin(np.arccos(mu))
    plt.semilogy(np.arccos(mu)*180/(np.pi),dSigdMu*jacobian/sr_per_bin, label=label)
    plt.xlabel(r"$\theta^\degree$")
    plt.ylabel(r"$\frac{d\sigma}{d\mu}$ [mb/sr]")
    plt.legend()

"""
Calculates a non-relativistic differential scattering cross section [fm^2] for a
neutral projectile incident on a spherically-symmetric real-valued potential.
Also calculates the total xs 3 ways for debugging.
@param E_inc  incident projectile energy in COM frame [MeV]
@param mu     grid of cos(theta) in the COM frame over which to determine xs (on [-1,1]).
@param w      quadrature weights for the grid over mu
@param pot    potential function for target-projectile pair, as a function of the distance
              between them. Object should be callable with a distance in fm. Should return a
              value in MeV.
"""
def neutralProjRealPotXS(target : Nuclide, proj : Projectile, pot,
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

    # grids over Legendre moments
    l_grid = np.arange(0,lmax+1,1)
    delta  = np.zeros(lmax+1) # scattering phase shift

    # for each angular momentum eigenstate
    sig_s = 0 # tally up contributions from each eigenstate
    for l in l_grid:
        # solve Schroedinger's equation in the internal region
        u = solve(l, E_inc, h2m, V, r, u)

        # stitch together internal numeric wvfxn with external analytic solution
        A1 = (Rmax - dr) * sc.spherical_jn(l, k * (Rmax - dr)) * u[-1] \
           - Rmax * sc.spherical_jn(l, k * Rmax) * u[-2]
        A2 = (Rmax - dr) * sc.spherical_yn(l, k * (Rmax - dr)) * u[-1] \
           - Rmax * sc.spherical_yn(l, k * Rmax) * u[-2]

        # calculate phase shift
        delta[l]  = np.arctan2(A1,A2)

        # tally total xs contribution
        sig_s    += (2*l +1) * np.sin(delta[l])**2


    # partial wave expansion gives us the matrix element S(mu) as a Legendre series
    # take Re and Im parts of series sum
    Refmu = 1/(2*k) * np.polynomial.legendre.legval(mu, (2 * l_grid + 1)*np.sin(2 * delta) )
    Imfmu = 1/(k)   * np.polynomial.legendre.legval(mu, (2 * l_grid + 1)*np.sin(delta)**2  )

    # Diff xs is complex square of matrix element: dSigma/dOmega = 2pi Smu
    # factor of 2pi comes from integrating over azimuth (symmetric)
    Smu = 2*np.pi*( Refmu**2 + Imfmu**2)

    # total xs (Gauss-Legendre integration over Smu)
    sig_s_GL  = np.dot(Smu,w)

    # total xs (Griffths 11.48 - direct series sum)
    sig_s_SS  = 4*np.pi/k**2 * sig_s

    # total xs (TPOPC 19.21, Optical theorem)
    # extrapolate Imfmu to mu=1 to get forward scattering amplitude
    slope =  (Imfmu[-1] - Imfmu[-2])/(mu[-1] - mu[-2])
    sig_s_opt = 4*np.pi/k * (Imfmu[-1] + slope * (1 - mu[-1]))

    return Smu, sig_s_GL, sig_s_SS, sig_s_opt


"""
Calculates a non-relativistic differential scattering cross section [mb] for an
arbitrary projectile incident on a spherically-symmetric complex potential.
@param pot    An Optical Model Potenrtial (OMP), with function V(r,E,ket) returning a complex
              np array for the potiential at incident energy E [MeV], over distance np array r [fm],
              and with SpinState ket
@param E_inc  incident projectile energy in COM frame [MeV]
@param mu     grid of cos(theta) in the COM frame over which to determine xs (on [-1,1]).
@param w      quadrature weights for the grid over mu
@param pot    potential function for target-projectile pair, as a function of the distance
              between them. Object should be callable with a distance in fm. Should return a
              value in MeV.
"""
def xs(target : Nuclide, proj : Projectile, pot,
       E_inc : float, mu : np.array, w : np.array,
       grid_size = 100 , lmax = 30, tol=1E-10, plot=False):

    # parameters and dimensional analysis
    h2m = reducedh2m_fm2(proj, target)/1E6  # MeV fm^2
    k   = np.sqrt(E_inc/h2m)                # 1/fm - external wavenumber

    # set up radial and potential grids for the internal region
    Rmax = 3*target.R
    r  = np.linspace(0,Rmax,grid_size)
    dr = r[-1] - r[-2]

    # spin of projectile
    s,_  = proj.Jpi

    # grids over Legendre moments
    l_grid = np.arange(0,lmax+1,1, dtype="int")
    S      = np.zeros(lmax+1, dtype='cdouble') # S-matrix element

    # for each angular momentum eigenstate
    sig_s = 0 # tally up contributions from each eigenstate
    for l in l_grid:
        # start with 0 matrix elements
        S_l = 0
        # TODO just take 1 spin for xs, assume polarized beam
        # let averaging happen externally
        # TODO use sympy.physics.quantum.spin to generalize
        # determine allowed total projectile angular momentum j
        # (eigenvalues of vector addition of l and s)
        lower = -s
        if (s>l):
            lower = s
        sgrid = np.arange(lower,s+1,1)
        jgrid = sgrid + l
        num_spins = jgrid.size
        for j in jgrid:

            # initialize potential
            ket = optical.SpinState(j,l,s)
            V = pot.V(E_inc,r,ket)

            # initialize interior wavefunction arrays
            u     = np.zeros(V.shape,dtype="cdouble")
            u[1]  = np.complex(1,1)

            # solve Schroedinger's equation in the internal region
            u = solve(l, E_inc, h2m, V, r,u)
            if plot:
                plotPsi(u,r,l,j)

            # match internal and external wavefunctions
            # use linear interpolation, with matching radius halfway between
            # last and second to last radial bin
            # TODO use Lagrange interpolation, match exactly at exterior
            rmatch = r[-1] - dr*0.5
            u0     = (u[-1] + u[-2])*0.5  # u_int(r_match)
            u1     = (u[-1] - u[-2])/dr   # d/dr u_int(r) |_{r=r_match}

            #TODO use Coulomb wavefxns instead of Bessel/Hankel below
            # use mpmath module
            # they reduce to this form for neutral projectiles

            # spherical Bessel function of 1st kind of order l, and derivative
            jkl  = sc.spherical_jn(l,k*rmatch)
            djkl = sc.spherical_jn(l,k*rmatch, derivative=True)

            # spherical Hankel function of 1st kind of order l, and derivative
            hkl  = sc.spherical_jn(l,k*rmatch) + 1j * sc.spherical_yn(l,k*rmatch)
            dhkl = sc.spherical_jn(l,k*rmatch,derivative=True) \
                 + 1j * sc.spherical_yn(l,k*rmatch, derivative=True)

            # solve the system:
            # u_ext(r_match) = u_int(rmatch)
            # d/dr u_ext(r)|_{r+rmatch} = d/dr u_int(r) |_{r+r_match}
            # for partial wave amplitude a_{l,s} in u_ext (by eliminating normalization factor A)
            a_ls = (1j/k) * (jkl * u1 - djkl * u0)/(hkl * u1 - dhkl * u0)

            # calculate the S matrix element and transmission coefficients
            S_l  += a_ls

        # record lth matrix element contribution to total differential cross section
        # by averaging over incoming spin states (assume un-polarized beam)
        S[l] = S_l / num_spins

        # break loop if relative change in S dips below a tolerance
        S2 = (S[l] * S[l].conj()).real
        print("|S_{}|^2: {}".format(l,S2))
        if ((l > 2)
            and
            ((S[l-1] * S[l-1].conj() - S2)/(S[l-1] * S[l-1].conj()) < tol )):
            print("max l: {}".format(l))
            break

    # now we have a matrix element S_l for each Legendre moment l
    # we can reconstruct our differential cross section like so:
    # d sigma / d theta ~ | Sum_l^lmax S_l P_l(cos(theta)) |^2
    # define the sum f(theta) =  Sum_l^lmax S_l P_l(cos(theta))
    # let's calculate the real and imaginary parts in turn:
    Reftheta = np.polynomial.legendre.legval(mu,(2*l_grid + 1)*S.real)
    Imftheta = np.polynomial.legendre.legval(mu,(2*l_grid + 1)*S.imag)

    # and now take the complex square to get the differential xs
    # - integrate over azimuth (factor of 2pi)
    # - convert from [fm^2] to [mb] (factor of 10)
    dSigdMu = (Reftheta**2 + Imftheta**2) * 10 * 2 * np.pi # [mb]
    return dSigdMu

def plotPsi(u,r,l,j):
    plt.title(r"$|l,j\rangle = |{},{}\rangle$".format(l,Fraction(j)))
    rho = (u.real**2 + u.imag**2)
    plt.plot(r,u.real/np.sum(u.real),label=r"Re[$\psi$]")
    plt.plot(r,u.imag/np.sum(u.imag),label=r"Im[$\psi$]")
    plt.plot(r,rho/np.sum(rho),label=r"$\|\psi\|^2$")
    plt.xlabel(r"$r$ [fm]")
    plt.ylabel(r"$\psi(r)$ [un-normalized]")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plotPot(r,realV,imagV):
    plt.plot(r,realV, label=r"Re($V(r)$)")
    plt.plot(r,imagV, label="Im($V(r)$)")
    plt.xlabel(r"$r$ [fm]")
    plt.ylabel(r"$V(r)$ [MeV]")
    plt.legend()
    plt.tight_layout()
    plt.show()

def test_dSigdmu(E):

    # parameters
    E_inc = E # MeV
    A     = 56
    Z     = 26
    lmax  = 30

    target    = Nuclide(A,Z)
    neutron   = Neutron()
    potential = optical.SimpleWoodSaxon(56.3, 0.32, 24.0, target)
    mu, w     = np.polynomial.legendre.leggauss(500)
    #potential.plot(E_inc, np.linspace(0,3*target.R,500))

    # run xs calculation
    dSigdMu, sig_GL, sig_ss, sig_opt = neutralProjRealPotXS(
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
    plt.tight_layout()
    plt.show()

    print("\nFor incident neutron energy: {:1.3e}".format(E_inc))
    print("Total xs series sum:     {:1.4e} [mb]".format(sig_ss))
    print("Total xs optical:        {:1.4e} [mb]".format(sig_opt))
    print("Total xs G_L inegration: {:1.4e} [mb]\n".format(sig_GL))

def test_SigE():
    A     = 56
    Z     = 26
    lmax  = 58

    Egrid_sz = 500
    Egrid    = np.logspace(-1,2,Egrid_sz)

    target    = Nuclide(A,Z)
    neutron   = Nuetron()
    potential = optical.simpleWoodSaxon(56.3, 0.32, 24.0, target)
    mu, w     = np.polynomial.legendre.leggauss(200)

    sig_ss_E  = np.zeros(Egrid_sz)
    sig_GL_E  = np.zeros(Egrid_sz)
    sig_opt_E = np.zeros(Egrid_sz)

    # calculate total xs over energy
    for i, E in enumerate(Egrid):
        # run xs calculation
        dSigdMu, sig_GL, sig_ss, sig_opt = neutralProjRealPotXS(
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

def realPotTests():
    # reproduce figs form TPOPC ch.19
    test_dSigdmu(2.776) # 19.10
    test_dSigdmu(10)    # 19.8
    test_dSigdmu(50)    # 19.9
    test_dSigdmu(1.7091E2)    # weird dip in xs(E)
    test_SigE()

def cmplPotTests():
    Fe56   = Nuclide(56,26)
    n      = Neutron()
    params = optical.OMPParams(Fe56,n)
    omp    = optical.OMP(Fe56, n, params)
    mu, w  = np.polynomial.legendre.leggauss(5000)
    #E_inc  = [4.6, 5.0, 5.6, 6.5, 7.6, 8.0, 10.0, 11.0, 12.0]
    E_inc  = [4.6]

    factor = 1
    for E in E_inc:
        dSigdMu = xs(Fe56,n,omp,E,mu,w,lmax=50, plot=False)
        plotDiffXSDeg(dSigdMu*factor,mu,r"$E=${0:1.3f} [MeV]".format(E))
        factor *= 0.1

    plt.title(r"$n + Fe_{56}^{26} \rightarrow n + Fe_{56}^{26}$")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #realPotTests()
    cmplPotTests()

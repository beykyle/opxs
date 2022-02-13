#! /usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from scipy import special as sc
from nuclide import Nuclide, reducedh2m_fm2
from scipy.special import genlaguerre as Lag

"""
This module numerically solves the radial Schroedinger's Equation in an arbitrary central potential using a finite differencing scheme

"""
def plot(u,r,l,title):
    plt.plot(r, u, label=r"$l = {}$".format(l))
    plt.xlabel("$r$ [fm]")
    plt.ylabel("$\psi(r)$")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def runTestAndPlot(l,E,h2m,V,r,title,plot_R=False):
    u = np.zeros(V.shape, dtype='cdouble')
    u[0] = np.complex(0,0)
    u[1] = np.complex(1,1)

    u = solve(l,E,h2m,V,r,u)
    if plot_R:
        u = u/r

    plot(u,r,l, title + ": E={:1.2e} [eV]".format(E))

def norm(psi):
    n = np.dot(psi,psi.conj())
    exit()
    return psi/n

"""
Uses the Fox-Goodwin method (TPOTPC Ch. 13) to solve for the radial component of a stationary
angular momentum state of the radial Schroedinger equation. Note the returned array is
u_l(r), where the full radial wavefunction is sum_l u_l(r)/r
"""
def solve(l : int, E : float, h2m : float, V : np.array, r : np.array, u : np.array):
    grid_size = V.size
    assert(r.size == grid_size)
    assert(u.size == grid_size)
    deltar = r[1:] - r[:-1]
    h = deltar[0]

    # set up potentials
    w    = (E - V) / h2m  - l * (l + 1) / r**2
    w[0] = np.complex(0,0)

    # finite difference: fox-goodwin scheme o(deltar^4)
    k = h**2/12
    for i in range(1,grid_size-1):
        u[i+1] = (2*u[i] - u[i-1] - k * (10 * w[i] * u[i] + w[i-1] * u[i-1])) /\
                 (1 + k * w[i+1] )

    return u
    #return norm(u)

def plotPotential(r,V):
    # plot potential
    plt.plot(r, V)
    plt.xlabel("$r$ [fm]")
    plt.ylabel(r"$V(r)$ [eV]")
    plt.tight_layout()
    plt.show()

def VAlpha(r , V0=122.694E6, beta=0.22):
    return -V0 * np.exp(-beta * r * r)

def HOTest():
    # physical constants and parameters
    hbar = 1 # if you're not using natural units, get a life
    m = 1
    omega = 1
    h2m = hbar/(2*m)
    # if n is even l is on {0,2,...,n-2,n}
    # if n is odd  l is on {1,3,...,n-2,n}
    l = 3
    n = 21
    if (n-l)%2 == 0:
        k = (n-l)/2
    else:
        k = 0
    En = hbar * omega * (n + 3/2)

    # radial grid
    # RMAX can't be too big or numerical instabilties will result
    RMAX = 1.5*np.sqrt(2*En/(m*omega)) # extend into forbidden region (E < V)
    NR   = 1000
    r  = np.linspace(0,RMAX,NR)

    # potential
    V = np.zeros(r.shape, dtype="cdouble")
    V.real = 1/2 * m * omega**2 * r**2

    # wavefunction
    u    = np.zeros(V.shape, dtype="cdouble") # complex
    u[1] = np.complex(1,1) # boundary condition

    u =  (solve(l,En,h2m,V,r,u)/r)[1:]

    L = Lag(k,l+1/2)
    analytic = (r**l * np.exp(- m*omega/(2*hbar) * r**2) * L( (2*m*omega)/(2*hbar) * r**2 ))[1:]

    plt.title(r"$|l,n\rangle = |{},{}\rangle$".format(l,n))
    rho = (u.real**2 + u.imag**2)
    plt.plot(r[1:],analytic/np.sum(analytic),label=r"analytic")
    plt.plot(r[1:],u.real/np.sum(u.real) ,'--',label=r"numerical")
    #plt.plot(r,u.imag/np.sum(u.imag),label=r"Im[$\psi$]")
    #plt.plot(r,rho/np.sum(rho),label=r"$\|\psi\|^2$")
    plt.xlabel(r"$r$ [a.u.]")
    plt.ylabel(r"$\psi(r)$ [un-normalized]")
    plt.legend()
    #plt.tight_layout()
    plt.show()

def AlphaTest():
    grid_sz = 10000
    Rmax    = 6.00 # fm
    r       = np.linspace(0,Rmax,grid_sz)
    proj    = Nuclide(4,2)
    targ    = Nuclide(4,2)
    h2m     = 10.375E6
    V0      = 122.694E6 # eV
    beta    = 0.22 #fm
    V       = np.complex(grid_sz)
    V       = VAlpha(r) + 1j * np.zeros(grid_sz)

    #TPOPC 13.7
    runTestAndPlot(0,-76.9036145E6,h2m,V,r,"13.7")

    #13.8
    runTestAndPlot(0,-29.00048E6,h2m,V,r,"13.8")

    r = np.linspace(0,18,grid_sz)
    V       = np.complex(grid_sz)
    V       = VAlpha(r) + 1j * np.zeros(grid_sz)

    # 13.10
    E = 1E6
    l = 0
    runTestAndPlot(0,1E6,h2m,V,r,"13.10")

    # 13.11
    runTestAndPlot(0,20E6,h2m,V,r,"13.11")

    # 13.12
    runTestAndPlot(4,5E6,h2m,V,r,"13.12")

    # 13.13
    runTestAndPlot(10,10E6,h2m,V,r,"13.13")

def BesselTest():
    # this test doesn't work
    grid_sz = 1000
    Rmax    = 6.00 # fm
    r       = np.linspace(0,Rmax*2,grid_sz)
    proj    = Nuclide(4,2)
    targ    = Nuclide(4,2)
    h2m     = reducedh2m_fm2(proj, targ)
    V0      = 1E10 # potential well w/ depth 1E9 eV
    V       = np.zeros(grid_sz,dtype='cdouble')
    V[grid_sz//2:].real = V0

    #plotPotential(r,V)

    # solve for multiple angular momenta
    lMax  = 1
    nperl = 2
    jn_zeros_sph = { 0 : [3.14, 6.28], 1 : [4.49, 7.73] }
    colors = ['r' , 'g', 'b', 'k', 'r', 'g', 'b' 'k']
    i = 0
    #TODO FG produces fp overflow, this test fails
    for l in range(0,lMax+1):
        allowed_k = np.array(jn_zeros_sph[l])/Rmax
        for n, k in enumerate(allowed_k):
            i += 1
            E =  h2m * k**2
            u = np.zeros(V.shape, dtype='cdouble')
            u[0] = np.complex(0,0)
            u[1] = np.complex(1,1)
            u = norm(solve(l,E,h2m,V,r,u)/r)
            u_anal = norm(sc.spherical_jn(l,k*r))
            plt.scatter(r[:grid_sz//2], u[:grid_sz//2],
                    label=r"$| {}{} \rangle$: Fox-Goodwin".format(n+1,l),
                    marker='*', color=colors[i])
            plt.plot(r[:grid_sz//2], u_anal[:grid_sz//2],
                    label=r"$j_{}(k_{}r)$".format(l,n+1),
                    color=colors[i])

    # plot resulting wavefunction
    #plt.plot(r, u*u/np.max(u), label=r"$|\psi_{}(r)|^2$".format(l))
    plt.xlabel("$r$ [fm]")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #print("Running alpha-alpha reaction example from TPOTPC, ch. 13")
    #AlphaTest()

    #print("Bessel test")
    #BesselTest() TODO broken

    print("Running 3D QHO test")
    HOTest()

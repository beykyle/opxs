#! /usr/bin/python3

import sys
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
sys.path.append('/home/beykyle/umich/opxs/src/')
print(sys.path)
import sigma
from nuclide import Nuclide, Projectile, reducedh2m_fm2
import optical

def runCalc(potential, target):
    lmax  = 58

    Egrid_sz = 1000
    Egrid    = np.logspace(-3,2,Egrid_sz)

    neutron   = Projectile(1.008665)
    mu, w     = np.polynomial.legendre.leggauss(200)

    sig_ss_E  = np.zeros(Egrid_sz)
    sig_GL_E  = np.zeros(Egrid_sz)
    sig_opt_E = np.zeros(Egrid_sz)

    # calculate total xs over energy
    for i, E in enumerate(Egrid):
        # run xs calculation
        dSigdMu, sig_GL, sig_ss, sig_opt = sigma.neutralProjXS(
                target, neutron,
                lambda r : potential.V(E,r),
                E, mu, w, lmax=lmax)
        sig_ss_E[i]  = sig_ss*10
        sig_opt_E[i] = sig_opt*10
        sig_GL_E[i]  = sig_GL*10

    return Egrid, sig_opt_E

if __name__ == "__main__":
    A     = 47
    Z     = 107
    target    = Nuclide(A,Z)
    potential = optical.simpleWoodSaxon(56.3, 0.32, 24.0, target)
    #E, sigs = runCalc(potential, target)
    sigs = np.load("./data/Ag107_simpleWS_sigs.npy")
    E = np.load(   "./data/Ag107_simpleWS_E.npy")

    data = pd.read_csv("/home/beykyle/umich/opxs/data/Ag108_nelastic_endf8.csv", sep=";")
    #plt.loglog(Egrid, sig_ss_E, label="Series Sum")
    plt.loglog(E, sigs, label="Optical Model")
    plt.loglog(data.iloc[:,0]/1E6, data.iloc[:,1]*1000, label="ENDF/B-VIII.0, MT=2")
    plt.title(r"$n + Ag^{107} \rightarrow n + Ag^{107}$")
    #plt.loglog(Egrid, sig_GL_E, label="Gauss-Legendre")
    plt.ylabel(r"$\sigma$ [mb]")
    plt.xlabel(r"$E$ [MeV] - COM Frame")
    plt.legend()
    plt.tight_layout()
    plt.show()

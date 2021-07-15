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


class EvaluatedData:
    def __init__(self, label, data):
        self.label = label
        self.data = data

def runCalc(potential, target):
    lmax  = 58

    Egrid_sz = 1000
    Egrid    = np.logspace(-3,2,Egrid_sz)

    neutron   = Neutron()
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

def run_all():
    target    = Nuclide(8,15)
    potential = optical.simpleWoodSaxon(56.3, 0.32, 24.0, target)
    E, sigs_15 = runCalc(potential, target)
    target    = Nuclide(8,16)
    potential = optical.simpleWoodSaxon(56.3, 0.32, 24.0, target)
    E, sigs_16 = runCalc(potential, target)
    target    = Nuclide(8,17)
    potential = optical.simpleWoodSaxon(56.3, 0.32, 24.0, target)
    E, sigs_17 = runCalc(potential, target)
    target    = Nuclide(8,18)
    potential = optical.simpleWoodSaxon(56.3, 0.32, 24.0, target)
    E, sigs_18 = runCalc(potential, target)

    np.save("./data/O18_sigs",sigs_18)
    np.save("./data/O17_sigs",sigs_17)
    np.save("./data/O15_sigs",sigs_15)
    np.save("./data/O16_sigs",sigs_16)
    np.save("./data/E",E)

    xs = {
            15 : sigs_15,
            16 : sigs_16,
            17 : sigs_17,
            18 : sigs_18,
        }

    return E , xs


def plot(E_optical, xs_optical, E_eval, xs_eval, cmap):
    for A in cmap:
        cm_opt  = cmap[A] + "-."
        cm_eval = cmap[A]
        plt.loglog(E_optical, xs_optical[A] ,cm_opt , label="O-{}".format(str(A)))
        plt.loglog(E_eval, xs_eval[A].data , cm_eval, label="O-{}".format(str(A))+": "
                                                            + xs_eval[A].label)

    plt.title(r"$n + O \rightarrow n + O$")
    plt.ylabel(r"$\sigma$ [mb]")
    plt.xlabel(r"$E$ [MeV] - COM Frame")
    plt.legend()
    plt.tight_layout()
    plt.show()

def read_evaluated():
    data = pd.read_csv("/home/beykyle/umich/opxs/data/Oxygen_evaluated_sigs.csv", sep=";")
    E = data.iloc[:,0]/1E6
    s17 = data.iloc[:,1]*1000
    s18 = data.iloc[:,2]*1000
    s15 = data.iloc[:,3]*1000
    s16 = data.iloc[:,4]*1000

    xs = {
            15 : EvaluatedData("TENDL-2019",    s15),
            16 : EvaluatedData("ENDF/B-VIII.0", s16),
            17 : EvaluatedData("ENDF/B-VIII.0", s17),
            18 : EvaluatedData("ENDF/B-VIII.0", s18)
        }



    return E, xs

def readOptical():
    sigs_18 = np.load("./data/O18_sigs.npy")
    sigs_17 = np.load("./data/O17_sigs.npy")
    sigs_15 = np.load("./data/O15_sigs.npy")
    sigs_16 = np.load("./data/O16_sigs.npy")
    E       = np.load("./data/E.npy")

    xs = {
            15 : sigs_15,
            16 : sigs_16,
            17 : sigs_17,
            18 : sigs_18,
        }
    return E,xs

if __name__ == "__main__":

   # E      , xs      = run_all()
    E, xs  = readOptical()
    E_eval , xs_eval = read_evaluated()
    plot(E, xs, E_eval, xs_eval, { 15 : "r", 16 : "k", 17 : "g" , 18 : "b"})


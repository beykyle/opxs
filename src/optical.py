#! /usr/bin/python3

import numpy as np
from matplotlib import pyplot as plt
from nuclide import Nuclide, Projectile, Neutron, e, hbar,c, reducedMass_eV

"""
This module provides simple optical potentials
"""

class DepthModel:
    def V0(self,E:float):
        return 0

""" from (Schmid, 1988, ch. 19) """
class TPAPCDepthModel(DepthModel):
    def __init__(self, V1 : float, V2 : float, V3: float, nuc : Nuclide):
        self.coeffs = np.array([V1, V2, V3])
        self.nuc = nuc

    def V0(self, E : float):
        return self.coeffs[0] - self.coeffs[1] * E \
                  - (1 - 2 * self.nuc.Z / self.nuc.A) * self.coeffs[2]

""" from (A.j. Koning, J.P. Delaroche, 2003) """
class KDVolumeDepth(DepthModel):
    def __init__(self, coeffs, fermi_energy):
        assert(coeffs.size == 4)
        self.v1 = coeffs[0]
        self.v2 = coeffs[1]
        self.v3 = coeffs[2]
        self.v4 = coeffs[3]
        self.Ef = fermi_energy

    def V0(self, E : float):
        theta = E - self.Ef
        return self.v1 * \
                (1 - self.v2 *theta + self.v3 * theta**2 - self.v4 * theta**3)

""" from (A.j. Koning, J.P. Delaroche, 2003) """
class KDComplexVolumeDepth(DepthModel):
    def __init__(self, coeffs, fermi_energy):
        assert(coeffs.size == 2)
        self.w1 = coeffs[0]
        self.w2 = coeffs[1]
        self.Ef = fermi_energy

    def V0(self, E : float):
        theta = E - self.Ef
        return self.w1 * theta**2/(theta**2 + self.w2**2)

""" from (A.j. Koning, J.P. Delaroche, 2003) """
class KDComplexSurfaceDepth(DepthModel):
    def __init__(self, coeffs, fermi_energy):
        assert(coeffs.size == 3)
        self.d1 = coeffs[0]
        self.d2 = coeffs[1]
        self.d3 = coeffs[2]
        self.Ef = fermi_energy

    def V0(self, E : float):
        theta = E - self.Ef
        return self.d1 * theta**2 /(theta**2 + self.d3**2) \
                * np.exp(-self.d2*theta)

""" from (A.j. Koning, J.P. Delaroche, 2003) """
class KDSpinOrbitDepth(DepthModel):
    def __init__(self, coeffs, fermi_energy):
        assert(coeffs.size == 2)
        self.vso1 = coeffs[0]
        self.vso2 = coeffs[1]
        self.Ef = fermi_energy

    def V0(self, E : float):
        theta = E - self.Ef
        return self.vso1 * np.exp(-self.vso2*theta)

""" from (A.j. Koning, J.P. Delaroche, 2003) """
class KDComplexSpinOrbitDepth(DepthModel):
    def __init__(self, coeffs, fermi_energy):
        assert(coeffs.size == 2)
        self.wso1 = coeffs[0]
        self.wso2 = coeffs[1]
        self.Ef = fermi_energy

    def V0(self, E : float):
        theta = E - self.Ef
        return self.wso1 * theta**2 / (theta**2 + self.wso2**2)

## Potential form factors
class WoodSaxon:
    def __init__(self, R : float, a : float, depth_model):
        self.R = R
        self.a = a
        self.depth_model = depth_model

    def V(self, E : float, r : np.array):
        V0 = self.depth_model.V0(E)
        return V0 / ( 1 + np.exp((r - self.R)/self.a))

    def dVdr(self, E : float, r : np.array):
        V0 = self.depth_model.V0(E)
        ex = np.exp((r - self.R)/self.a)
        return - V0 * ex /(self.a * (1 + ex)**2)


# use default nuclide radius and diffusivity
class SimpleWoodSaxon(WoodSaxon):
    def __init__(self, nuc : Nuclide, depth_model):
        self.R = nuc.R
        self.a = nuc.a
        self.depth_model = depth_model

class CoulombPotential:
    def __init__(self, target : Nuclide, proj : Projectile, R : float):
        self.target = target
        self.proj   = proj
        self.R      = R

    def V(self, E : float, r : np.array):
        return np.where(
                (r < self.R),
                self.target.Z * self.proj.Z * e**2  / (2*self.R) * (3 - r**2/self.R**2),
                self.target.Z * self.proj.Z * e**2/r )

## Parameters
class PotParams:
    def __init__(self, R :float, a :float, coeffs : np.array):
        self.R      = R
        self.a      = a
        self.coeffs = coeffs

class OMPParams:
    def __init__(self, target : Nuclide, proj : Projectile):
        self.target    = target
        self.proj      = proj
        self.compound  = target.compound(proj)

        #TODO pull data from map (A,Z) -> params
        # for now, hardcode Fe-56 from (A.j. Koning, J.P. Delaroche, 2003)
        assert(target.A == 56)
        assert(target.Z == 26)
        self.real_vol  = PotParams(1.186*target.A**(1./3.), 0.663,
                np.array([56.8, 0.0071, 0.000019, 7E-9]))
        self.cmpl_vol  = PotParams(1.186*target.A**(1./3.), 0.663, np.array([13.0, 80]))
        self.cmpl_surf = PotParams(1.282*target.A**(1./3.),0.532, np.array([15.3, 0.0211, 10.9]))
        self.real_so   = PotParams(1.0*target.A**(1./3.), 0.58, np.array([6.1, 0.0040]))
        self.cmpl_so   = PotParams(1.0*target.A**(1./3.), 0.58, np.array([-3.1, 160]))
        self.coulomb_radius = target.R ##TODO not specified in KD paper
        self.Ef        = -9.42
        #self.Ef = -0.5 * (target.binding() + self.compound.binding())

class SpinState:
    def __init__(self, j : int, l : int, s : int):
        self.j = j
        self.l = l
        self.s = s
        self.parity = (l%2==0)

    def IdotSigma(self):
        return (self.j*(self.j+1) - self.l*(self.l+1) - self.s*(self.s+1))/2

class OMP:
    def __init__(self, target : Nuclide, proj : Projectile, p : OMPParams):
        self.params   = p
        self.target   = target
        self.proj     = proj

        # set up depth models
        self.real_vol_depth  = KDVolumeDepth(p.real_vol.coeffs, p.Ef)
        self.cmpl_vol_depth  = KDComplexVolumeDepth(p.cmpl_vol.coeffs, p.Ef)
        self.cmpl_surf_depth = KDComplexSurfaceDepth(p.cmpl_surf.coeffs, p.Ef)
        self.real_so_depth   = KDSpinOrbitDepth(p.real_so.coeffs, p.Ef)
        self.cmpl_so_depth   = KDComplexSpinOrbitDepth(p.cmpl_so.coeffs, p.Ef)

        # set up functional forms
        self.real_vol  = WoodSaxon(p.real_vol.R,  p.real_vol.a,  self.real_vol_depth)
        self.cmpl_vol  = WoodSaxon(p.cmpl_vol.R,  p.cmpl_vol.a,  self.cmpl_vol_depth)
        self.cmpl_surf = WoodSaxon(p.cmpl_surf.R, p.cmpl_surf.a, self.cmpl_surf_depth)
        self.real_so   = WoodSaxon(p.real_so.R,   p.real_so.a,   self.real_so_depth)
        self.cmpl_so   = WoodSaxon(p.cmpl_so.R,   p.cmpl_so.a,   self.cmpl_so_depth)
        self.coulomb   = CoulombPotential(target, proj, p.coulomb_radius)
        self.so_factor = (hbar * c/ (reducedMass_eV(proj,target)))**2 # fm^2

    def real(self,E,r, ket : SpinState):
        spin_cpl = ket.IdotSigma() * self.so_factor * self.real_so.dVdr(E,r) / r
        spin_cpl[0] = 0
        return  - self.real_vol.V(E,r) \
                + spin_cpl \
                + self.coulomb.V(E,r)

    def imag(self,E,r, ket : SpinState):
        spin_cpl = ket.IdotSigma() * self.so_factor * self.cmpl_so.V(E,r)
        spin_cpl[0] = 0
        cmpl_surf = self.cmpl_surf.dVdr(E,r) / r
        cmpl_surf[0] = 0
        return  - self.cmpl_vol.V(E,r) \
                + spin_cpl \
                - cmpl_surf

    def V(self,E,r, ket : SpinState):
        return self.real(E,r,ket) + np.complex(0,1) * self.imag(E,r,ket)

    def plot(self,V,E,r,label):
        plt.plot(r, V, label=label)
        plt.xlabel("$r$ [fm]")
        plt.ylabel(r"$V(r)$ [MeV]")

    def plotAll(self,E,r):
        plt.plot(r, -self.real_vol.V(E,r), label=r"$V_v$")
        plt.plot(r, self.coulomb.V(E,r), label=r"$V_C$")
        plt.plot(r, self.so_factor * self.real_so.dVdr(E,r) / r, label=r"$V_{so}$")
        plt.plot(r, self.cmpl_vol.V(E,r), label=r"$W_v$")
        plt.plot(r, self.so_factor * self.cmpl_so.dVdr(E,r) / r, label=r"$W_{so}$")
        plt.plot(r, self.cmpl_surf.dVdr(E,r) / r, label=r"$W_{D}$")
        plt.xlabel(r"$r$ [fm]")
        plt.ylabel(r"$V(r)$ [MeV]")
        plt.tight_layout()
        plt.legend()
        plt.show()

    def plotDepths(self,E : np.array):
        plt.plot(E, self.real_vol_depth.V0(E),  label=r"$V_v$")
        plt.plot(E, self.real_so_depth.V0(E),   label=r"$V_{so}$")
        plt.plot(E, self.cmpl_vol_depth.V0(E),  label=r"$W_v$")
        plt.plot(E, self.cmpl_so_depth.V0(E),   label=r"$W_{so}$")
        plt.plot(E, self.cmpl_surf_depth.V0(E), label=r"$W_{D}$")
        plt.xlabel(r"$E_{CM}$ [MeV]")
        plt.ylabel(r"Potential Depth $V_0$ [MeV]")
        plt.tight_layout()
        plt.legend()
        plt.show()

if __name__ == "__main__":
    print("Plotting Fe-56 potentials:")
    Fe56   = Nuclide(56,26)
    n      = Neutron()
    params = OMPParams(Fe56,n)
    omp  = OMP(Fe56, n, params)
    omp.plotAll(10.0, np.linspace(0,15,100))
    omp.plotDepths(np.linspace(0.01,200,200))

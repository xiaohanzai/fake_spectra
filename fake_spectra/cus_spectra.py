"""A customized RandSpectra class including a range of features for my LAF project"""

from __future__ import print_function
import numpy as np
from scipy.interpolate import interp1d
from . import abstractsnapshot as absn
from . import spectra

class CusSpectra(spectra.Spectra):
    """Included features:
        TDR_file - if not empty, this file contains a temperature density relation to interpolate from.
        include_shockheated - whether to collapse shock heated gas (T > 1e5 K) onto the TDR as well.
        pecvel - if False, set peculiar velocity to 0.
    """
    def __init__(self, num, base, cofm, axis, TDR_file="", include_shockheated=False, pecvel=True, **kwargs):
        spectra.Spectra.__init__(self, num, base, cofm, axis, **kwargs)

        self.TDR_file = TDR_file
        self.include_shockheated = include_shockheated
        self.pecvel = pecvel

    def _read_particle_data(self, fn, elem, ion, get_tau):
        """Read the particle data for a single interpolation"""
        pos = self.snapshot_set.get_data(0,"Position",segment = fn).astype(np.float32)
        hh = self.snapshot_set.get_smooth_length(0,segment=fn).astype(np.float32)

        #Find particles we care about
        if self.cofm_final:
            try:
                ind = self.part_ind[fn]
            except KeyError:
                ind = self.particles_near_lines(pos, hh,self.axis,self.cofm)
                self.part_ind[fn] = ind
        else:
            ind = self.particles_near_lines(pos, hh,self.axis,self.cofm)
        #Do nothing if there aren't any, and return a suitably shaped zero array
        if np.size(ind) == 0:
            return (False, False, False, False,False,False)
        pos = pos[ind,:]
        hh = hh[ind]
        #Get the rest of the arrays: reducing them each time to have a smaller memory footprint
        vel = np.zeros(1,dtype=np.float32)
        temp = np.zeros(1,dtype=np.float32)
        if get_tau:
            vel = self.snapshot_set.get_data(0,"Velocity",segment = fn).astype(np.float32)
            vel = vel[ind,:]
        #gas density amu / cm^3
        den=self.gasprop.get_code_rhoH(0, segment=fn).astype(np.float32)
        # Get mass of atomic species
        if elem != "Z":
            amumass = self.lines.get_mass(elem)
        else:
            amumass = 1
        den = den[ind]
        #Only need temp for ionic density, and tau later
        if get_tau or (ion != -1 and elem != 'H'):
            temp = self.snapshot_set.get_temp(0,segment=fn, units=self.units).astype(np.float32)
            temp = temp[ind]
        #Find the mass fraction in this ion
        #Get the mass fraction in this species: elem_den is now density in ionic species in amu/cm^2 kpc/h
        #(these weird units are chosen to be correct when multiplied by the smoothing length)
        elem_den = (den*self.rscale)*self.get_mass_frac(elem,fn,ind)
        #Special case H1:
        if elem == 'H' and ion == 1:
            # Neutral hydrogen mass frac
            elem_den *= (self.gasprop.get_reproc_HI(0, segment=fn)[ind]).astype(np.float32)
        elif ion != -1:
            #Cloudy density in physical H atoms / cm^3
            ind2 = self._filter_particles(elem_den, pos, vel, den)
            if np.size(ind2) == 0:
                return (False, False, False, False,False,False)
            #Shrink arrays: we don't want to interpolate particles
            #with no mass in them
            temp = temp[ind2]
            pos = pos[ind2]
            hh = hh[ind2]
            if get_tau:
                vel = vel[ind2]
            elem_den = elem_den[ind2] * self._get_elem_den(elem, ion, den[ind2], temp, ind, ind2)
            del ind2
        
        # Now deal with the customized flags
        # TDR
        if self.TDR_file != "":
            TDR = np.loadtxt(self.TDR_file)
            logDelta_lo = TDR[0,0] # lower bound of log(Delta) for doing interpolation
            logDelta_hi = TDR[-1,0] # upper bound
            TDR = TDR[1:-1] # the first and last lines of this file are not used for interpolation

            rho_mean = self.units.rho_crit(self.hubble) * self.omegab * (1+self.red)**3 # in physical units
            logDelta = np.log10(den*self.units.protonmass/rho_mean)
            if not self.include_shockheated:
                ii = (logDelta >= logDelta_lo) & (logDelta <= logDelta_hi) & (temp < 1e5)
            else:
                ii = (logDelta >= logDelta_lo) & (logDelta <= logDelta_hi)

            # deal with temperature
            f = interp1d(TDR[:,0], TDR[:,1], kind="linear", fill_value="extrapolate")
            temp_new = 10**f(logDelta[ii])

            if len(temp) > 1:
                temp_old = temp[ii].copy()
                temp[ii] = temp_new
            else:
                temp_old = self.snapshot_set.get_temp(0,segment=fn, units=self.units).astype(np.float32)[ind][ii]

            # deal with HI fraction
            if elem == 'H' and ion == 1:
                alpha = lambda T: T**-0.7 / (1. + (T / 1e6)**0.7)
                elem_den[ii] *= alpha(temp_new)/alpha(temp_old)

            del TDR, f, logDelta, ii, temp_new, temp_old

        # peculiar velocity
        if not self.pecvel:
            vel *= 0.

        #Get rid of ind so we have some memory for the interpolator
        del den
        #Put density into number density of particles, from amu
        elem_den/=amumass
        #Do interpolation.

        return (pos, vel, elem_den, temp, hh, amumass)

class CusRandSpectra(CusSpectra):
    def __init__(self, num, base, ax=1, numlos=5000, savefile="rand_spectra.hdf5", elem="H", ion=1, **kwargs):
        #Load halos to push lines through them
        f = absn.AbstractSnapshotFactory(num, base)
        self.box = f.get_header_attr("BoxSize")
        del f
        self.NumLos = numlos
        #All through x axis
        axis = np.ones(self.NumLos)*ax
        #Sightlines at random positions
        #Re-seed for repeatability
        np.random.seed(23)
        cofm = self.get_cofm()
        CusSpectra.__init__(self, num, base, cofm, axis, savefile=savefile, reload_file=True, load_halo=False, **kwargs)

    def get_cofm(self, num=None):
        """Find a bunch more sightlines: should be overriden by child classes"""
        if num is None:
            num = self.NumLos
        cofm = self.box*np.random.random_sample((num,3))
        return cofm


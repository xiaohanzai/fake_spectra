# -*- coding: utf-8 -*-
"""Contains the power spectrum - specific functions for the spectrum analysis code."""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

from . import spectra
from . import fluxstatistics as fstat

def flux_power(tau, vmax, spec_res = 8, mean_flux_=None, mean_flux_desired=None, scale=1., window=True):
    """ I made changes to the original flux_power function in fluxstatistics.
        Get the power spectrum of (variations in) the flux along the line of sight.
        This is: P_F(k_F) = <d_F d_F>
                 d_F = e^-tau / mean(e^-tau) - 1
        If mean_flux_desired is set, the spectral optical depths will be rescaled
        to match the desired mean flux.
        We compute the power spectrum along each sightline and then average the result.
        Arguments:
            tau - optical depths. Shape is (NumLos, npix)
            mean_flux_desired - Mean flux to rescale to.
	    vmax - velocity scale corresponding to maximal length of the sightline.
            scale - scaling factor of tau (intensity of the UVB)
        Returns:
            flux_power - flux power spectrum in km/s. Shape is (npix)
            bins - the frequency space bins of the power spectrum, in s/km.
    """
    if mean_flux_desired is not None:
        scale = fstat.mean_flux(tau, mean_flux_desired)
        #print("rescaled: ",scale,"frac: ",np.sum(tau>1)/np.sum(tau>0))
    else:
        mean_flux_desired = np.mean(np.exp(-scale*tau))
    (nspec, npix) = np.shape(tau)
    mean_flux_power = np.zeros(npix//2+1, dtype=tau.dtype)
    for i in range(10):
        end = min((i+1)*nspec//10, nspec)
        dflux=np.exp(-scale*tau[i*nspec//10:end])/mean_flux_desired - 1.
        if mean_flux_ is not None:
            dflux=np.exp(-scale*tau[i*nspec//10:end])/mean_flux_ - 1.
        # Calculate flux power for each spectrum in turn
        flux_power_perspectra = fstat._powerspectrum(dflux, axis=1)
        #Take the mean and convert units.
        mean_flux_power += vmax*np.sum(flux_power_perspectra, axis=0)
    mean_flux_power/= nspec
    assert np.shape(mean_flux_power) == (npix//2+1,)
    kf = fstat._flux_power_bins(vmax, npix)
    #Divide out the window function
    if window:
        mean_flux_power /= fstat._window_function(kf, R=spec_res, dv=vmax/npix)**2
    return kf,mean_flux_power

try:
    xrange(1)
except NameError:
    xrange = range


class CalcPowerspectrum(spectra.Spectra):
    """Class to calculate power spectrum and associated things.
    I assume we already have a spectra file written."""
    def __init__(self, num, base, savefile, **kwargs):
        spectra.Spectra.__init__(
            self, num, base, cofm=None, axis=None, savefile=savefile, reload_file=False, **kwargs)

    def calc_scaling(self, elem="H", ion=1, line=1215, mean_flux_desired=None):
        """Calculate the scaling factor of UVB in order to obtain mean_flux_desired"""
        if mean_flux_desired is None:
            return 1.

        tau = self.get_tau(elem, ion, line)
        return fstat.mean_flux(tau, mean_flux_desired)

    def get_flux_power_1D(self, elem="H",ion=1, line=1215, mean_flux_=None, mean_flux_desired = None, scale=1., window=True):
        """This is a rewrite of the function in spectra.Spectra"""
        tau = self.get_tau(elem, ion, line)
        #Mean flux rescaling does not commute with the spectrum resolution correction!
        if mean_flux_desired is not None and self.spec_res > 0:
            raise ValueError("Cannot sensibly rescale mean flux with gaussian smoothing")
        (kf, avg_flux_power) = flux_power(tau, self.vmax, spec_res=self.spec_res, mean_flux_=mean_flux_, mean_flux_desired=mean_flux_desired, scale=scale, window=window)
        return kf[1:],avg_flux_power[1:]

    def calc_powerspectrum(self, elem="H", ion=1, line=1215, mean_flux_=None, mean_flux_desired=None, scale=1., window=True, kmin=0.001, kmax=0.3, N=1000):
        """Calculate power spectrum and do an interpolation in the range [kmin, kmax], using N points. Return the interpolated values.
        Can choose to input mean_flux_desired or scaling of the UVB."""
        xinterp = np.linspace(np.log10(kmin), np.log10(kmax), N)
        rst = self.get_flux_power_1D(
            elem, ion, line, mean_flux_, mean_flux_desired, scale, window)
        yinterp = np.interp(xinterp, np.log10(rst[0]), np.log10(
            rst[1]), left=np.nan, right=np.nan)
        return 10**xinterp, 10**yinterp

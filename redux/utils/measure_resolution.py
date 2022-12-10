# Test code to measure resolution of spectra

import numpy as np
from astropy.io import fits
import glob
from astropy.modeling import models, fitting

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

wvlregion = [3700.,5100.]

def xshooter():
    # Measure std dev of absorption line in XSL spectrum

    xsl_path = '/Users/miadelosreyes/anaconda3/envs/ifu/lib/python3.7/site-packages/ppxf/xshooter_models'
    templates = glob.glob(xsl_path + '/XSL_SSP_logT7.8_MH-0.2_Kroupa_PC.fits')

    # Open template spectrum in order to get the size of the template array
    hdu = fits.open(templates[0])
    ssp = hdu[0].data
    h2 = hdu[0].header
    logLam = np.arange(len(ssp)) * h2['CDELT1'] + h2['CRVAL1']
    wvl = np.power(10.,logLam) * 10.  # Convert to Angstroms

    # Mask out extraneous regions (anything outside range of 3700-5100A)
    mask = (wvl > wvlregion[0]) & (wvl < wvlregion[1])

    # Mask out everything except Mg line
    mask_mgb = (wvl > 4770) & (wvl < 4970)

    # Fit line with Gaussian + linear background
    flux = ssp[mask_mgb]/np.median(ssp[mask_mgb])
    gaussian_model = models.Gaussian1D(amplitude=-0.1, mean=4862., stddev=5) + models.Linear1D(0,1)
    fitter = fitting.LevMarLSQFitter()
    gaussian_fit = fitter(gaussian_model, wvl[mask_mgb], flux)

    # Get best-fit parameters
    params = gaussian_fit.parameters
    amp, mean, stddev = params[0:3]
    print(params)
    print('Xshooter sigma:',stddev)

    wvltest = np.linspace(4770, 4970, 1000)

    plt.plot(wvl[mask_mgb], flux, 'k-')
    plt.plot(wvltest, gaussian_fit(wvltest), 'r-')
    plt.show()

    return

def arclamp():
    # Measure std dev of arc lamp line

    f = fits.open('/Users/miadelosreyes/Documents/Research/VoidDwarfs/data/arctest.fits')
    flux = f[0].data
    wvl = np.arange(len(flux))

    mask = (wvl > 670.) & (wvl < 690.)

    # Fit line with Gaussian + linear background
    flux = flux/np.median(flux)
    gaussian_model = models.Gaussian1D(amplitude=1.5, mean=680, stddev=0.5) + models.Linear1D(0,1)
    fitter = fitting.LevMarLSQFitter()
    gaussian_fit = fitter(gaussian_model, wvl[mask], flux[mask])

    # Get best-fit parameters
    params = gaussian_fit.parameters
    amp, mean, stddev = params[0:3]
    print(params)
    print('arc lamp sigma:',stddev)

    wvltest = np.linspace(670, 690, 50)

    plt.plot(wvl[mask], flux[mask], 'k-')
    plt.plot(wvltest, gaussian_fit(wvltest), 'r-')
    plt.show()

    return

def integratedspec():
    # Measure std dev of absorption line in integrated galaxy spec

    from kcwiintegrated import Cube
    c = Cube('reines65', folder='/Users/miadelosreyes/Documents/Research/VoidDwarfs/data/', verbose=False, wcscorr=[174.17801 - 174.1787083, 26.727126 - 26.7263583], z=0.0331, EBV=0.0217)
    c.integrate(plot=False, covparams=[0.108, 1.65, 80])

    wvl = c.wvl_zcorr[c.goodwvl]
    spec = c.totalspec[c.goodwvl]
    mask_mgb = (wvl > 4565) & (wvl < 4576)

    print(np.diff(wvl))

    plt.plot(wvl, spec, 'k-')
    plt.show()

    # Fit line with Gaussian + linear background
    flux = spec[mask_mgb]/np.median(spec[mask_mgb])
    gaussian_model = models.Gaussian1D(amplitude=-0.1, mean=4571., stddev=1.) + models.Linear1D(0,1)
    fitter = fitting.LevMarLSQFitter()
    gaussian_fit = fitter(gaussian_model, wvl[mask_mgb], flux)

    # Get best-fit parameters
    params = gaussian_fit.parameters
    amp, mean, stddev = params[0:3]
    print(params)
    print('Galaxy sigma:',stddev)

    wvltest = np.linspace(4565, 4576, 100)

    plt.plot(wvl[mask_mgb], flux, 'k-')
    plt.plot(wvltest, gaussian_fit(wvltest), 'r-')
    plt.show()

    return

if __name__ == "__main__":
    xshooter()
    #arclamp()
    #integratedspec()
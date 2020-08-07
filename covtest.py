# kcwiredux.py
# Script to compute spatial correlation effects
#
# Created: 11 July 2020
######################################

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from cwitools import reduction, modeling
import pandas as pd
from scipy.optimize import curve_fit

def makeblankfits(filename, folder='/raid/keck/kcwi/2020jan22/'):
	"""Copy fits file and replace image data with random data

		Inputs:
		- filename 	-- name of fits file to copy
	"""

	# Open intk file
	with fits.open(filename+'_intk.fits', mode='update') as f:

		# Replace data with ones
		f[0].data = np.ones(size=np.shape(f[0].data))
		f.flush()

	# Open vark file
	with fits.open(filename+'_vark.fits', mode='update') as f:

		# Replace variance with random values (normally distributed with mean=1 and stddev=1)
		f[0].data = np.power(np.random.normal(loc=1.0, scale=1.0, size=np.shape(f[0].data)),2.)
		f.flush()

	return

def testcovar(filename, folder='/Users/miadelosreyes/Documents/Research/VoidDwarfs/data/'):
	"""Code to run covariance curve code

		Inputs:
		- fits (astropy HDU): input HDU with 3d data
		- var (np.array): variance cube
		- mask (np.array): mask cube

		Keywords:

		Outputs:
		- bin_sizes (np.array): Bin sizes for the independently measured data points
		- noise_ratios (np.array): Rescaling factor (ratio of actual noise to
            naively propagated noise) for the independely measured data points
	"""

	# Open file
	hdu = fits.open(folder+filename+'_icubes.fits')
	vhdu = fits.open(folder+filename+'_vcubes.fits')
	mhdu = fits.open(folder+filename+'_mcubes.fits')

	# Get arrays
	var = vhdu[0].data
	mask = mhdu[0].data

	# Run covariance fitting algorithm by Donal/Yuguang
	hdu, param, bin_sizes, noise_ratios = reduction.fit_covar_xy(hdu, var, mask, return_all=True, plot=True)
	plt.tight_layout()
	plt.show()
	print(param)

	# Save the data from this
	np.savetxt('binsizes.out', bin_sizes)
	np.savetxt('noiseratios.out', noise_ratios)

	return bin_sizes, noise_ratios

def plotcovtest(bin_sizes=None, noise_ratios=None):
	"""Code to plot covariance curve data and fit

	Inputs:
	- bin_sizes (np.array): Bin sizes for the independently measured data points
	- noise_ratios (np.array): Rescaling factor (ratio of actual noise to
        naively propagated noise) for the independely measured data points

	Keywords:

	Outputs:

	"""

	# Get data
	if bin_sizes is None:
		bin_sizes = np.loadtxt('binsizes.out')
	if noise_ratios is None:
		noise_ratios = np.loadtxt('noiseratios.out')

	# Define functional form from Husemann et al. (2013)
	def beta(N, alpha):
		res = 1. + alpha*np.log(N)
		threshold = 100
		res[N > threshold] = 1. + alpha*np.log(threshold)
		return res

	# Do fitting
	idx = np.where((bin_sizes > 1) & (bin_sizes < 101))[0]
	popt, pcov = curve_fit(beta, bin_sizes, noise_ratios)
	print('alpha = ', popt[0])

	# Make plot
	plt.plot(bin_sizes, noise_ratios, 'ko', alpha=0.2)
	xplot = np.linspace(0,360,100)
	plt.plot(xplot, beta(xplot,popt[0]), 'r-')
	plt.axvline(100, color='b', linestyle='--')
	plt.xlabel('Bin size', fontsize=16)
	plt.ylabel(r'$n_{\mathrm{measured}}/n_{\mathrm{no covar}}$', fontsize=16)
	plt.savefig('figures/covtest2_fluxed.png', bbox_inches='tight')
	plt.show()

	return

def main():

	testcovar('kb200122_00079')
	plotcovtest()

	return

if __name__ == "__main__":
	main()
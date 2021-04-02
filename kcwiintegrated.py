# kcwiintegrated.py
# Script to compute integrated properties
# from KCWI data cubes
#
# Created: 30 March 2021
######################################

#Backend for python3 on mahler
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Change fonts
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import os
from params import params

# Packages for isophote fitting
from astropy.modeling import models, fitting
from photutils.isophote import EllipseGeometry, Ellipse
from photutils.aperture import EllipticalAperture

# Packages for stellar continuum fitting
import ppxf as ppxf_package
from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
from os import path
import glob
from scipy import ndimage

# Packages for emission line fitting
from astropy.modeling import models, fitting
from k_lambda import k_lambda
from tqdm import tqdm

# Wavelength dictionary for standard lines (from NIST when possible)
wvldict = {'Hbeta':4861.35, 'Hgamma':4340.472, 'Hdelta':4101.734, 'Hepsilon':3970.075,
		'OII3727':3727.320, 'OII3729':3729.225, 'OII3727_doublet':3728., 'OIII4363':4363.209, 'OIII4959':4959., 'OIII5007':5006.8}

class Cube:
	"""
	A class for each reduced IFU datacube.

	Attributes:
		hdu (astropy HDU): FITS HDU of intensity cube
		header (astropy HDU header): FITS header of data cube
		data (3D array): intensity cube
		var (3D array): variance cube
		mask (3D array): mask cube
		wvl_zcorr (1D array): redshift-corrected wavelength
		wcs (astropy WCS): coordinate system
		verbose (bool): if 'True', make test plots

	Methods:

	"""

	def __init__(self, filename, folder='/Users/miadelosreyes/Documents/Research/VoidDwarfs/data/', verbose=False, wcscorr=None, z=0., sn_wvl=[4250.,4300.], wvlrange=[3700., 5100.], EBV=0.):

		"""Opens datacube and sets base attributes.

			Args:
				filename (str): Name of cube to open.
				folder (str): Path in which data are stored.
				verbose (bool): If 'True', print the header and make test plots
				wcscorr (float list): [delta(RA), delta(Dec)] to correct for pointing errors
								where delta(RA) = (new RA) - (old RA)
				z (float): redshift
				sn_wvl (float list): lower and upper wavelength bounds across which to compute S/N
				wvlrange (float list): lower and upper wavelength bounds to keep
				EBV (float): E(B-V) value from Schlafly & Finkbeiner (2011), used to correct for Galactic reddening
		"""

		print('Initializing cube...')

		# Define galaxy name
		self.folder = folder
		self.galaxyname = filename

		# Make output folders
		if not os.path.exists('output/'+self.galaxyname):
			os.makedirs('output/'+self.galaxyname)
		if not os.path.exists('figures/'+self.galaxyname):
			os.makedirs('figures/'+self.galaxyname)

		# Open main intensity cube
		icube = fits.open(folder+filename+'_icubes.fits')
		data = icube[0].data
		self.hdu = icube
		self.header = icube[0].header

		# Open variance cube
		with fits.open(folder+filename+'_vcubes.fits') as ecube:
			var = ecube[0].data

		# Make mask
		self.mask = np.zeros_like(data)
		badidx = np.where((np.isclose(data,0)) & (np.isclose(var,0)))
		self.mask[badidx] = True
		badidx = np.where((data < 0.) | (~np.isfinite(var)))
		self.mask[badidx] = True

		# Mask the data and variance cubes
		self.data = np.ma.array(data, mask=self.mask)
		self.var = np.ma.array(var, mask=self.mask)

		# Apply WCS shift to correct for pointing errors
		if len(wcscorr)==2:
			self.header['CRVAL1'] += wcscorr[0]
			self.header['CRVAL2'] += wcscorr[1]
		self.wcs = WCS(icube[0].header)

		# Make wavelength array
		N = len(self.data[:,0,0])
		wvl0 = self.header['CRVAL3'] # wvl zeropoint
		wvld = self.header['CD3_3'] # wvl Angstroms per pixel
		wvl = np.arange(wvl0,wvl0+N*wvld,wvld)

		# Do redshift correction
		self.z = z
		self.wvl_zcorr = wvl / (1.+self.z)

		# Do correction for Galactic reddening
		if EBV > 0.:
			Alam = k_lambda(self.wvl_zcorr)*EBV
			Alam_array = np.tile(Alam[:, np.newaxis, np.newaxis], (1, self.data.shape[1], self.data.shape[2]))
			self.data = self.data/np.power(10.,(Alam_array/(-2.5)))

		# Define wavelength ranges
		bband_center = 4130.
		bbandwvl = np.where((self.wvl_zcorr > (bband_center - 50.)) & (self.wvl_zcorr < (bband_center + 50.)))[0]  # Wavelength range for B-band
		self.goodwvl = np.where((self.wvl_zcorr > wvlrange[0]) & (self.wvl_zcorr < wvlrange[1]))[0]  # Wavelength range for stellar template fitting
		self.snwvl = np.where((self.wvl_zcorr > sn_wvl[0]) & (self.wvl_zcorr < sn_wvl[1]))[0]  # Wavelength range for S/N fitting

		# Compute pixel area (in arsec^2)
		dx = np.sqrt(self.header['CD1_1']**2+self.header['CD2_1']**2)*3600.
		dy = np.sqrt(self.header['CD1_2']**2+self.header['CD2_2']**2)*3600.
		self.area = dx*dy

		# Get zeropoints and deltas for coordinates
		self.ra0 = self.header['CRVAL1']
		self.dec0 = self.header['CRVAL2']
		self.rad = self.header['CD1_1'] # RA degrees per col
		self.decd = self.header['CD2_2'] # Dec degrees per row

		self.xsize = self.data.shape[2]
		self.ysize = self.data.shape[1]

		# B-band image in surface brightness units
		self.bband = np.ma.sum(self.data[bbandwvl,:,:], axis=0)
		self.bband /= self.area 

		# Convert to mag/arcsec2
		# Original units: 10^(-16) erg/s/cm2/A/arcsec2
		# mag = mag_ref - 2.5*np.log10(flux/flux_ref)
		# Use reference constants for Johnson B band: https://irsa.ipac.caltech.edu/data/SPITZER/docs/dataanalysistools/tools/pet/magtojy/
		self.mag = -20.4 - 2.5*np.log10(self.bband*1e-16)

		# Plot B-band surface brightness in mag/arcsec2
		if verbose:
			fig = plt.figure(figsize=(5,5))
			ax = plt.subplot(projection=self.wcs,slices=('x', 'y', 50))
			plt.imshow(self.mag, cmap='viridis_r', vmax=22)
			plt.colorbar()
			plt.show()

	def ellipsefit(self, plot=True, mode='sersic'):
		""" Fit cube with elliptical isophote

			Args:
				plot (bool): if 'True', make plots
				mode (str): method to use to do fit 
					('sersic' for Sersic fitting, 'isophote' for elliptical isophotes)

			Outputs:
				Re (float): half-light radius
		"""

		print('Fitting ellipse to white-light image...')

		if mode=='sersic':
			# Fit with 2D Sersic profile
			y, x = np.mgrid[:self.ysize, :self.xsize]
			p_init = models.Sersic2D(amplitude=0.05, r_eff=25, n=4, x_0=25, y_0=43, ellip=0.5, theta=5)
			fit_p = fitting.LevMarLSQFitter()
			p = fit_p(p_init, x, y, self.bband)
			print(p)

			if plot:
				fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,5), sharex=True)
				axs[0].imshow(self.bband, origin='lower', cmap='viridis', vmin=0, vmax=13)
				axs[1].imshow(p(x, y), origin='lower', cmap='viridis', vmin=0, vmax=13)
				plt.show()

		# TODO: finish elliptical isophote fitting
		if mode=='isophote':
			pass	

		return
	
	def integrate(self, plot=False, mode='snr', covparams=None):
		""" Get integrated spectrum

			Args:
				plot (bool): if 'True', make plots
				mode (str): method to use to decide which pixels to stack
					'snr': add pixels with S/N > 1,
					'isophote': add pixels within elliptical isophote (from ellipsefit),
					'sersic': add pixels within best-fit Sersic profile (from ellipsefit)
				covparams (float tuple): (alpha, norm, threshold) parameters for cov correction
		"""

		if mode=='snr':

			# Compute signal
			signal = np.mean(self.data[self.snwvl,:,:], axis=0)
			
			# Compute noise as detrended standard deviation
			noise = np.zeros(np.shape(signal))
			for i in range(self.ysize):
				for j in range(self.xsize):
					linfit = np.polyfit(self.wvl_zcorr[self.snwvl],self.data[self.snwvl,i,j],deg=1)
					poly = np.poly1d(linfit)
					noise[i,j] = np.std(self.data[self.snwvl,i,j] - np.asarray(poly(self.wvl_zcorr[self.snwvl]))**2.)

			# Define S/N
			sntest = signal/noise
			if plot:
				plt.imshow(sntest, origin='lower', vmin=1)
				plt.colorbar()
				plt.show()

			# Create mask to determine where to add up data
			snmask = np.where(sntest > 1)
			n_goodpix = len(snmask[0])

		# Apply covariance correction
		alpha, norm, threshold = covparams
		self.covcorr = norm * (1 + alpha * np.log(threshold))
		self.var *= self.covcorr**2.

		# Coadd spectra
		self.totalspec = np.ma.average(self.data[:,snmask[0],snmask[1]], axis=1, weights=1./self.var[:,snmask[0],snmask[1]])
		self.totalvar = np.ma.average(self.var[:,snmask[0],snmask[1]], axis=1)/n_goodpix

		# Convert to non-masked arrays
		self.totalspec = self.totalspec.compressed()
		self.totalvar = self.totalvar.compressed()

		if plot:
			plt.plot(self.wvl_zcorr[self.goodwvl], self.totalspec[self.goodwvl])
			plt.fill_between(self.wvl_zcorr[self.goodwvl], (self.totalspec-np.sqrt(self.totalvar))[self.goodwvl], \
				(self.totalspec+np.sqrt(self.totalvar))[self.goodwvl], color='r', alpha=0.5)
			plt.show()

		# Compute final S/N
		signalfinal = np.mean(self.totalspec[self.snwvl])
		noisefinal = np.sqrt(np.mean(self.totalvar[self.snwvl]))
		print('Final S/N:', signalfinal/noisefinal)

		return

	def stellarfit(self, plot=True):
		""" Fit stellar continuum of integrated spectra

			Args:
				plot (bool): if 'True', make plots
		"""

		print('Preparing templates for stellar kinematics fit...')

		# Define path to pPXF directory
		ppxf_dir = path.dirname(path.realpath(ppxf_package.__file__))
		print(ppxf_dir)

		# Define spectrum
		spectrum = self.totalspec[self.goodwvl]
		noise = self.totalvar[self.goodwvl]
		wvl = self.wvl_zcorr[self.goodwvl]

		print(spectrum.shape, noise.shape, wvl.shape)

		# Define wavelength range
		lamRange1 = [wvl[0],wvl[-1]]
		fwhm_gal = 2.4/(1+self.z)  # KCWI instrumental FWHM of ~2.4A

		# Rebin spectrum into log scale to get initial velocity scale
		galaxy, logLam1, velscale = util.log_rebin(lamRange1, spectrum)

		# Read the list of filenames from the E-MILES SSP library
		vazdekis = glob.glob(ppxf_dir + '/miles_models/Mun1.30*.fits')
		fwhm_tem = 2.51  # Vazdekis+10 spectra have a constant resolution FWHM of 2.51A.

		# Open template spectrum in order to get the size of the template array
		hdu = fits.open(vazdekis[0])
		ssp = hdu[0].data
		h2 = hdu[0].header
		lamRange2 = h2['CRVAL1'] + np.array([0., h2['CDELT1']*(h2['NAXIS1'] - 1)])
		sspNew, logLam2, velscale_temp = util.log_rebin(lamRange2, ssp, velscale=velscale)
		templates = np.empty((sspNew.size, len(vazdekis)))

		# Convolve observed spectrum with quadratic difference between observed and template resolution.
		# (This is valid if shapes of instrumental spectral profiles are well approximated by Gaussians.)
		fwhm_dif = np.sqrt(np.abs(fwhm_gal**2 - fwhm_tem**2))
		sigma = fwhm_dif/2.355/h2['CDELT1']  # Sigma difference in pixels
		galspec = ndimage.gaussian_filter1d(spectrum, sigma)

		# Now logarithmically rebin this new observed spectrum
		galaxy, logLam1, velscale = util.log_rebin(lamRange1, galspec, velscale=velscale)

		# Open and normalize all the templates
		for j, file in enumerate(vazdekis):
			hdu = fits.open(file)
			ssp = hdu[0].data
			sspNew, logLam2, velscale_temp = util.log_rebin(lamRange2, ssp, velscale=velscale)
			templates[:, j] = sspNew/np.median(sspNew)  # Normalizes templates

		# Prep the observed spectrum
		galaxy = galaxy/np.median(galaxy)

		print('Doing stellar kinematics fit...')

		# Shift the template to fit the starting wavelength of the galaxy spectrum
		c = 299792.458
		dv = (logLam2[0] - logLam1[0])*c  # km/s

		goodPixels = util.determine_goodpixels(logLam1, lamRange2, 0)

		# Here the actual fit starts. The best fit is plotted on the screen
		start = [0., 200.]  # (km/s), starting guess for [V, sigma]

		pp = ppxf(templates, galaxy, np.sqrt(noise), velscale, start,
				  goodpixels=goodPixels, plot=plot, moments=2,
				  degree=6, vsyst=dv, clean=False, quiet=True)
		if plot:
			plt.show()

		print('Chi2:', pp.chi2)
		print('Best-fitting redshift z:', (self.z + 1)*(1 + pp.sol[0]/c) - 1)
		print('Final solution:', pp.sol)
		print("Errors:", pp.error*np.sqrt(pp.chi2))

		if plot:
			plt.figure(figsize=(9,3))
			lines = np.array([3726.03, 3728.82, 3970.08, 4101.76, 4340.47, 4363.21, 4861.33, 
					4958.92, 5006.84, 6300.30, 6548.03, 6583.41, 6562.80, 6716.47, 6730.85])
			for line in lines:
				if line < np.exp(logLam1)[-1]:
					plt.axvspan(line-10., line+10., color='gray', alpha=0.25)
			plt.fill_between(np.exp(logLam1), galaxy-np.sqrt(noise),galaxy+np.sqrt(noise), color='r', alpha=0.8)
			plt.plot(np.exp(logLam1), pp.bestfit, 'k-')
			plt.xlabel(r'$\lambda (\AA)$', fontsize=14)
			plt.ylabel(r'Normalized flux', fontsize=14)
			plt.ylim(-0.05,2.0)
			plt.savefig('figures/'+self.galaxyname+'/'+'intspec.png', bbox_inches='tight') 
			plt.show()

		return np.asarray([pp.sol[0], pp.sol[1], pp.error[0]*np.sqrt(pp.chi2), pp.error[1]*np.sqrt(pp.chi2)]), np.exp(logLam1), pp.bestfit, galaxy, noise

	# TODO: Compute integrated stellar abundances???

	# TODO: Compute integrated emission-line fluxes

	# TODO: Compute integrated gas-phase metallicity

def getsystvel(galaxyname, folder='/raid/madlr/voids/analysis/stackedcubes/'):
	""" Run full pipeline to get systemic stellar kinematics.

	Arguments:
		verbose (bool): if 'True', make diagnostic plots
		overwrite (bool): if 'True', overwrite existing data files
		makeplots (bool): if 'True', just run the steps required to make plots
			(note: only works if all steps have been run before!)
	"""

	# Open params
	param = params[galaxyname]

	# Open cube
	c = Cube(galaxyname, folder=folder, verbose=param['verbose'], wcscorr=param['wcscorr'], z=param['z'], EBV=param['EBV'])
	c.integrate(plot=False, covparams=param['covparams'])
	c.stellarfit(plot=True)

	return

def main():

	c = Cube('reines65', folder='/Users/miadelosreyes/Documents/Research/VoidDwarfs/analysis/stackedcubes/', verbose=False, wcscorr=[174.17801 - 174.1787083, 26.727126 - 26.7263583], z=0.0331, EBV=0.0217)
	#c.ellipsefit()
	c.integrate(plot=False, covparams=[0.108,1.65,80])
	c.stellarfit(plot=True)

	return

if __name__ == "__main__":
	main()
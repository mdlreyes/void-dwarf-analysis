# kcwiintegrated.py
# Script to compute integrated properties
# from KCWI data cubes
#
# Created: 30 March 2021
######################################

#Backend for python3 on mahler
import matplotlib
matplotlib.use('TkAgg')
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
#import pyneb as pn

# Wavelength dictionary for standard lines (from NIST when possible)
wvldict = {'Hbeta':4861.35, 'Hgamma':4340.472, 'Hdelta':4101.734, 'Hepsilon':3970.075,
		'OII3727':3727.320, 'OII3729':3729.225, 'OII3727doublet':3728., 'OIII4363':4363.209, 'OIII4959':4959., 'OIII5007':5006.8}

def fitline(data, wvlarray, err, line_name, xlim=10., plot=False):
	""" Fits gas emission lines and makes emission line maps.

		Arguments:
			datanorm, wvlnorm, errnorm (1D array): data, wavelength, and error arrays to be fit
			line_name (string): name of line to fit
			xlim (float): in Angstroms, half of wavelength range about line center to fit
			plot (bool): if 'True', make test plots

		Returns:
			integral, integral_err, stddev, stddev_err, snr (floats): line flux, width, signal/noise
	"""

	# Get central line wavelength
	line = wvldict[line_name]

	# Crop spectrum arrays to only contain area around the line
	idx = np.where((wvlarray > (line-xlim)) & (wvlarray < (line+xlim)))[0]
	wvl = wvlarray[idx]
	flux = data[idx]
	err = err[idx]

	# Fit line with Gaussian + linear background
	gaussian_model = models.Gaussian1D(np.max(flux), line, 2) + models.Linear1D(0,0)
	fitter = fitting.LevMarLSQFitter()
	gaussian_fit = fitter(gaussian_model, wvl, flux, weights=1./err)

	# Get best-fit parameters
	params = gaussian_fit.parameters
	amp, mean, stddev = params[0:3]

	try:
		paramerrs = np.sqrt(np.diag(fitter.fit_info['param_cov']))
		amp_err, mean_err, stddev_err = paramerrs[0:3]
	except:
		amp_err, mean_err, stddev_err = [np.nan, np.nan, np.nan]

	integral = np.sqrt(2.*np.pi)*amp*stddev
	integral_err = integral * np.sqrt(2.*np.pi)*np.sqrt((amp_err/amp)**2. + (stddev_err/stddev)**2.)

	if plot:
		fig, ax = plt.subplots()
		plt.plot(wvlarray, data, 'k-')
		plt.plot(wvl, gaussian_fit(wvl), 'r-')
		plt.axvspan(gaussian_fit[0].mean-2.5*gaussian_fit[0].stddev, gaussian_fit[0].mean+2.5*gaussian_fit[0].stddev, alpha=0.5)
		plt.axvspan(gaussian_fit[0].mean+5*gaussian_fit[0].stddev,gaussian_fit[0].mean+10*gaussian_fit[0].stddev, color='C1', alpha=0.5)
		plt.axvspan(gaussian_fit[0].mean-10*gaussian_fit[0].stddev,gaussian_fit[0].mean-5*gaussian_fit[0].stddev, color='C1', alpha=0.5)
		plt.xlim(gaussian_fit[0].mean-50,gaussian_fit[0].mean+50)
		plt.xlabel(r'$\lambda (\AA)$', fontsize=14)
		plt.ylabel(r'Normalized flux', fontsize=14)
		plt.text(0.05, 0.9, line_name, transform=ax.transAxes)
		plt.show()

	# Compute SNR
	if integral > 1e-3 and stddev < xlim/2. and np.abs(mean - line) < xlim/2.:

		emidx = np.where((wvl > (mean-2.5*stddev)) & (wvl < (mean+2.5*stddev)))[0]
		emflux = flux[emidx]

		cont1idx = np.where((wvl < (mean-5*stddev)))[0]
		cont2idx = np.where((wvl > (mean+5*stddev)))[0]
		contflux1 = flux[cont1idx]
		contflux2 = flux[cont2idx]

		signal = np.sum(emflux - np.mean(np.hstack((contflux1,contflux2)))) / np.sqrt(len(emflux))
		noisecont = (np.std(contflux1) + np.std(contflux2)) / 2. # Continuum noise
		pois = np.random.poisson(size=len(emflux))
		noisepois = np.std(pois/np.sum(pois)*np.sqrt(emflux)) # Poisson noise
		noise = np.sqrt(noisecont**2. + noisepois**2.)

	else:
		signal, noise = [np.nan, np.nan]

	if signal/noise > 0.:
		snr = signal/noise
	else:
		snr = np.nan

	return integral, integral_err, stddev, stddev_err, snr

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

	def __init__(self, filename, folder='/Users/miadelosreyes/Documents/Research/VoidDwarfs/data/', verbose=False, wcscorr=None, z=0., sn_wvl=[4750.,4800.], wvlrange=[3700., 5100.], EBV=0.):

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
		self.z = 0.033044123962904015 #z
		self.wvl_zcorr = wvl / (1.+self.z)

		# Do correction for Galactic reddening
		if EBV > 0.:
			Alam = k_lambda(self.wvl_zcorr)*EBV
			Alam_array = np.tile(Alam[:, np.newaxis, np.newaxis], (1, self.data.shape[1], self.data.shape[2]))
			self.data /= np.power(10.,(Alam_array/(-2.5)))
			self.var /= (np.power(10.,(Alam_array/(-2.5))))**2.

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
				plt.imshow(sntest, origin='lower')
				plt.colorbar()
				plt.show()

			# Create mask to determine where to add up data
			newdata = np.zeros_like(self.data)
			newvar =  np.zeros_like(self.data)
			n_goodpix = 0
			for i in range(self.ysize):
				for j in range(self.xsize):
					if sntest[i,j] > 1:
						newdata[:,i,j] = self.data[:,i,j]
						newvar[:,i,j] = self.var[:,i,j]
						n_goodpix += 1

		# TODO: add other modes to integrate spectra?
		if mode=='isophote':
			pass
		if mode=='sersic':
			pass

		# Apply covariance correction
		alpha, norm, threshold = covparams
		self.covcorr = norm * (1 + alpha * np.log(threshold))
		self.var *= self.covcorr**2.

		# Coadd spectra
		#self.totalspec = np.ma.average(self.data[:,snmask[0],snmask[1]], axis=1, weights=1./self.var[:,snmask[0],snmask[1]])
		#self.totalvar = np.ma.average(self.var[:,snmask[0],snmask[1]], axis=1)/n_goodpix
		weights = np.sum(newdata,axis=0)
		test = np.sum(newdata*weights,axis=(1,2))/np.sum(weights)
		testvar = np.sum(newvar*weights,axis=(1,2))/np.sum(weights)
		self.totalspec = test #np.ma.average(newdata, axis=(1,2)) #, weights=np.sum(newdata,axis=0))
		self.totalvar = testvar/n_goodpix #np.ma.average(newvar, axis=(1,2), weights=np.sum(newdata,axis=0))/n_goodpix

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
		# Compute noise as detrended standard deviation
		linfit = np.polyfit(self.wvl_zcorr[self.snwvl],self.data[self.snwvl,i,j],deg=1)
		poly = np.poly1d(linfit)
		noisefinal = np.std(self.totalspec[self.snwvl] - np.asarray(poly(self.wvl_zcorr[self.snwvl])))
		print(noisefinal)
		#noisefinal = np.sqrt(np.mean(self.totalvar[self.snwvl]))

		print('Final S/N:', signalfinal/noisefinal)
		self.sn = signalfinal/noisefinal

		return

	def stellarfit(self, plot=True):
		""" Fit stellar continuum of integrated spectra, return stellar kinematics,
			and subtract continuum/absorption features.

			Args:
				plot (bool): if 'True', make plots

			Returns:
				kinematics (array): [vel, veldisp, vel_err, veldisp_err, wvlarray, fitspectrum, obsspectrum, obsspectrum_err]
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

		# Read the list of filenames from template library 
		vazdekis = glob.glob(ppxf_dir + '/miles_models/Mku*.fits') # From E-MILES SSP library
		#vazdekis = glob.glob(ppxf_dir + '/miles_stellar/s*.fits') # From MILES stellar library
		fwhm_tem = 2.51  # Vazdekis+10 spectra have a constant resolution FWHM of 2.51A.

		# Open template spectrum in order to get the size of the template array
		hdu = fits.open(vazdekis[0])
		ssp = np.squeeze(hdu[0].data)
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

		# TEST: log-rebin the error spectrum too?
		noise, _, _ = util.log_rebin(lamRange1, noise, velscale=velscale)

		# Open and normalize all the templates
		for j, file in enumerate(vazdekis):
			hdu = fits.open(file)
			ssp = np.squeeze(hdu[0].data)
			sspNew, logLam2, velscale_temp = util.log_rebin(lamRange2, ssp, velscale=velscale)
			templates[:, j] = sspNew/np.median(sspNew)  # Normalizes templates

		# Prep the observed spectrum
		galaxy = galaxy/np.median(galaxy)

		print('Doing stellar kinematics fit...')
		print(galaxy.shape, noise.shape)

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
		print("v: {:.2f}\t{:.2f}".format(pp.sol[0], pp.error[0]*np.sqrt(pp.chi2)))
		print("sigma: {:.2f}\t{:.2f}".format(pp.sol[1], pp.error[1]*np.sqrt(pp.chi2)))

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
			plt.text(3750, 1.35, 'Integrated spectrum: S/N={:.1f}'.format(self.sn), fontsize=15)
			plt.ylim(0.5,1.5)
			plt.xlim(3700,5100)
			plt.savefig('figures/'+self.galaxyname+'/'+'intspec.pdf', bbox_inches='tight') 
			plt.show()

		# Subtract stellar contribution from spectrum
		print('Normalizing data by best-fit stellar template...')
		self.spectrum_norm = galaxy - pp.bestfit*np.median(galaxy)
		self.kinematics_wvl = np.exp(logLam1)
		np.save('output/'+self.galaxyname+'/intspec_norm', self.spectrum_norm)
		np.save('output/'+self.galaxyname+'/intspec_wvl', self.kinematics_wvl)

		# Plot image for testing
		if plot:

			# Plot example spectrum
			plt.figure(figsize=(9,3))
			lines = np.array([3726.03, 3728.82, 3970.08, 4101.76, 4340.47, 4363.21, 4861.33, 
					4958.92, 5006.84, 6300.30, 6548.03, 6583.41, 6562.80, 6716.47, 6730.85])
			for line in lines:
				if line < np.exp(logLam1)[-1]:
					plt.axvspan(line-10., line+10., color='gray', alpha=0.25)
			plt.fill_between(np.exp(logLam1), self.spectrum_norm-np.sqrt(noise),self.spectrum_norm+np.sqrt(noise), color='r', alpha=0.8)
			plt.plot(np.exp(logLam1), self.spectrum_norm, 'k-')
			plt.xlabel(r'$\lambda (\AA)$', fontsize=14)
			plt.ylabel(r'Normalized flux', fontsize=14)
			plt.xlim(3500,5100)

			plt.savefig('figures/'+self.galaxyname+'/'+'intspec_norm.pdf', bbox_inches='tight') 
			plt.show()

		return np.asarray([pp.sol[0], pp.sol[1], pp.error[0]*np.sqrt(pp.chi2), pp.error[1]*np.sqrt(pp.chi2)]), np.exp(logLam1), pp.bestfit, galaxy, noise

	# TODO: Compute integrated stellar abundances???

	def reddening(self, verbose=False):
		""" Compute and apply reddening correction to spectrum.

			Arguments:
				plot (bool): if 'True', make diagnostic plots
		"""

		print('Computing Balmer decrement...')

		try:
			data_norm = self.spectrum_norm
			kinematics_wvl = self.kinematics_wvl
		except AttributeError:
			data_norm = np.load('output/'+self.galaxyname+'/'+'intspec_norm.npy')
			kinematics_wvl = np.load('output/'+self.galaxyname+'/'+'intspec_wvl.npy')

		errors = np.sqrt(self.totalvar[self.goodwvl])

		# Get Balmer line maps
		resultHgamma = fitline(data_norm, kinematics_wvl, errors, 'Hgamma', plot=verbose)
		resultHbeta = fitline(data_norm, kinematics_wvl, errors, 'Hbeta', plot=verbose)
		#print(resultHgamma, resultHbeta)

		# Compute E(B-V) using MC method to get errors
		print('Computing E(B-V)...')
		Niter = int(1e3)
		ebv = np.zeros(Niter)
		balmer0 = 0.468 # Intrinsic Hgamma/Hbeta ratio (assuming Case B recombination, T=10^4K, electron density 100/cm^3)
		for i in tqdm(range(Niter)):

			# Compute Balmer decrement
			Hbeta = np.random.default_rng().normal(loc=resultHbeta[0], scale=resultHbeta[1])
			Hgamma = np.random.default_rng().normal(loc=resultHgamma[0], scale=resultHgamma[1])
			balmer = Hgamma/Hbeta

			# Compute E(B-V)
			ebv[i] = np.log10(balmer/balmer0)/(-0.4*(k_lambda(wvldict['Hgamma'])-k_lambda(wvldict['Hbeta'])))

		# Compute E(B-V) mean and errors
		ebv_mean = np.nanmean(ebv, axis=0)
		ebv_err = np.nanstd(ebv, axis=0)

		# Make test plots
		if verbose:
			# Compute E(B-V) from measured fluxes
			testbalmer = resultHgamma[0]/resultHbeta[0]
			testebv = np.log10(testbalmer/balmer0)/(-0.4*(k_lambda(wvldict['Hgamma'])-k_lambda(wvldict['Hbeta'])))[0]

			# Plot histogram of MC results
			plt.hist(ebv)
			plt.axvline(testebv, color='r', label="Measured from original fluxes: {:.2f}".format(testebv))
			plt.axvline(ebv_mean, color='k', linestyle='--', label="Mean: {:.2f}".format(ebv_mean))
			plt.axvspan(ebv_mean-ebv_err,ebv_mean+ebv_err, color='gray', alpha=0.25, label=r"$\sigma$: {:.2f}".format(ebv_err))
			plt.xlabel('E(B-V)')
			#plt.savefig('figures/'+self.galaxyname+'/intspec_EBVtest.png', bbox_inches='tight')
			plt.legend(loc='best')
			plt.show()

		# Apply reddening correction if E(B-V) isn't unreasonable
		print('Applying reddening correction...')
		if np.isfinite(ebv_mean) and ebv_mean > 0. and ebv_mean < 1.:

			# Apply reddening correction
			Alam = k_lambda(kinematics_wvl)*ebv_mean
			self.data_dered = data_norm/np.power(10.,(Alam/(-2.5)))
			self.errs_dered = errors/np.power(10.,(Alam/(-2.5)))

		else:
			self.data_dered = np.copy(data_norm)
			self.errs_dered = np.copy(errors)
			print("E(B-V) value ({:.2f}) is sus! No reddening correction applied.".format(ebv_mean))

		# Save de-reddened spectrum
		np.save('output/'+self.galaxyname+'/intspec_dered', self.data_dered)
		np.save('output/'+self.galaxyname+'/intspec_errs', self.errs_dered)

		if verbose:
			plt.figure(figsize=(9,3))
			plt.plot(kinematics_wvl,self.data_dered, label='De-reddened')
			plt.plot(kinematics_wvl,data_norm, label='Original')
			plt.xlabel(r'$\lambda (\AA)$', fontsize=16)
			plt.ylabel('Normalized flux', fontsize=16)
			plt.legend()
			plt.xlim(3500,5100)
			plt.show()

		return

	def metallicity_Te(self, verbose=False):
		""" Compute electron temperature metallicities

			Arguments:
				verbose (bool): if 'True', make diagnostic plots
		"""

		print('Measuring gas-phase abundance...')

		# Get de-reddened data
		try:
			data_norm = self.data_dered
			errs_norm = self.errs_dered
			kinematics_wvl = self.kinematics_wvl
		except AttributeError:
			data_norm = np.load('output/'+self.galaxyname+'/'+'intspec_dered.npy')
			errs_norm = np.load('output/'+self.galaxyname+'/'+'intspec_errs.npy')
			kinematics_wvl = np.load('output/'+self.galaxyname+'/'+'intspec_wvl.npy')

		# Get line fluxes
		resultOIII4363 = fitline(data_norm, kinematics_wvl, errs_norm, 'OIII4363', plot=verbose)
		resultOIII4959 = fitline(data_norm, kinematics_wvl, errs_norm, 'OIII4959', plot=verbose)
		resultOIII5007 = fitline(data_norm, kinematics_wvl, errs_norm, 'OIII5007', plot=verbose)
		resultOII3727 = fitline(data_norm, kinematics_wvl, errs_norm, 'OII3727doublet', plot=verbose)
		resultHbeta = fitline(data_norm, kinematics_wvl, errs_norm, 'Hbeta', plot=verbose)

		if np.any(~np.isfinite([i[-1] for i in [resultOIII4363,resultOIII4959,resultOIII5007]])):
			print('One or more of the OIII lines is not well measured. Check fluxes: ', [i[0] for i in [resultOIII4363,resultOIII4959,resultOIII5007]])

		# Compute metallicity using MC method to get errors
		Niter = int(1e4)
		gasZ_Te = np.zeros(Niter)
		for i in tqdm(range(Niter)):

			# Pull from distribution of line fluxes
			OIII4363 = np.random.default_rng().normal(loc=resultOIII4363[0], scale=resultOIII4363[1])
			OIII4959 = np.random.default_rng().normal(loc=resultOIII4959[0], scale=resultOIII4959[1])
			OIII5007 = np.random.default_rng().normal(loc=resultOIII5007[0], scale=resultOIII5007[1])
			OII3727 = np.random.default_rng().normal(loc=resultOII3727[0], scale=resultOII3727[1])
			Hbeta = np.random.default_rng().normal(loc=resultHbeta[0], scale=resultHbeta[1])

			# Measure OIII electron temp using Nicholls et al. (2020) calibration
			x = np.log10(OIII4363/(OIII4959 + OIII5007))  # log10(f4363/f5007)
			Te_OIII = np.power(10., (3.3027 + 9.1917*x)/(1. + 2.092*x - 0.1503*x**2 - 0.0093*x**3))  # K

			# Measure OII electron temp using Lopez-Sanchez et al. (2012)
			Te_OII = Te_OIII + 450 - 70*np.exp((Te_OIII/5000.)**1.22)

			# Compute O/H ionic ratios using Pérez-Montero (2017) analytical functions
			# Note that these units are 12 + log(O/H)
			tO2 = Te_OIII/1e4
			tO = Te_OII/1e4
			ne = 100.  # assume fixed electron density (cm^-3)
			O2H2 = np.log10((OIII4959 + OIII5007)/Hbeta) + 6.1868 + 1.2491/tO2 - 0.5816 * np.log10(tO2)
			OH = np.log10(OII3727/Hbeta) + 5.887 + 1.641/tO - 0.543*np.log10(tO) + 0.000114 * ne

			gasZ = 12. + np.log10(10.**(O2H2 - 12.) + 10.**(OH - 12.))
			if gasZ < 12.:
				gasZ_Te[i] = gasZ
			else:
				gasZ_Te[i] = np.nan

		# Compute metallicity from measured values
		x = np.log10(resultOIII4363[0]/(resultOIII4959[0] + resultOIII5007[0]))  # log10(f4363/f5007)
		Te_OIII = np.power(10., (3.3027 + 9.1917*x)/(1. + 2.092*x - 0.1503*x**2 - 0.0093*x**3))  # K
		Te_OII = Te_OIII + 450 - 70*np.exp((Te_OIII/5000.)**1.22)
		ne = 100.  # assume fixed electron density (cm^-3)
		O2H2 = np.log10((resultOIII4959[0] + resultOIII5007[0])/resultHbeta[0]) + 6.1868 + 1.2491/(Te_OIII/1e4) - 0.5816 * np.log10(Te_OIII/1e4)
		OH = np.log10(resultOII3727[0]/resultHbeta[0]) + 5.887 + 1.641/(Te_OII/1e4) - 0.543*np.log10(Te_OII/1e4) + 0.000114 * ne
		gasZ = 12. + np.log10(10.**(O2H2 - 12.) + 10.**(OH - 12.))
		print('Gas-phase metallicity (from measurements): ', gasZ)

		# Compute metallicity mean and errors
		self.gasZ = np.nanmean(gasZ_Te, axis=0)
		self.gasZ_err = np.nanstd(gasZ_Te, axis=0)
		print('Gas-phase metallicity (from distribution): ', self.gasZ, self.gasZ_err)

		# Make test plots
		if verbose:

			# Plot histogram of MC results
			plt.hist(gasZ_Te)
			plt.axvline(gasZ, color='r', label="Measured from original fluxes: {:.2f}".format(gasZ))
			plt.axvline(self.gasZ, color='k', linestyle='--', label="Mean: {:.2f}".format(self.gasZ))
			plt.axvspan(self.gasZ-self.gasZ_err,self.gasZ+self.gasZ_err, color='gray', alpha=0.25, label=r"$\sigma$: {:.2f}".format(self.gasZ_err))
			plt.axvline(np.nanpercentile(gasZ_Te, 16.), color='b', linestyle='--')
			plt.axvline(np.nanpercentile(gasZ_Te, 84.), color='b', linestyle='--')
			plt.xlabel('12+log(O/H)')
			plt.ylabel('N')
			plt.savefig('figures/'+self.galaxyname+'/intspec_gasZtest.png', bbox_inches='tight')
			plt.legend(loc='best')
			plt.show()

		return self.gasZ, self.gasZ_err

def integratedpipeline(galaxyname, folder='/raid/madlr/voids/analysis/stackedcubes/'):
	""" Run full pipeline to get: 
		- Systemic stellar kinematics
		- Gas-phase abundances

	Arguments:
		galaxyname (str): name of galaxy
		folder (str): folder where stacked data cubes are stored
	"""

	# Open params
	param = params[galaxyname]

	# Open cube
	c = Cube(galaxyname, folder=folder, verbose=param['verbose'], wcscorr=param['wcscorr'], z=param['z'], EBV=param['EBV'])
	c.integrate(plot=False, covparams=param['covparams'])
	c.stellarfit(plot=True)

	#c.reddening(verbose=False)
	#c.metallicity_Te(verbose=False)

	return

def main():

	integratedpipeline('reines65', folder='/Users/miadelosreyes/Documents/Research/VoidDwarfs/redux/stackedcubes/')

	#c = Cube('reines65', folder='/Users/miadelosreyes/Documents/Research/VoidDwarfs/redux/stackedcubes/', verbose=False, wcscorr=[174.17801 - 174.1787083, 26.727126 - 26.7263583], z=0.0331, EBV=0.0217)
	#c.ellipsefit()
	#c.integrate(plot=False, covparams=[0.108,1.65,80])
	#c.stellarfit(plot=True)
	#c.reddening(verbose=False)
	#c.metallicity_Te(verbose=True)

	return

if __name__ == "__main__":
	main()
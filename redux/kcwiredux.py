# kcwiredux.py
# Script to analyze KCWI data cubes
#
# Created: 6 July 2020
######################################

#Backend for python3 on mahler
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
#plt.interactive('on')

# Change fonts
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import os
from params import params
import cmasher as cmr

# Astropy packages for plotting
import astropy.units as u
from astropy.visualization.wcsaxes import add_scalebar
from astropy.cosmology import FlatLambdaCDM  # needed to compute redshifts
cosmo = FlatLambdaCDM(H0=67.4, Om0=0.315)  # using Planck (2018) params
from astropy.coordinates import Distance

# Packages for binning
import kcwiutils.kcwialign as kcwialign
import vorbin
from vorbin.voronoi_2d_binning import voronoi_2d_binning

# Packages for stellar continuum fitting
import ppxf as ppxf_package
from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
from os import path
import glob
from scipy import ndimage
from scipy import stats

# Packages for emission line fitting
from astropy.modeling import models, fitting
from utils.k_lambda import k_lambda
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

		# Define galaxy name
		self.folder = folder
		self.galaxyname = filename

		print('Initializing cube '+self.galaxyname+'...')

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

		# Open mask cube
		with fits.open(folder+filename+'_mcubes.fits') as mcube:
			self.mask = mcube[0].data

		# Remove negative and inf variances
		var[np.where(var < 0)] = np.mean(var[np.where((np.isfinite(var)))])
		var[np.where((~np.isfinite(var)))] = np.mean(var[np.where((np.isfinite(var)))])

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
		#wvl = np.arange(wvl0,wvl0+N*wvld,wvld)
		wvl = (np.arange(self.header['NAXIS3']) + 1  - self.header['CRPIX3']) * self.header['CD3_3'] + self.header['CRVAL3'] # from Zhuyun's code

		# Do correction for Galactic reddening
		if EBV > 0.:
			Alam = k_lambda(wvl)*EBV
			Alam_array = np.tile(Alam[:, np.newaxis, np.newaxis], (1, self.data.shape[1], self.data.shape[2]))
			self.data = self.data/np.power(10.,(Alam_array/(-2.5)))

		# Do redshift correction
		self.z = z
		self.wvl_zcorr = wvl / (1.+self.z)

		# Define wavelength ranges
		self.wvlsection = np.where((self.wvl_zcorr > sn_wvl[0]) & (self.wvl_zcorr < sn_wvl[1]))[0] # Wavelength range for S/N fitting
		self.goodwvl = np.where((self.wvl_zcorr > wvlrange[0]) & (self.wvl_zcorr < wvlrange[1]))[0] # Wavelength range for stellar template fitting
		self.wvl_cropped = self.wvl_zcorr[self.goodwvl]
		self.data_cropped = self.data[self.goodwvl, :, :]
		self.mask_cropped = self.mask[self.goodwvl, :, :]
		self.goodwvl_sn = np.where((self.wvl_cropped > sn_wvl[0]) & (self.wvl_cropped < sn_wvl[1]))[0] # Use this when cropping wavelength range twice (for both S/N and stellar template fitting)

		# Get zeropoints and deltas for coordinates
		self.ra0 = self.header['CRVAL1']
		self.dec0 = self.header['CRVAL2']
		self.rad = self.header['CD1_1'] # RA degrees per col
		self.decd = self.header['CD2_2'] # Dec degrees per row

		# Plot image for testing
		if verbose:

			# Print header
			print(repr(self.header))

			totaldata = np.ma.sum(self.data, axis=0)
			fig = plt.figure(figsize=(8,8))
			ax = plt.subplot(projection=self.wcs,slices=('x', 'y', 50))
			plt.imshow(totaldata)
			plt.colorbar()
			plt.show()

			# Plot example spectrum
			plt.figure(figsize=(12,5))
			idx = 46
			idy = 40
			plt.plot(self.wvl_zcorr,self.data[:,idx,idy])

			plt.xlabel(r'$\lambda (\AA)$', fontsize=16)
			plt.ylabel('Flux', fontsize=16)
			plt.xlim(3500,5100)

			# Plot error
			testerror = np.sqrt(self.var[:,idx,idy])
			plt.fill_between(self.wvl_zcorr,self.data[:,idx,idy]-testerror,self.data[:,idx,idy]+testerror,facecolor='C0',alpha=0.5,edgecolor='None')
			plt.show()

	def binspaxels(self, params=[1., 1., 60.], verbose=False, targetsn=10., emline=None):
		""" Bin spaxels spatially to increase S/N

			Args:
				params (float list): parameters [alpha, norm, thresh] to use for covar correction
				verbose (bool): if 'True', make test plots
				targetsn (float): target value of S/N
				emline (str): compute emission line S/N of input line; 
							if None (default), compute continuum S/N
		"""

		print('Binning cube...')

		# Define sizes of array
		xsize = np.shape(self.data[0,:,:])[1]
		ysize = np.shape(self.data[0,:,:])[0]

		# Compute signal/noise
		if emline is None:
			signal = np.mean(self.data[self.wvlsection,:,:], axis=0)
			
			# Compute noise as detrended standard deviation
			noise = np.zeros(np.shape(signal))
			for i in range(ysize):
				for j in range(xsize):
					linfit = np.polyfit(self.wvl_zcorr[self.wvlsection],self.data[self.wvlsection,i,j],deg=1)
					poly = np.poly1d(linfit)
					noise[i,j] = np.std(self.data[self.wvlsection,i,j] - np.asarray(poly(self.wvl_zcorr[self.wvlsection])))

			# Define S/N
			sntest = signal/noise
			if verbose:
				plt.imshow(sntest)
				plt.colorbar()
				plt.show()
			np.save('output/'+self.galaxyname+'/contsnr', np.ma.getdata(sntest))
		
		else:
			# Open cubed datafile
			counts = fits.open(self.folder + self.galaxyname + '_icubed.fits')[0].data
			counts = np.ma.array(counts, mask=self.mask)

			# Get central line wavelength
			line = wvldict[emline]

			# TODO: fit line?

			# Define wavelength ranges
			emidx = np.where((self.wvl_zcorr > (line-3*stddev)) & (self.wvl_zcorr < (line+3*stddev)))[0]
			cont1idx = np.where((self.wvl_zcorr < (line-3*stddev)))[0]
			cont2idx = np.where((self.wvl_zcorr > (line+3*stddev)))[0]

			# Compute fluxes
			for i in range(ysize):
				for j in range(xsize):
					emflux = self.data[emidx, i, j]
					contflux1 = self.data[cont1idx, i, j]
					contflux2 = self.data[cont2idx, i, j]
					
					signal = np.sum(emflux - np.mean(np.hstack((contflux1,contflux2)))) / np.sqrt(len(emflux))
					noisecont = (np.std(contflux1) + np.std(contflux2)) / 2. # Continuum noise
					pois = np.random.poisson(size=len(emflux))
					noisepois = np.std(pois/np.sum(pois)*np.sqrt(np.abs(emflux))) # Poisson noise
					noise = np.sqrt(noisecont**2. + noisepois**2.)

			# Define S/N
			sntest = signal/noise
			if verbose:
				plt.imshow(sntest)
				plt.colorbar()
				plt.show()
			np.save('output/'+self.galaxyname+'/'+emline+'snr', np.ma.getdata(sntest))

		# Prep data for binning by making lists that vorbin can read
		y_use, x_use = np.where(sntest > 1) #indices for non-masked pixels
		self.x = x_use
		self.y = y_use
		s = signal[y_use, x_use]
		n = noise[y_use, x_use]

		# Define S/N function
		def snfunc(index, signal, noise):
			sn = np.sum(signal[index])/np.sqrt(np.sum(noise[index]**2))

			# Apply covariance correction
			self.alpha, self.norm, self.threshold = params
			if index.size >= self.threshold:
				correction = self.norm * (1 + self.alpha * np.log(self.threshold))
			else:
				correction = self.norm * (1 + self.alpha * np.log(index.size))

			return sn/correction

		# Do actual binning
		self.binNum, xNode, yNode, xBar, yBar, self.sn, nPixels, scale = voronoi_2d_binning(self.x, self.y, s, n, targetsn, sn_func=snfunc, plot=1, quiet=1)
		if verbose:
			plt.tight_layout()
			plt.show()
		else:
			plt.close()

		# Get list of bins
		self.bins = np.unique(self.binNum)

		# Make cubes to hold stacked data
		self.stacked_spec = np.zeros((len(self.bins), len(self.goodwvl)))
		self.stacked_errs = np.zeros((len(self.bins), len(self.goodwvl)))
		if verbose:
			stacked_data = np.zeros(np.shape(self.data[self.goodwvl,:,:])) # For plotting binned image

		# Make cube to hold bin ID number for each pixel
		self.binIDarray = np.zeros(np.shape(self.data[0,:,:]))

		# Loop over all bins
		for binID in range(len(self.bins)):

			# Get all IDs in that bin
			idx = np.where(self.binNum==self.bins[binID])[0]

			# Get image coords of all the pixels in a bin
			xarray = np.asarray(self.x[idx])
			yarray = np.asarray(self.y[idx])

			# Loop over all pixels in the bin and append the spectrum and variance from each pixel to lists
			binned_spec = []
			binned_var = []
			binned_mask = []
			for i in range(len(xarray)):
				binned_spec.append(self.data[self.goodwvl,yarray[i],xarray[i]].data)
				binned_var.append(self.var[self.goodwvl,yarray[i],xarray[i]].data)
				binned_mask.append(self.mask[self.goodwvl,yarray[i],xarray[i]])

				# Also record the bin ID for each pixel
				self.binIDarray[yarray[i], xarray[i]] = binID

			# Loop again over all pixels in the bin and put the new mean spectra in the list
			if verbose:
				if self.sn[binID] > 1.:
					for i in range(len(xarray)):
						stacked_data[:,yarray[i],xarray[i]] = np.ma.mean(binned_spec)

			# Compute covariance correction
			if len(xarray) >= self.threshold:
				correction = self.norm * (1 + self.alpha * np.log(self.threshold))
			else:
				correction = self.norm * (1 + self.alpha * np.log(len(xarray)))

			#binned_var = np.ma.array(np.asarray(binned_var), mask=np.asarray(binned_mask))

			self.stacked_spec[binID] = np.ma.mean(binned_spec, axis=0) #np.ma.average(binned_spec, axis=0, weights=1/np.power(binned_var,2))
			self.stacked_errs[binID] = np.ma.mean(binned_var, axis=0) * correction**2. / len(xarray) #np.sqrt(1./np.ma.sum(1./np.asarray(binned_var), axis=0))

		# For testing purposes, save arrays of bin IDs and bin errors
		np.save('output/'+self.galaxyname+'/binIDarray', self.binIDarray)
		np.save('output/'+self.galaxyname+'/binerrs', self.stacked_errs)

		# Find luminosity-weighted center
		xcenter = np.sum(self.x * s)/np.sum(s)
		ycenter = np.sum(self.y * s)/np.sum(s)
		print('center:', xcenter, ycenter)

		# Find bin where center is located
		self.centeridx = (np.sqrt((xNode-xcenter)**2. + (yNode-ycenter)**2.)).argmin()
		print(self.centeridx)

		# Plot test figures
		if verbose:

			fig = plt.figure(figsize=(6,6))
			ax = fig.add_subplot(111,projection=self.wcs,slices=('x', 'y', 50))
			ax.set_title('Original white-light image', fontsize=14)
			im=ax.imshow(np.ma.sum(self.data[self.wvlsection,:,:], axis=0), vmin=0)
			fig.colorbar(im, ax=ax)
			plt.savefig('figures/'+self.galaxyname+'/whitelight.pdf', bbox_inches='tight')
			plt.close()

			fig = plt.figure(figsize=(6,6))
			ax = fig.add_subplot(111,projection=self.wcs,slices=('x', 'y', 50))
			ax.set_title('Binned image (with covar correction)', fontsize=14)
			im=ax.imshow(np.ma.sum(stacked_data[self.goodwvl_sn,:,:], axis=0), vmin=0)
			fig.colorbar(im, ax=ax)
			plt.savefig('figures/'+self.galaxyname+'/bintest.pdf', bbox_inches='tight')
			plt.show()

		return

	def prepstellarfit(self, spectrum=None, wvl=None):
		""" Prepare stacked cube for spectral fitting with pPXF. Only need to run this once per galaxy!

			Arguments:
				spectrum, wvl (1D arrays): observed spectrum to use as template (default: use stacked spectrum)
		"""

		print('Preparing for stellar kinematics fit...')

		# Define path to pPXF directory
		ppxf_dir = path.dirname(path.realpath(ppxf_package.__file__))
		print(ppxf_dir)

		# Define spectrum
		spectrum = self.stacked_spec[0] if spectrum is None else spectrum
		wvl = self.wvl_cropped if wvl is None else wvl

		# Define wavelength range
		self.lamRange1 = [wvl[0],wvl[-1]]
		fwhm_gal = 2.4/(1+self.z)  # KCWI instrumental FWHM of ~2.4A

		# Rebin spectrum into log scale to get initial velocity scale
		galaxy, logLam1, velscale = util.log_rebin(self.lamRange1, spectrum)

		# Read the list of filenames from the E-MILES SSP library
		vazdekis = glob.glob(ppxf_dir + '/miles_stellar/s*.fits')
		fwhm_tem = 2.51  # Vazdekis+10 spectra have a constant resolution FWHM of 2.51A.

		# Open template spectrum in order to make get the size of the template array
		hdu = fits.open(vazdekis[0])
		ssp = np.squeeze(hdu[0].data)
		h2 = hdu[0].header
		self.lamRange2 = h2['CRVAL1'] + np.array([0., h2['CDELT1']*(h2['NAXIS1'] - 1)])

		# Crop template to shorter wavelength range to reduce computational time
		lam_temp = h2['CRVAL1'] + h2['CDELT1']*np.arange(h2['NAXIS1'])
		good_lam = (lam_temp > 3500) & (lam_temp < 5500)
		lam_temp = lam_temp[good_lam]
		lamRange_temp = [np.min(lam_temp), np.max(lam_temp)]

		sspNew, ln_lam_temp = util.log_rebin(lamRange_temp, ssp[good_lam], velscale=velscale)[:2]
		self.templates = np.empty((sspNew.size, len(vazdekis)))

		# Convolve observed spectrum with quadratic difference between observed and template resolution.
		# (This is valid if shapes of instrumental spectral profiles are well approximated by Gaussians.)
		fwhm_dif = np.sqrt(np.abs(fwhm_gal**2 - fwhm_tem**2))
		self.sigma = fwhm_dif/2.355/h2['CDELT1'] # Sigma difference in pixels
		galspec = ndimage.gaussian_filter1d(spectrum, self.sigma)

		# Now logarithmically rebin this new observed spectrum
		galaxy, logLam1, velscale = util.log_rebin(self.lamRange1, galspec, velscale=velscale)

		# Open and normalize all the templates
		for j, file in enumerate(vazdekis):
			hdu = fits.open(file)
			ssp = np.squeeze(hdu[0].data)
			sspNew, self.logLam2, velscale_temp = util.log_rebin(lamRange_temp, ssp[good_lam], velscale=velscale)
			self.templates[:, j] = sspNew/np.median(sspNew)  # Normalizes templates

		return

	def ppxf_fit(self, spectrum, noise, verbose=False):
		"""Run pPXF on one spaxel. Note: must run prepstellarfit() first!

		Arguments:
			spectrum, noise (1D array): Observed spectrum and variance array
			verbose (bool): If 'True', make diagnostic plots

		Returns:
			params (float 1D array): Best fit velocity, velocity dispersion, velocity error, velocity dispersion error
		"""

		# Make sure spectrum and noise arrays are the right length
		if spectrum.size != self.wvl_cropped.size:
			raise ValueError('Size of spectrum array %s does not match size of wvl array %s' % (spectrum.size, self.wvl_cropped.size))
		if noise.size != self.wvl_cropped.size:
			raise ValueError('Size of noise array %s does not match size of wvl array %s' % (noise.size, self.wvl_cropped.size))

		# Prep the observed spectrum
		galspec = ndimage.gaussian_filter1d(spectrum, self.sigma)
		galaxy, logLam1, velscale = util.log_rebin(self.lamRange1, galspec)
		galaxy = galaxy/np.median(galaxy)

		# Shift the template to fit the starting wavelength of the galaxy spectrum
		c = 299792.458
		dv = (self.logLam2[0] - logLam1[0])*c  # km/s

		goodPixels = util.determine_goodpixels(logLam1, self.lamRange2, 0)

		# Here the actual fit starts. The best fit is plotted on the screen
		start = [0., 200.]  # (km/s), starting guess for [V, sigma]

		if verbose: plot=True
		else: plot=False

		pp = ppxf(self.templates, galaxy, np.sqrt(noise), velscale, start,
				  goodpixels=goodPixels, plot=plot, moments=2,
				  degree=6, vsyst=dv, clean=False, quiet=True)
		if verbose:
			##plt.show()
			plt.close()

			print("Formal errors:")
			print("     dV    dsigma")
			print("".join("%8.2g" % f for f in pp.error*np.sqrt(pp.chi2)))

			print('Best-fitting redshift z:', (self.z + 1)*(1 + pp.sol[0]/c) - 1)

		return np.asarray([pp.sol[0], pp.sol[1], pp.error[0]*np.sqrt(pp.chi2), pp.error[1]*np.sqrt(pp.chi2), pp.chi2]), np.exp(logLam1), pp.bestfit, galaxy, noise

	def stellarkinematics(self, verbose=False, removekinematics=True, overwrite=False, snr_mask=1, plottest=False, vsigma=True, plotveldist=False):
		""" Do stellar kinematics fitting with pPXF. Note: must run prepstellarfit() first!

			Arguments:
				verbose (bool): if 'True', make diagnostic plots
				removekinematics (bool): if 'True' (default), output a data_norm cube with best-fit stellar template subtracted
				overwrite (bool): if 'True', overwrite existing files
				snr_mask (float): produce velmask.out file marking any bins where S/N > snr_mask
				plottest (bool): if 'True', plot all potentially "bad" bins
				vsigma (bool): if 'True', compute global v/sigma for galaxy
		"""

		# If output files don't exist, run kinematics fitting
		if overwrite or os.path.exists('output/new/'+self.galaxyname+'/velocity.out')==False: # or os.path.exists('output/'+self.galaxyname+'/kinfit.npy')==False:

			# Prep the stellar fit
			self.prepstellarfit()

			# Create data structures to hold final outputs
			self.vel = np.zeros(len(self.bins))
			self.veldisp = np.zeros(len(self.bins))
			self.velmask =np.zeros(len(self.bins), dtype=bool)
			self.vel_err = np.zeros(len(self.bins))
			self.veldisp_err = np.zeros(len(self.bins))
			self.kinematics_fit_bin = np.zeros(np.shape(self.stacked_spec))
			self.kinematics_wvl_bin = np.zeros(np.shape(self.stacked_spec))

			print('Doing stellar kinematics fit...')

			# Compute central velocity and velocity dispersion (useful for comparisons)
			'''
			if np.any(self.stacked_errs[self.centeridx] < 0) or ~np.all(np.isfinite(self.stacked_errs[self.centeridx])):
				self.stacked_errs[self.centeridx][self.stacked_errs[self.centeridx] < 0] = 1e-6
				self.stacked_errs[self.centeridx][~np.isfinite(self.stacked_errs[self.centeridx])] = 1e-6
			params, fitwvl, fit, obs, obserr = self.ppxf_fit(self.stacked_spec[self.centeridx], self.stacked_errs[self.centeridx], verbose=False)
			print('Central velocity:')
			print("\t".join("%.2f" % f for f in [params[0],params[2]]))
			print('Central velocity dispersion:')
			print("\t".join("%.2f" % f for f in [params[1],params[3]]))
			'''
			
			# Loop over all bins
			for binID in tqdm(range(len(self.bins))):

				#if np.any(self.stacked_errs[binID] < 0) or np.any(np.isclose(self.stacked_errs[binID], 1.)):
				#	self.stacked_errs[binID][self.stacked_errs[binID] < 0] = 1e-6
				#	self.stacked_errs[binID][np.isclose(self.stacked_errs[binID],1.)] = 1e-6

				# Do stellar kinematic fit
				params, fitwvl, fit, obs, obserr = self.ppxf_fit(self.stacked_spec[binID], self.stacked_errs[binID], verbose=False)

				if np.isclose(params[2],0.) or np.isclose(params[3],0.): # or (params[2] > 70):
					params = [np.nan, np.nan, np.nan, np.nan, np.nan]

				if plottest:
					if binID==0:
						plt.figure(figsize=(9,3))
						lines = np.array([3726.03, 3728.82, 3970.08, 4101.76, 4340.47, 4363.21, 4861.33, 
								4958.92, 5006.84, 6300.30, 6548.03, 6583.41, 6562.80, 6716.47, 6730.85])
						for line in lines:
							if line < fitwvl[-1]:
								plt.axvspan(line-10., line+10., color='gray', alpha=0.25)
						plt.fill_between(fitwvl, obs-np.sqrt(obserr),obs+np.sqrt(obserr), color='r', alpha=0.8)
						plt.plot(fitwvl, fit, 'k-')
						plt.xlabel(r'$\lambda (\AA)$', fontsize=14)
						plt.ylabel(r'Normalized flux', fontsize=14)
						plt.text(3750, 1.35, 'Central bin: S/N={:.1f}'.format(self.sn[binID]), fontsize=15)
						plt.ylim(0.5,1.5)
						plt.xlim(3700,5100)
						plt.savefig('figures/'+self.galaxyname+'/'+'centerspec.pdf', bbox_inches='tight') 
						#plt.show()

					if binID==15:
						plt.figure(figsize=(9,3))
						lines = np.array([3726.03, 3728.82, 3970.08, 4101.76, 4340.47, 4363.21, 4861.33, 
								4958.92, 5006.84, 6300.30, 6548.03, 6583.41, 6562.80, 6716.47, 6730.85])
						for line in lines:
							if line < fitwvl[-1]:
								plt.axvspan(line-10., line+10., color='gray', alpha=0.25)
						plt.fill_between(fitwvl, obs-np.sqrt(obserr),obs+np.sqrt(obserr), color='r', alpha=0.8)
						plt.plot(fitwvl, fit, 'k-')
						plt.xlabel(r'$\lambda (\AA)$', fontsize=14)
						plt.ylabel(r'Normalized flux', fontsize=14)
						plt.text(3750, 1.35, 'Outer bin: S/N={:.1f}'.format(self.sn[binID]), fontsize=15)
						plt.ylim(0.5,1.5)
						plt.xlim(3700,5100)
						plt.savefig('figures/'+self.galaxyname+'/'+'outerspec.pdf', bbox_inches='tight') 
						#plt.show()

				# Save data from each bin
				self.vel[binID] = params[0]
				self.veldisp[binID] = params[1]
				self.vel_err[binID] = params[2]
				self.veldisp_err[binID] = params[3]
				if self.sn[binID] > snr_mask:
					self.velmask[binID] = 1

				# Put fit for each bin into an array
				self.kinematics_fit_bin[binID] = fit
				self.kinematics_wvl_bin[binID] = fitwvl

			# Compute systemic velocity
			goodidx = np.where(~np.isnan(self.vel) & ~np.isnan(self.veldisp))
			flux = np.nansum(self.stacked_spec[:,self.goodwvl_sn], axis=1)
			systvel = np.average(self.vel[goodidx], weights=1./(self.vel_err[goodidx]**2.))
			print("systvel: {:.2f}".format(systvel))

			# Subtract systemic velocity
			self.vel += -systvel

			np.savetxt('output/'+self.galaxyname+'/velocity.out', self.vel)
			np.savetxt('output/'+self.galaxyname+'/veldisp.out', self.veldisp)
			np.savetxt('output/'+self.galaxyname+'/vel_err.out', self.vel_err)
			np.savetxt('output/'+self.galaxyname+'/veldisp_err.out', self.veldisp_err)
			np.savetxt('output/'+self.galaxyname+'/velmask.out', self.velmask)
			np.save('output/'+self.galaxyname+'/kinfit_bin', self.kinematics_fit_bin)
			np.save('output/'+self.galaxyname+'/kinwvl_bin', self.kinematics_wvl_bin)

		# Else, just open existing kinematics fit files
		#else:
		#	self.prepstellarfit()
		#	self.kinematics_fit_bin = np.load('output/'+self.galaxyname+'/kinfit_bin.npy')
		#	self.kinematics_wvl_bin = np.load('output/'+self.galaxyname+'/kinwvl_bin.npy')

		# Else, try computing vsigma from existing kinematics maps
		else:

			# Define data
			self.vel = np.loadtxt('output/new/'+self.galaxyname+'/velocity.out')
			self.veldisp = np.loadtxt('output/new/'+self.galaxyname+'/veldisp.out')
			self.vel_err = np.loadtxt('output/new/'+self.galaxyname+'/vel_err.out')
			self.veldisp_err = np.loadtxt('output/new/'+self.galaxyname+'/veldisp_err.out')
			self.velmask = np.loadtxt('output/new/'+self.galaxyname+'/velmask.out')

			# Compute sigma
			goodidx = np.where(~np.isnan(self.vel) & ~np.isnan(self.veldisp) & (self.veldisp_err > 1e-5) & (self.veldisp > 1.)) # & (self.veldisp > self.veldisp_err))
			flux = np.nansum(self.stacked_spec[:,self.goodwvl_sn], axis=1)

			sigma = np.average(self.veldisp[goodidx], weights=flux[goodidx])
			sigma_err = np.sqrt(np.sum( self.veldisp_err[goodidx]**2. * (flux[goodidx]/np.sum(flux[goodidx]))**2. ))
			print("sigma: {:.2f} \pm {:.2f}".format(sigma, sigma_err))
			
			# Compute vmax
			if self.galaxyname=='2502521':
				goodidx = np.where((self.vel_err > 0.) & (self.veldisp_err > 0) & (self.vel_err < 350.) & (self.velmask==True) & (self.vel_err < np.max(np.abs(self.vel))))
			else:
				goodidx = np.where((self.vel_err > 0.) & (self.veldisp_err > 0) & (self.velmask==1) & (self.vel_err < np.max(np.abs(self.vel))))
			
			Niter = 10000
			vmaxes = np.zeros(Niter)
			vsigmas = np.zeros(Niter)
			for iteration in range(Niter):
				# Compute vmax
				velocities = np.random.default_rng().normal(loc=self.vel[goodidx], scale=self.vel_err[goodidx])
				maxvel = np.median([np.percentile(velocities, 95), np.max(velocities)])
				minvel = np.median([np.min(velocities), np.percentile(velocities, 5)])
				vmaxes[iteration] = 0.5*(maxvel - minvel)

				# Compute v/sigma
				newsigma = np.random.default_rng().normal(loc=sigma, scale=sigma_err)
				vsigmas[iteration] = vmaxes[iteration]/newsigma

			# Plot vmax distribution and compute global vmax
			controlnames = np.asarray(['control757','control801','control872','control842','PiscesA','PiscesB','control751','control775','control658'])
			fullnames = np.asarray(['AGC 112504','LEDA 3524','IC 0225','LEDA 101427','Pisces A','Pisces B','UM 240','SHOC 150','SDSS J0133+1342'])

			if self.galaxyname in controlnames:
				name = fullnames[np.where(controlnames==self.galaxyname)[0]][0]
			else:
				name = self.galaxyname

			goodvsigma = np.where((vsigmas > 0.) & (vsigmas < 5.))[0]
			vsigmas = vsigmas[goodvsigma]

			def plotsmoothhist(array, filename):
				fig = plt.figure(figsize=(7,5))
				ax = fig.add_subplot(111)
				hist, bins, patches = plt.hist(array, bins=50, density=True)

				# Compute global vmax
				smoothhist = stats.gaussian_kde(array)
				max = bins[np.argmax(smoothhist(bins))]
				err_up = np.percentile(array, 84)
				err_lo = np.percentile(array, 16)

				plt.plot(bins, smoothhist(bins), 'b-')
				plt.axvline(max, color='k', ls='--')
				plt.axvspan(err_lo, err_up, color='r', alpha=0.3)
				if filename=='vmax':
					plt.xlabel(r'$v_{\mathrm{max}}$ (km/s)', fontsize=20)
				if filename=='vsigma':
					plt.xlabel(r'$v_{\mathrm{rot}}/\sigma_{\star}$', fontsize=20)
					plt.xlim(0,5)
				plt.ylabel(r'Normalized $N$', fontsize=20)
				plt.text(0.75, 0.9, name, fontsize=18, fontweight='extra bold', transform=ax.transAxes, bbox=dict(alpha=0.5, facecolor='white', edgecolor='black'))
				plt.yticks(fontsize=14)
				plt.xticks(fontsize=14)
				plt.savefig('figures/distributions/'+filename+'_'+self.galaxyname+'.pdf', bbox_inches='tight')
				plt.close()
				#plt.show()

				return max, err_up-max, max-err_lo

			vmax, vmax_up, vmax_lo = plotsmoothhist(vmaxes, 'vmax')
			vsigma, vsigma_up, vsigma_lo = plotsmoothhist(vsigmas, 'vsigma')
			print(r"vmax: {:.2f} + {:.2f} - {:.2f}".format(vmax, vmax_up, vmax_lo))
			print(r"vsigma: {:.2f} + {:.2f} - {:.2f}".format(vsigma, vsigma_up, vsigma_lo))

			# Plot velocity distribution
			'''
			fig = plt.figure(figsize=(7,5))
			ax = fig.add_subplot(111)
			ax.hist(self.vel, bins=10, color='cornflowerblue')
			plt.axvline(maxv, linestyle='--', color='r')
			plt.axvline(minv, linestyle='--', color='r')
			plt.axvspan(maxv - maxv_err_lo, maxv + maxv_err_up, color='r', alpha=0.3)
			plt.axvspan(minv - minv_err_lo, minv + minv_err_up, color='r', alpha=0.3)
			plt.axvline(0, linestyle=':', lw=2, color='k')
			plt.xlabel(r'$v_{i}$', fontsize=20)
			plt.ylabel(r'$N$', fontsize=20)
			plt.text(0.1, 0.9, name, fontsize=18, fontweight='extra bold', transform=ax.transAxes, bbox=dict(alpha=0.5, facecolor='white', edgecolor='black'))
			plt.yticks(fontsize=14)
			plt.xticks(fontsize=14)
			plt.savefig('figures/veldist/'+self.galaxyname+'.pdf', bbox_inches='tight')
			plt.close()
			'''

		# Remove best-fit stellar template from each spaxel
		if removekinematics:

			print('Normalizing data by best-fit stellar template for each spaxel...')

			# Define sizes of array
			xsize = np.shape(self.data[0,:,:])[0]
			ysize = np.shape(self.data[0,:,:])[1]

			# Prep array for output
			self.data_norm = np.zeros(np.shape(self.data_cropped))

			# Loop over all bins
			for binID in range(len(self.bins)):

				# Get all IDs in that bin
				idx = np.where(self.binNum==self.bins[binID])[0]

				# Get image coords of all the pixels in a bin
				xarray = np.asarray(self.x[idx])
				yarray = np.asarray(self.y[idx])

				# Loop over all pixels in the bin and subtract best-fit stellar continuum
				for i in range(len(xarray)):

					# Smooth observed spectrum to match template
					spectrum = self.data_cropped[:,yarray[i],xarray[i]]
					galspec = ndimage.gaussian_filter1d(spectrum, self.sigma)

					# Save log-rebinned spectrum
					galaxy, _, _ = util.log_rebin(self.lamRange1, galspec)

					# Subtract kinematics fit
					self.data_norm[:,yarray[i],xarray[i]] = galaxy - self.kinematics_fit_bin[binID]*np.median(galaxy)

			# Save file
			np.save('output/'+self.galaxyname+'/data_norm', self.data_norm)

			print('Normalizing data by best-fit stellar template for each bin...')

			# Prep array for output
			self.data_norm_bin = np.zeros(np.shape(self.stacked_spec))

			# Loop over all bins
			for binID in range(len(self.bins)):

				# Smooth observed spectrum to match template
				spectrum = self.stacked_spec[binID]
				galspec = ndimage.gaussian_filter1d(spectrum, self.sigma)

				# Save log-rebinned spectrum
				galaxy, _, _ = util.log_rebin(self.lamRange1, galspec)

				# Subtract kinematics fit from binned data
				self.data_norm_bin[binID] = galaxy - self.kinematics_fit_bin[binID]*np.median(galaxy)

			# Save file
			np.save('output/'+self.galaxyname+'/data_norm_bin', self.data_norm_bin)

			# Plot image for testing
			if verbose:

				# Plot example spectrum
				plt.figure(figsize=(12,5))
				plt.title('single bin')
				idx = 10
				plt.plot(self.kinematics_wvl_bin[idx], self.data_norm_bin[idx])
				plt.xlabel(r'$\lambda (\AA)$', fontsize=16)
				plt.ylabel('Flux', fontsize=16)
				plt.xlim(3500,5100)

				# Plot another example spectrum
				# Get all IDs in that bin
				idx = np.where(self.binNum==self.bins[0])[0][0]

				# Get image coords of one of the pixels in a bin
				xarray = np.asarray(self.x[idx])
				yarray = np.asarray(self.y[idx])

				plt.figure(figsize=(12,5))
				plt.title('single spaxel')
				plt.plot(self.kinematics_wvl_bin[0], self.data_norm[:,yarray, xarray])
				plt.xlabel(r'$\lambda (\AA)$', fontsize=16)
				plt.ylabel('Flux', fontsize=16)
				plt.xlim(3500,5100)

				# Plot error
				#plt.show()

				#plt.show()

		return

	def plotkinematics(self, vel='velocity.out', veldisp='veldisp.out', vel_err='vel_err.out', 
		veldisp_err='veldisp_err.out', velmask='velmask.out', vellimit=None, veldisplimit=None, ploterrs=False):
		""" Make kinematic plots.

			Arguments:
				vel, veldisp, vel_err, veldisp_err (2D arrays): kinematics measurements
					(If set to 'None', will use output directly from stellarkinematics())
				velmask (2D bool array): mask marking bins where total S/N > some value (1 = good, 0 = bad)
				instdisp (bool): if 'True', subtract (in quadrature) instrument dispersion from vel dispersion
				vellimit, veldisplimit (float, float list): limits for velocity and velocity dispersion maps
					(velocity map goes from [-vellimit, vellimit], dispersion map goes from [veldisplimit[0], veldisplimit[1]])
				ploterrs (bool): if 'True', also plot and save velocity/dispersion error maps
		"""

		print('Plotting stellar kinematics maps...')

		# Define data
		vel = self.vel #np.loadtxt('output/'+self.galaxyname+'/'+vel) if vel=='velocity.out' else self.vel
		veldisp = self.veldisp #np.loadtxt('output/'+self.galaxyname+'/'+veldisp) if veldisp=='veldisp.out' else self.veldisp
		vel_err = self.vel_err #np.loadtxt('output/'+self.galaxyname+'/'+vel_err) if vel_err=='vel_err.out' else self.vel_err
		veldisp_err = self.veldisp_err #np.loadtxt('output/'+self.galaxyname+'/'+veldisp_err) if veldisp_err=='veldisp_err.out' else self.veldisp_err

		if velmask=='velmask.out': 
			velmask = np.loadtxt('output/new/'+self.galaxyname+'/'+velmask)
		else:
			velmask = self.velmask

		def unpack_binneddata(array):
			"""Unpack binned data into spaxels"""

			unpacked_array = np.zeros_like(self.data[0,:,:])

			# Loop over all bins
			for binID in range(len(self.bins)-1): #-1 (have to add for 955106)

				# Get all IDs in that bin
				idx = np.where(self.binNum==self.bins[binID])[0]

				# Get image coords of all the pixels in a bin
				xarray = np.asarray(self.x[idx])
				yarray = np.asarray(self.y[idx])

				# Loop over all pixels in the bin
				for i in range(len(xarray)):

					# Check if any spaxels are masked
					#if np.all(self.mask_cropped[:,yarray[i], xarray[i]]==False):
					unpacked_array[yarray[i],xarray[i]] = array[binID]	

					#else:
					#	print(binID)
					#	unpacked_array[yarray[i],xarray[i]]	= np.nan			

			return unpacked_array

		def plot(array, error=None, sn=None, limits=None, mask=None, cmap='magma', title=None, plotname='', showplot=False): 

			# Copy array
			copy = np.copy(unpack_binneddata(array))

			# Do S/N cut on data
			#if error is not None and sn is not None:
			#	error = np.copy(unpack_binneddata(error))
			#	idx = np.where(np.abs(copy/error) < sn)
			#	copy[idx] = np.nan

			# Mask any bad data
			if mask is not None:
				mask = np.array(unpack_binneddata(mask), dtype=bool)
				copy[~mask] = np.nan

				mask = np.array(unpack_binneddata(self.vel_err) > np.max(np.abs(self.vel)))
				copy[mask] = np.nan

			# Mask bad measurements of sigma
			if plotname=='veldisp':
				mask = np.array(unpack_binneddata(self.veldisp_err) < 1e-5)
				copy[mask] = np.nan

				mask = np.array(copy < 1)
				copy[mask] = np.nan

			# Remove bad bin from 2502521
			if self.galaxyname=='2502521':
				mask = np.array(unpack_binneddata(self.vel_err) > 350.)
				copy[mask] = np.nan

			if showplot:
				fig = plt.figure(figsize=(5,5))
				ax = plt.subplot(projection=self.wcs,slices=('x', 'y', 50))
				#ax.set_title(self.galaxyname, fontsize=18, weight='bold')
				plt.grid(color='black', ls='dotted')
				if limits is None:
					im = ax.imshow(copy, cmap=cmap)
				else:
					im = ax.imshow(copy, vmin=limits[0], vmax=limits[1], cmap=cmap)
				cb = fig.colorbar(im, ax=ax)
				cb.set_label(label=title, size=18, weight='bold')

				plt.savefig('figures/'+self.galaxyname+'/'+plotname+'.pdf', bbox_inches='tight') 
				plt.show()
				#plt.close()

			return copy

		if ploterrs:
			plot(vel, title=r'$V$ (km/s)', plotname='vel', mask=velmask, showplot=True)
			plot(vel_err, title=r'$V$ error (km/s)', plotname='velerr', mask=velmask, showplot=True)
			#plot(veldisp_err, title=r'$\sigma$ error (km/s)', plotname='veldisperr', mask=velmask, showplot=True)

		controlnames = np.asarray(['control757','control801','control872','control842','PiscesA','PiscesB','control751','control775','control658'])
		fullnames = np.asarray(['AGC 112504','LEDA 3524','IC 0225','LEDA 101427','Pisces A','Pisces B','UM 240','SHOC 150','SDSS J0133+1342'])

		if self.galaxyname in controlnames:
			name = fullnames[np.where(controlnames==self.galaxyname)[0]][0]
		else:
			name = self.galaxyname

		fig = plt.figure(figsize=(15,5))

		# Plot white-light image
		ax0 = fig.add_subplot(131,projection=self.wcs,slices=('x', 'y', 50))
		ax0.grid(color='black', ls='dotted')
		ax0.coords[0].set_axislabel(' ')
		ax0.coords[1].set_axislabel(' ')
		im=ax0.imshow(np.ma.sum(self.data[self.wvlsection,:,:], axis=0), vmin=0, cmap=cmr.neutral_r)
		cb = fig.colorbar(im, ax=ax0, pad=0.) #, shrink=0.9)
		cb.set_label(label='Flux', size=18, weight='bold')
		ax0.text(0.1, 0.9, name, fontsize=18, fontweight='extra bold', transform=ax0.transAxes, bbox=dict(alpha=0.5, facecolor='white', edgecolor='black'))

		# Test physical size of region
		test_angle = cosmo.angular_diameter_distance(self.z)
		testdist = test_angle.to(u.kpc)/206265.
		print('test', 20*testdist, 16.5*testdist)  # kpc / arcsec

		# Create scalebar 
		distance = Distance(z=self.z, cosmology=cosmo).to(u.kpc)
		if (testdist*16.5).value < 1:
			scalebar_length = 100 * u.pc 
			scalebar_label = "100 pc"
		else:
			scalebar_length = 1 * u.kpc
			scalebar_label = "1 kpc"
		scalebar_angle = (scalebar_length / distance).to(u.deg, equivalencies=u.dimensionless_angles())

		test_dist =  cosmo.arcsec_per_kpc_proper(self.z)
		#print('test', test_dist)  # arcsec per kpc

		# Add a scale bar
		add_scalebar(ax0, scalebar_angle, label=scalebar_label, color="black")

		# Plot velocity
		vcopy = plot(vel, error=vel_err, limits=[-vellimit,vellimit], cmap='coolwarm', title=r'$V$ (km/s)', mask=velmask, plotname='vel')
		ax1 = fig.add_subplot(132,projection=self.wcs,slices=('x', 'y', 50))
		ax1.grid(color='black', ls='dotted')
		ax1.coords[0].set_axislabel(' ')
		ax1.coords[1].set_axislabel(' ')
		im = ax1.imshow(vcopy, vmin=-vellimit, vmax=vellimit, cmap='coolwarm')
		cb = fig.colorbar(im, ax=ax1, pad=0.) #, shrink=0.9)
		cb.set_label(label=r'$v$ (km/s)', size=18, weight='bold')

		# Plot veldisp
		vdispcopy = plot(veldisp, error=veldisp_err, limits=[veldisplimit[0], veldisplimit[1]], cmap=cmr.ember, title=r'$\sigma$ (km/s)', mask=velmask, plotname='veldisp')
		ax2 = fig.add_subplot(133,projection=self.wcs,slices=('x', 'y', 50))
		ax2.grid(color='black', ls='dotted')
		ax2.coords[0].set_axislabel(' ')
		ax2.coords[1].set_axislabel(' ')
		im = ax2.imshow(vdispcopy, vmin=veldisplimit[0], vmax=veldisplimit[1], cmap=cmr.bubblegum)
		cb = fig.colorbar(im, ax=ax2, pad=0.) #, shrink=0.9)
		cb.set_label(label=r'$\sigma_{\star}$ (km/s)', size=18, weight='bold')

		fig.tight_layout(pad=4.0)
		plt.savefig('figures/kinematics/'+self.galaxyname+'.pdf', bbox_inches='tight')
		#plt.show()
		plt.close()

		return

	def make_emline_map(self, datanorm, wvlnorm, errnorm, line_name, velmask='velmask.out', snrmask=3, xlim=10., overwrite=False, binned=False):
		""" Fits gas emission lines and makes emission line maps.

			Arguments:
				datanorm, wvlnorm, errnorm (3D arrays): data, wavelength, and error arrays to be fit
				line_name (string): name of line to fit
				velmask (2D bool array): mask marking bins where total S/N > 8 (1 = good, 0 = bad)
				snrmask (float): if not None, do continuum S/N cut on individual spaxels
				xlim (float): in Angstroms, half of wavelength range about line center to fit
				overwrite (bool): if 'True', overwrite any existing files
				binned (bool): if 'True', make emission line map of individual bins

			Returns:
				lineflux, width (2D arrays): output line flux and width maps
		"""

		print('Making emission line map of '+line_name+'...')

		# Output path
		outpath = 'output/'+self.galaxyname+'/'+line_name
		if binned:
			outpath = outpath+'_binned'

		# Check if data already exists
		flux_file = outpath+'_flux.out'
		fluxerr_file = outpath+'_fluxerr.out'
		width_file = outpath+'_std.out'
		widtherr_file = outpath+'_stderr.out'
		snr_file = outpath+'_snr.out'
		if overwrite==False and os.path.exists(flux_file) and os.path.exists(width_file) and os.path.exists(snr_file):
			return np.loadtxt(flux_file), np.loadtxt(fluxerr_file), np.loadtxt(width_file), np.loadtxt(widtherr_file), np.loadtxt(snr_file)

		# Define where to measure emission lines based on velocity measurements
		if velmask =='velmask.out':
			velmask = np.loadtxt('output/'+self.galaxyname+'/'+velmask)
		else:
			velmask = self.velmask

		# Define where to measure emission lines based on continuum S/N
		if snrmask is not None:
			snrfile = np.load('output/'+self.galaxyname+'/contsnr.npy')
			snrtest = np.ones(np.shape(snrfile), dtype=bool)
			snrtest[snrfile < snrmask] = False

		else:
			snrtest = np.ones(np.shape(velmask), dtype=bool)

		# Get central line wavelength
		line = wvldict[line_name]

		# Prep arrays to hold outputs
		if binned == False:
			lineflux = np.zeros(np.shape(datanorm[0,:,:]))
			lineflux_err = np.zeros(np.shape(datanorm[0,:,:]))
			width = np.zeros(np.shape(datanorm[0,:,:]))
			width_err = np.zeros(np.shape(datanorm[0,:,:]))
			snr = np.zeros(np.shape(datanorm[0,:,:]))
		else:
			lineflux = np.zeros(len(self.bins))
			lineflux_err = np.zeros(len(self.bins))
			width = np.zeros(len(self.bins))
			width_err = np.zeros(len(self.bins))
			snr = np.zeros(len(self.bins))

		def fitline(wvl, flux, err, line):
			''' Function to fit line '''

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

			if integral > 1e-3 and stddev < xlim/2. and np.abs(mean - line) < xlim/2.: # and amp_err < amp: #and gaussian_fit[0].stddev.value*2.355 > 2.4

					# Compute SNR
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

			return amp, mean, stddev, amp_err, mean_err, stddev_err, integral, signal, noise

		if binned==False:
			# Loop over all spaxels
			for i in range(len(datanorm[0,:,0])):
				for j in range(len(datanorm[0,0,:])):
					if velmask[i,j] and snrtest[i,j]:

						# Crop flux, wvl arrays to only contain the area around the line
						idx = np.where((wvlnorm[:,i,j] > (line-xlim)) & (wvlnorm[:,i,j] < (line+xlim)))[0]
						wvl = wvlnorm[:,i,j][idx]
						flux = datanorm[:,i,j][idx]
						err = errnorm[:,i,j][idx]

						amp, mean, stddev, amp_err, mean_err, stddev_err, integral, signal, noise = fitline(wvl, flux, err, line)

						if signal > 0.:
							lineflux[i,j] = integral
							lineflux_err[i,j] = integral * np.sqrt(2.*np.pi)*np.sqrt((amp_err/amp)**2. + (stddev_err/stddev)**2.)
							width[i,j] = stddev
							width_err[i,j] = stddev_err

						if signal/noise > 0.:
							snr[i,j] = signal/noise

			fig, ax = plt.subplots()
			im = ax.imshow(snr, cmap='viridis', interpolation='nearest')
			fig.colorbar(im, ax=ax)
			#plt.show()

		else:
			# Loop over all bins
			for binID in range(len(self.bins)):

				# Crop flux, wvl arrays to only contain the area around the line
				idx = np.where((wvlnorm[binID] > (line-xlim)) & (wvlnorm[binID] < (line+xlim)))[0]
				wvl = wvlnorm[binID][idx]
				flux = datanorm[binID][idx]
				err = errnorm[binID][idx]

				amp, mean, stddev, amp_err, mean_err, stddev_err, integral, signal, noise = fitline(wvl, flux, err, line)

				if signal > 0.:
					lineflux[binID] = integral
					lineflux_err[binID] = integral * np.sqrt(2.*np.pi)*np.sqrt((amp_err/amp)**2. + (stddev_err/stddev)**2.)
					width[binID] = stddev
					width_err[binID] = stddev_err

				if signal/noise > 0.:
					snr[binID] = signal/noise

		# Save data
		np.savetxt(flux_file, lineflux)
		np.savetxt(fluxerr_file, lineflux_err)
		np.savetxt(width_file, width)
		np.savetxt(widtherr_file, width_err)
		np.savetxt(snr_file, snr)

		return lineflux, lineflux_err, width, width_err, snr

	def reddening(self, verbose=False, overwrite=False, binned=False):
		""" Compute and apply reddening correction to each spaxel. Only need to run this once per galaxy!

			Arguments:
				verbose (bool): if 'True', make diagnostic plots
				overwrite (bool): if 'True', overwrite existing data files
				binned (bool): if 'True', use binned spectra instead of individual spaxels
		"""

		print('Computing Balmer decrement...')

		# Get data
		if binned:
			errors = self.stacked_errs
			try:
				data_norm = self.data_norm_bin
				kinematics_wvl = self.kinematics_wvl_bin
			except AttributeError:
				data_norm = np.load('output/'+self.galaxyname+'/'+'data_norm_bin.npy')
				kinematics_wvl = np.load('output/'+self.galaxyname+'/'+'kinwvl_bin.npy')

		else:
			# Errors for individual spaxels
			errors = np.sqrt(self.var[self.goodwvl, :, :])

			try:
				data_norm = self.data_norm
				kinematics_wvl = self.kinematics_wvl
			except AttributeError:
				data_norm = np.load('output/'+self.galaxyname+'/'+'data_norm.npy')
				kinematics_wvl = np.load('output/'+self.galaxyname+'/'+'kinwvl.npy')

			# Mask data
			data_norm = np.ma.array(data_norm, mask=self.mask_cropped)
			kinematics_wvl = np.ma.array(kinematics_wvl, mask=self.mask_cropped)
			errors = np.ma.array(errors, mask=self.mask_cropped)

		# Get Balmer line maps
		resultHgamma = self.make_emline_map(data_norm, kinematics_wvl, errors, 'Hgamma', overwrite=overwrite, binned=binned)
		resultHbeta = self.make_emline_map(data_norm, kinematics_wvl, errors, 'Hbeta', overwrite=overwrite, binned=binned)

		# Make array to hold E(B-V) iterations
		Niter = 1000
		if binned:
			ebv = np.zeros((Niter, len(self.bins)))
		else:
			ebv = np.zeros((Niter, self.data.shape[1], self.data.shape[2]))

		# Compute relevant quantities using MC method to get errors
		print('Computing E(B-V)...')
		for i in tqdm(range(Niter)):

			Hbeta = np.random.default_rng().normal(loc=resultHbeta[0], scale=resultHbeta[1])
			Hgamma = np.random.default_rng().normal(loc=resultHgamma[0], scale=resultHgamma[1])

			# Compute Balmer decrement
			balmer = Hgamma/Hbeta

			# Compute E(B-V)
			balmer0 = 0.468 # Intrinsic Hgamma/Hbeta ratio (assuming Case B recombination, T=10^4K, electron density 100/cm^3)
			if binned:
				ebv[i,:] = np.log10(balmer/balmer0)/(-0.4*(k_lambda(wvldict['Hgamma'])-k_lambda(wvldict['Hbeta'])))
			else:
				ebv[i,:,:] = np.log10(balmer/balmer0)/(-0.4*(k_lambda(wvldict['Hgamma'])-k_lambda(wvldict['Hbeta'])))

		if binned:

			# Make arrays to hold binned data
			ebv_unbinned = np.zeros((Niter, self.data.shape[1], self.data.shape[2]))
			Hbeta_unbinned = np.zeros(np.shape(self.data[0,:,:]))
			Hgamma_unbinned = np.zeros(np.shape(self.data[0,:,:]))

			# Loop over all bins
			for binID in range(len(self.bins)):

				# Get all IDs in that bin
				idx = np.where(self.binNum==self.bins[binID])[0]

				# Get RA/Dec of all the pixels in a bin
				xarray = np.asarray(self.x[idx])
				yarray = np.asarray(self.y[idx])

				# Loop again over all pixels in the bin and put the correct emline values in the array
				for i in range(len(xarray)):
					Hbeta_unbinned[yarray[i], xarray[i]] = resultHbeta[0][binID]
					Hgamma_unbinned[yarray[i], xarray[i]] = resultHgamma[0][binID]
					ebv_unbinned[:, yarray[i], xarray[i]] = ebv[:, binID]

			ebv = ebv_unbinned

		# Compute E(B-V) mean and errors
		ebv_mean = np.nanmean(ebv, axis=0)
		ebv_err = np.nanstd(ebv, axis=0)

		if verbose:

			# Test error propagation
			testidx = 41
			testidy = 40
			if binned:
				testbalmer = Hgamma_unbinned/Hbeta_unbinned
			else:
				testbalmer = resultHgamma[0]/resultHbeta[0]
			testebv = np.log10(testbalmer/balmer0)/(-0.4*(k_lambda(wvldict['Hgamma'])-k_lambda(wvldict['Hbeta'])))

			print(np.shape(Hbeta))
			print(np.shape(testebv))
			plt.hist(ebv[:,testidx,testidy])
			plt.axvline(testebv[testidx,testidy], color='r')
			plt.axvline(ebv_mean[testidx,testidy], color='k', linestyle='--')
			plt.axvspan(ebv_mean[testidx,testidy]-ebv_err[testidx,testidy],ebv_mean[testidx,testidy]+ebv_err[testidx,testidy], color='gray', alpha=0.25)
			#plt.show()

			# Test Balmer line calculations
			'''
			fig = plt.figure(figsize=(15,6))

			ax = fig.add_subplot(131) #,projection=self.wcs,slices=('x', 'y', 50))
			ax.set_title(r'H$\beta$')
			im=ax.imshow(Hbeta, vmin=0, vmax=1)
			fig.colorbar(im, ax=ax)

			ax = fig.add_subplot(132) #,projection=self.wcs,slices=('x', 'y', 50))
			ax.set_title(r'H$\gamma$')
			im=ax.imshow(Hgamma, vmin=0, vmax=1)
			fig.colorbar(im, ax=ax)

			ax = fig.add_subplot(133) #,projection=self.wcs,slices=('x', 'y', 50))
			ax.set_title(r'$\mathrm{H}\gamma/\mathrm{H}\beta$')
			im=ax.imshow(balmer, vmax=0.8)
			fig.colorbar(im, ax=ax)

			#plt.show()
			'''

			# Test E(B-V)
			fig = plt.figure(figsize=(15,6))

			ax = fig.add_subplot(131) #,projection=self.wcs,slices=('x', 'y', 50))
			ax.set_title(r'$E(B-V)$')
			im = ax.imshow(testebv, cmap='viridis', interpolation='nearest')
			fig.colorbar(im, ax=ax)

			ax = fig.add_subplot(132) #,projection=self.wcs,slices=('x', 'y', 50))
			ax.set_title(r'$E(B-V)$ (propagated)')
			im = ax.imshow(ebv_mean, cmap='viridis', interpolation='nearest')
			fig.colorbar(im, ax=ax)

			ax = fig.add_subplot(133) #,projection=self.wcs,slices=('x', 'y', 50))
			ax.set_title(r'$E(B-V)$ error')
			im = ax.imshow(ebv_err, cmap='viridis', interpolation='nearest')
			fig.colorbar(im, ax=ax)

			plt.savefig('figures/'+self.galaxyname+'/EBVtest.png', bbox_inches='tight')
			#plt.show()

		return
		print('Applying reddening correction to all spaxels...')

		# Initialize output array
		self.data_dered = np.copy(data_norm)

		# Test how many spaxels have reasonable E(B-V) values
		velmask = np.loadtxt('output/'+self.galaxyname+'/velmask.out')
		print('Total spaxels in mask:', len(velmask[velmask>0]))
		print('Spaxels with E(B-V)>0:', len(np.where((ebv_mean>0))[0]))
		print('Spaxels with reasonable E(B-V):', len(np.where(np.logical_and(ebv_mean>0, ebv_mean<1))[0]))

		# Loop over all spaxels
		for i in range(len(data_norm[0,:,0])):
			for j in range(len(data_norm[0,0,:])):
				if np.isfinite(ebv_mean[i,j]): #Ebv[i,j] > 0. and Ebv[i,j] < 1. and 

					# Get flux, wvl arrays for each spectrum
					wvl = kinematics_wvl[:,i,j]
					flux = data_norm[:,i,j]

					# Apply reddening correction
					Alam = k_lambda(wvl)*ebv_mean[i,j]
					self.data_dered[:,i,j] = flux/np.power(10.,(Alam/(-2.5)))

		if verbose:
			plt.figure(figsize=(12,5))
			idx = 40
			idy = 36
			plt.plot(kinematics_wvl[:,idx,idy],self.data_dered[:,idx,idy], label='De-reddened')
			plt.plot(kinematics_wvl[:,idx,idy],data_norm[:,idx,idy], label='Original')
			plt.xlabel(r'$\lambda (\AA)$', fontsize=16)
			plt.ylabel('Flux', fontsize=16)
			plt.legend()
			plt.xlim(3500,5100)
			#plt.show()

		np.save('output/'+self.galaxyname+'/data_dered', self.data_dered)

		return

	def metallicity_Te(verbose=False, overwrite=False):
		""" Compute electron temperature metallicities

			Arguments:
				verbose (bool): if 'True', make diagnostic plots
				overwrite (bool): if 'True', overwrite existing data files
		"""

		print('Electron temperature...')

		# Get de-reddened data
		try:
			data_norm = self.data_dered
			kinematics_wvl = self.kinematics_wvl
		except AttributeError:
			data_norm = np.load('output/'+self.galaxyname+'/'+'data_dered.npy')
			kinematics_wvl = np.load('output/'+self.galaxyname+'/'+'kinwvl.npy')

		# Get OIII line maps
		resultOIII4363 = self.make_emline_map(data_norm, kinematics_wvl, 'OIII4363', overwrite=overwrite)
		resultOIII4959 = self.make_emline_map(data_norm, kinematics_wvl, 'OIII4959', overwrite=overwrite)
		resultOIII5007 = self.make_emline_map(data_norm, kinematics_wvl, 'OIII5007', overwrite=overwrite)

		return

def runredux(galaxyname, folder='/raid/madlr/voids/analysis/stackedcubes/', makeplots=False):
	""" Run full redux pipeline.

	Arguments:
		verbose (bool): if 'True', make diagnostic plots
		overwrite (bool): if 'True', overwrite existing data files
		binned (bool): if 'True', use binned spectra instead of individual spaxels
		makeplots (bool): if 'True', just run the steps required to make plots
			(note: only works if all steps have been run before!)
	"""

	# Open params
	param = params[galaxyname]

	# Open cube
	c = Cube(galaxyname, folder=folder, verbose=param['verbose'], wcscorr=param['wcscorr'], z=param['z'], EBV=param['EBV'])

	if (not makeplots) and (param['plotcovar']):
		# Get covariance estimate
		covparams = kcwialign.estimatecovar(galaxyname, folder=folder, plot=True, maskfile=folder+galaxyname+'_mcubes.fits')
	else:
		covparams = param['covparams']

	print(covparams, param['targetsn'])

	# Bin spaxels by continuum S/N, accounting for covariance
	c.binspaxels(targetsn=param['targetsn'], params=covparams, emline=None, verbose=param['verbose'])

	if not makeplots:
		# Do continuum fitting to get stellar kinematics
		c.stellarkinematics(overwrite=True, plottest=True, removekinematics=False, snr_mask=param['snr_mask'], verbose=param['verbose'], vsigma=True)
	else:
		c.stellarkinematics(overwrite=False, plottest=True, removekinematics=False, snr_mask=param['snr_mask'], verbose=param['verbose'], vsigma=True, plotveldist=True)

	# Make kinematics plots
	c.plotkinematics(vellimit=param['vellimit'], veldisplimit=param['veldisplimit'], ploterrs=False)

	# TODO: Re-bin, this time using emission line S/N
	#c.binspaxels(verbose=False, targetsn=10, params=covparams, emline='Hbeta')

	# TODO: Correct for reddening
	#c.reddening(verbose=True, overwrite=True, binned=True)

	# TODO: Compute metallicity
	#c.metallicity_Te(overwrite=True)

	return

def runallgalaxies():
	# List of all galaxies
	galaxylist = ['reines65','1180506','281238','1142116','1876887','1904061','2502521','821857',
			'1126100','1158932','1782069','1785212','866934','825059','1063413','1074435',
			'1228631','1246626','955106','1280160','control757','control801','control872',
			'control842','PiscesA','PiscesB','control751','control775','control658']

	# Run reduction pipeline for each galaxy
	for galaxy in galaxylist:
		try:
			runredux(galaxy, folder='/Users/miadelosreyes/Documents/Research/VoidDwarfs/redux/stackedcubes/', makeplots=True)
		except:
			print('Failed on '+galaxy)

	return

def main():

	#runallgalaxies()

	runredux('1782069', folder='/Users/miadelosreyes/Documents/Research/VoidDwarfs/redux/stackedcubes/', makeplots=True)

	return

if __name__ == "__main__":
	main()

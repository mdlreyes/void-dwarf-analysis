# kcwiredux.py
# Script to analyze KCWI data cubes
#
# Created: 6 July 2020
######################################

#Backend for python3 on mahler
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import os

# Packages for binning
import vorbin
from vorbin.voronoi_2d_binning import voronoi_2d_binning

# Packages for stellar kinematic fitting
import ppxf as ppxf_package
from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
from os import path
import glob
from scipy import ndimage

# Packages for emission line fitting
from astropy.modeling import models, fitting
from k_lambda import k_lambda

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

	def __init__(self, filename, folder='/Users/miadelosreyes/Documents/Research/VoidDwarfs/data/', verbose=False, wcscorr=None, z=0., sn_wvl=[4250.,4340.], wvlrange=[3700., 5100.]):

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
		"""

		print('Initializing cube...')

		# Define galaxy name
		self.galaxyname = filename

		# Make output folders
		if not os.path.exists('output/'+self.galaxyname):
			os.makedirs('output/'+self.galaxyname)

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

		# Define wavelength range
		wvlsection = np.where((self.wvl_zcorr > sn_wvl[0]) & (self.wvl_zcorr < sn_wvl[1]))[0]
		goodwvl = np.where((self.wvl_zcorr > wvlrange[0]) & (self.wvl_zcorr < wvlrange[1]))[0]
		self.wvl_cropped = self.wvl_zcorr[goodwvl]
		self.data_cropped = self.data[goodwvl, :, :]
		self.mask_cropped = self.mask[goodwvl, :, :]

	def testcovar(self, threshold=60, savedata=False, verbose=False):
		""" Code to run covariance curve code and set attribute alpha.

			Args:
				threshold (int): Threshold bin size
				savedata (bool): If 'True', save bin sizes and noise ratios into text files
				verbose (bool): If 'True', make diagnostic plots
		"""

		print('Doing covariance test...')

		# Open important packages
		from cwitools import reduction, modeling
		import pandas as pd
		from scipy.optimize import curve_fit

		# Run covariance fitting algorithm by Donal & Yuguang
		hdu, param, bin_sizes, noise_ratios = reduction.fit_covar_xy(self.hdu, self.var, self.mask, return_all=True, plot=True) #, xybins=np.array([1,3,5,7,9])) #, mask_neb=self.z)

		# Print output plots/results
		if verbose:
			plt.tight_layout()
			plt.show()

		print('Initial alpha: %s \nInitial threshold: %s' % (param[0], param[2]))
		self.alpha = param[0]

		# If needed, save the output data
		if savedata:
			np.savetxt('output/'+self.galaxyname+'/binsizes.out', bin_sizes)
			np.savetxt('output/'+self.galaxyname+'/noiseratios.out', noise_ratios)

		# Do separate fit if want to set a specific threshold value
		if threshold > 60:

			# Define functional form from Husemann et al. (2013)
			def beta(N, alpha):
				res = 1. + alpha*np.log(N)
				res[N > threshold] = 1. + alpha*np.log(threshold)
				return res

			# Do fitting
			popt, pcov = curve_fit(beta, bin_sizes, noise_ratios)
			print('alpha = ', popt[0])
			self.alpha = popt[0]

			# Make plot
			if verbose:
				plt.plot(bin_sizes, noise_ratios, 'ko', alpha=0.2)
				xplot = np.linspace(0,360,100)
				plt.plot(xplot, beta(xplot,popt[0]), 'r-')
				plt.plot(xplot, 1. + 1.*np.log(xplot), 'b-')
				plt.axvline(100, color='b', linestyle='--')
				plt.xlabel('Bin size', fontsize=16)
				plt.ylabel(r'$n_{\mathrm{measured}}/n_{\mathrm{no covar}}$', fontsize=16)
				#plt.savefig('figures/covtest2_fluxed.png', bbox_inches='tight')
				plt.show()

		return

	def binspaxels(self, alpha=1., verbose=False, targetsn=10.):
		""" Bin spaxels spatially to increase S/N

			Args:
				alpha (:obj: float): alpha value to be used if testcovar function is not run
				verbose (bool): if 'True', make test plots
				targetsn (float): target value of S/N
		"""

		print('Binning cube...')

		# Define sizes of array
		xsize = np.shape(self.data[0,:,:])[1]
		ysize = np.shape(self.data[0,:,:])[0]

		# Compute signal/noise
		signal = np.ma.mean(self.data[wvlsection,:,:], axis=0)
		
		# Compute noise as detrended standard deviation
		#noise = np.sqrt(np.ma.mean(self.var[wvlsection,:,:], axis=0))
		noise = np.zeros(np.shape(signal))
		for i in range(ysize):
			for j in range(xsize):
				linfit = np.polyfit(self.wvl_zcorr[wvlsection],self.data[wvlsection,i,j],deg=1)
				poly = np.poly1d(linfit)
				noise[i,j] = np.std(self.data[wvlsection,i,j] - np.asarray(poly(self.wvl_zcorr[wvlsection]))**2.)

		# Define S/N
		sntest = signal/noise
		if verbose:
			plt.imshow(sntest,vmin=0)
			plt.colorbar()
			plt.show()

		# Get zeropoints and deltas for coordinates
		ra0 = self.header['CRVAL1']
		dec0 = self.header['CRVAL2']
		rad = self.header['CD1_1'] # RA degrees per col
		decd = self.header['CD2_2'] # Dec degrees per row

		# Prep data for binning by making lists that vorbin can read
		x = []
		y = []
		s = []
		n = []
		for i in range(ysize):
			for j in range(xsize):
				if sntest[i,j] > 1.:

					# Convert from RA/Dec to image coords
					x.append(ra0 - i*rad)
					y.append(dec0 + j*decd)

					# Also append signal and noise to list
					s.append(signal[i,j])
					n.append(noise[i,j])

		self.x = np.asarray(x)
		self.y = np.asarray(y)
		s = np.asarray(s)
		n = np.asarray(n)

		# Default to using class attribute alpha unless it doesn't exist, then use function argument
		try:
			a = self.alpha
		except AttributeError:
			a = alpha
		print('Using alpha: ', a)

		# Define S/N function
		def snfunc(index, signal, noise):
			sn = np.sum(signal[index])/np.sqrt(np.sum(noise[index]**2))
			sn /= 1 + alpha*np.log(index.size)
			return sn

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
		self.stacked_spec = np.zeros((len(self.bins), len(goodwvl)))
		self.stacked_errs = np.zeros((len(self.bins), len(goodwvl)))
		if verbose:
			stacked_data = np.zeros(np.shape(self.data[goodwvl,:,:])) # For plotting binned image

		# Loop over all bins
		for binID in range(len(self.bins)):

			# Get all IDs in that bin
			idx = np.where(self.binNum==self.bins[binID])[0]

			# Get RA/Dec of all the pixels in a bin
			xarray = np.asarray(-(self.x[idx]-ra0)/rad)
			yarray = np.asarray((self.y[idx]-dec0)/decd)

			# Loop over all pixels in the bin and append the spectrum and errors from each pixel to lists
			binned_spec = []
			binned_err = []
			for i in range(len(xarray)):
				binned_spec.append(self.data[goodwvl,np.int(round(xarray[i])),np.int(round(yarray[i]))])
				binned_err.append(self.var[goodwvl,np.int(round(xarray[i])),np.int(round(yarray[i]))])

			# Loop again over all pixels in the bin and put the new mean spectra in the list
			if verbose:
				if self.sn[binID] > 1.:
					for i in range(len(xarray)):
						stacked_data[:,np.int(round(xarray[i])),np.int(round(yarray[i]))] = np.ma.mean(binned_spec)

			self.stacked_spec[binID] = np.ma.mean(binned_spec, axis=0)
			self.stacked_errs[binID] = np.ma.mean(binned_err, axis=0)

		# Plot test figures
		if verbose:

			fig = plt.figure(figsize=(12,6))
			ax = fig.add_subplot(121,projection=self.wcs,slices=('x', 'y', 50))
			ax.set_title('Original white-light image')
			im=ax.imshow(np.ma.sum(self.data[wvlsection,:,:], axis=0), vmin=0)
			fig.colorbar(im, ax=ax)

			ax = fig.add_subplot(122,projection=self.wcs,slices=('x', 'y', 50))
			ax.set_title('Binned white-light image')
			im=ax.imshow(np.ma.sum(stacked_data[wvlsection,:,:], axis=0), vmin=0)
			fig.colorbar(im, ax=ax)

			plt.tight_layout()
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

		# Define spectrum
		spectrum = self.stacked_spec[0] if spectrum is None else spectrum
		wvl = self.wvl_cropped if wvl is None else wvl

		# Define wavelength range
		self.lamRange1 = [wvl[0],wvl[-1]]
		fwhm_gal = 2.4/(1+self.z)  # KCWI instrumental FWHM of ~2.4A

		# Rebin spectrum into log scale to get initial velocity scale
		galaxy, logLam1, velscale = util.log_rebin(self.lamRange1, spectrum)

		# Read the list of filenames from the E-MILES SSP library
		vazdekis = glob.glob(ppxf_dir + '/miles_models/Mun1.30*.fits')
		fwhm_tem = 2.51  # Vazdekis+10 spectra have a constant resolution FWHM of 2.51A.

		# Open template spectrum in order to make get the size of the template array
		hdu = fits.open(vazdekis[0])
		ssp = hdu[0].data
		h2 = hdu[0].header
		self.lamRange2 = h2['CRVAL1'] + np.array([0., h2['CDELT1']*(h2['NAXIS1'] - 1)])
		sspNew, logLam2, velscale_temp = util.log_rebin(self.lamRange2, ssp, velscale=velscale)
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
			ssp = hdu[0].data
			sspNew, self.logLam2, velscale_temp = util.log_rebin(self.lamRange2, ssp, velscale=velscale)
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
			plt.show()

			print("Formal errors:")
			print("     dV    dsigma   dh3      dh4")
			print("".join("%8.2g" % f for f in pp.error*np.sqrt(pp.chi2)))

			print('Best-fitting redshift z:', (self.z + 1)*(1 + pp.sol[0]/c) - 1)

		return np.asarray([pp.sol[0], pp.sol[1], pp.error[0]*np.sqrt(pp.chi2), pp.error[1]*np.sqrt(pp.chi2)]), np.exp(logLam1), pp.bestfit

	def stellarkinematics(self, verbose=False, removekinematics=True, overwrite=False, snr_mask=1):
		""" Do stellar kinematics fitting with pPXF. Note: must run prepstellarfit() first!

			Arguments:
				verbose (bool): if 'True', make diagnostic plots
				removekinematics (bool): if 'True' (default), output a data_norm cube with best-fit stellar template subtracted
				overwrite (bool): if 'True', overwrite existing files
				snr_mask (float): produce velmask.out file marking any bins where S/N > snr_mask

		"""

		# If output files don't exist, run kinematics fitting
		if overwrite or os.path.exists('output/'+self.galaxyname+'/velocity.out')==False or os.path.exists('output/'+self.galaxyname+'/kinfit.npy')==False:

			# Prep the stellar fit
			self.prepstellarfit()

			# Create data structure to hold output
			self.vel = np.zeros(np.shape(self.data[0,:,:]))
			self.veldisp = np.zeros(np.shape(self.data[0,:,:]))
			self.vel_err = np.zeros(np.shape(self.data[0,:,:]))
			self.veldisp_err = np.zeros(np.shape(self.data[0,:,:]))
			self.velmask = np.zeros(np.shape(self.data[0,:,:]), dtype=bool)

			self.kinematics_fit = np.zeros(np.shape(self.data_cropped))
			self.kinematics_wvl = np.zeros(np.shape(self.data_cropped))

			# Get zeropoints and deltas for coordinates
			ra0 = self.header['CRVAL1']
			dec0 = self.header['CRVAL2']
			rad = self.header['CD1_1'] # RA degrees per col
			decd = self.header['CD2_2'] # Dec degrees per row

			print('Doing stellar kinematics fit...')

			# Loop over all bins
			for binID in range(len(self.bins)):

				print('Fitting bin %s/%s' % (binID, len(self.bins)))

				# Do stellar kinematic fit
				params, fitwvl, fit = self.ppxf_fit(self.stacked_spec[binID], self.stacked_errs[binID], verbose=False)

				# Get all IDs in that bin
				idx = np.where(self.binNum==self.bins[binID])[0]

				# Get RA/Dec of all the pixels in a bin
				xarray = np.asarray(-(self.x[idx]-ra0)/rad)
				yarray = np.asarray((self.y[idx]-dec0)/decd)

				# Loop over all pixels in the bin and append the spectrum and errors from each pixel to lists
				for i in range(len(xarray)):
					self.vel[np.int(round(xarray[i])),np.int(round(yarray[i]))] = params[0]
					self.veldisp[np.int(round(xarray[i])),np.int(round(yarray[i]))] = params[1]
					self.vel_err[np.int(round(xarray[i])),np.int(round(yarray[i]))] = params[2]
					self.veldisp_err[np.int(round(xarray[i])),np.int(round(yarray[i]))] = params[3]
					if self.sn[binID] > snr_mask:
						self.velmask[np.int(round(xarray[i])),np.int(round(yarray[i]))] = 1

					# Add the best-fit solution to arrays
					self.kinematics_fit[:,np.int(round(xarray[i])),np.int(round(yarray[i]))] = fit
					self.kinematics_wvl[:,np.int(round(xarray[i])),np.int(round(yarray[i]))] = fitwvl

			np.savetxt('output/'+self.galaxyname+'/velocity.out', self.vel)
			np.savetxt('output/'+self.galaxyname+'/veldisp.out', self.veldisp)
			np.savetxt('output/'+self.galaxyname+'/vel_err.out', self.vel_err)
			np.savetxt('output/'+self.galaxyname+'/veldisp_err.out', self.veldisp_err)
			np.savetxt('output/'+self.galaxyname+'/velmask.out', self.velmask)

			np.save('output/'+self.galaxyname+'/kinfit', self.kinematics_fit)
			np.save('output/'+self.galaxyname+'/kinwvl', self.kinematics_wvl)

			if verbose:
				fig = plt.figure(figsize=(8,8))
				ax = plt.subplot(projection=self.wcs,slices=('x', 'y', 50))
				plt.imshow(self.vel)
				plt.colorbar()
				plt.show()

				fig = plt.figure(figsize=(8,8))
				ax = plt.subplot(projection=self.wcs,slices=('x', 'y', 50))
				plt.imshow(self.velmask)
				plt.colorbar()
				plt.show()

		# Else, just open existing kinematics fit files
		else:
			self.prepstellarfit()
			self.kinematics_fit = np.load('output/'+self.galaxyname+'/kinfit.npy')
			self.kinematics_wvl = np.load('output/'+self.galaxyname+'/kinwvl.npy')

		# Remove best-fit stellar template from each spaxel
		if removekinematics:

			print('Normalizing data by best-fit stellar template...')

			# Define sizes of array
			xsize = np.shape(self.data[0,:,:])[1]
			ysize = np.shape(self.data[0,:,:])[0]

			self.data_norm = np.zeros(np.shape(self.data_cropped))

			for i in range(xsize):
				for j in range(ysize):

					# Smooth observed spectrum to match template
					spectrum = self.data_cropped[:,i,j]
					galspec = ndimage.gaussian_filter1d(spectrum, self.sigma)

					# Save log-rebinned spectrum
					galaxy, logLam1, velscale = util.log_rebin(self.lamRange1, galspec)
					self.data_norm[:,i,j] = galaxy

			self.data_norm = self.data_norm - self.kinematics_fit*np.median(self.data_norm, axis=0)

			np.save('output/'+self.galaxyname+'/data_norm', self.data_norm)

			# Plot image for testing
			if verbose:

				# Plot example spectrum
				plt.figure(figsize=(12,5))
				idx = 46
				idy = 40
				#plt.plot(self.kinematics_wvl[:,idx,idy],self.data_norm[:,idx,idy])
				#plt.plot(self.kinematics_wvl[:,idx,idy],(self.kinematics_fit*np.ma.median(self.data_norm, axis=0))[:,idx,idy])
				plt.plot(self.kinematics_wvl[:,idx,idy], self.data_norm[:,idx,idy])

				plt.xlabel(r'$\lambda (\AA)$', fontsize=16)
				plt.ylabel('Flux', fontsize=16)
				plt.xlim(3500,5100)

				# Plot error
				plt.show()

		return

	def plotkinematics(self, vel='velocity.out', veldisp='veldisp.out', vel_err='vel_err.out', veldisp_err='veldisp_err.out', velmask='velmask.out', instdisp=False):
		""" Make kinematic plots.

			Arguments:
				vel, veldisp, vel_err, veldisp_err (2D arrays): kinematics measurements
				(If set to 'None', will use output directly from stellarkinematics())
				velmask (2D bool array): mask marking bins where total S/N > some value (1 = good, 0 = bad)
				instdisp (bool): if 'True', subtract (in quadrature) instrument dispersion from vel dispersion
		"""

		print('Plotting stellar kinematic values...')

		# Define spectrum
		vel = np.loadtxt('output/'+self.galaxyname+'/'+vel) if vel=='velocity.out' else self.vel
		veldisp = np.loadtxt('output/'+self.galaxyname+'/'+veldisp) if veldisp=='veldisp.out' else self.veldisp
		vel_err = np.loadtxt('output/'+self.galaxyname+'/'+vel_err) if vel_err=='vel_err.out' else self.vel_err
		veldisp_err = np.loadtxt('output/'+self.galaxyname+'/'+veldisp_err) if veldisp_err=='veldisp_err.out' else self.veldisp_err

		if velmask=='velmask.out': 
			velmask = np.loadtxt('output/'+self.galaxyname+'/'+velmask)
		else:
			velmask = self.velmask

		# Account for difference in the instrumental dispersion of the template and the data
		if instdisp:
			c = 299792.458
			instdisp = 0.40/4400. * c # 0.40 A ~ quadratic difference between template and data; 4400 A ~ mean wavelength of spectrum
			print(instdisp)
			veldisp = np.sqrt(np.power(veldisp,2.) - instdisp**2.)

		def plot(array, lowerlim=None, upperlim=None, error=None, nan=False, sn=None, velshift=None, mask=None, cmap='viridis', title=None): 

			# Copy array
			copy = np.copy(array)
			copy[copy==0] = np.nan

			# Do S/N cut on data
			if error is not None and sn is not None:
				idx = np.where(np.abs(copy/error) < sn)
				copy[idx] = np.nan

			if velshift is not None:
				copy += velshift

			if mask is not None:
				mask = np.array(mask, dtype=bool)
				copy[~mask] = np.nan

			if nan:
				if lowerlim is not None:
					copy[copy < lowerlim] = np.nan
				if upperlim is not None:
					copy[copy > upperlim] = np.nan

			#fig = plt.figure(figsize=(12,6))
			#ax = fig.add_subplot(121,projection=self.wcs,slices=('x', 'y', 50))
			#ax.set_title('Original white-light image')
			#im=ax.imshow(np.ma.mean(self.data, axis=0), vmin=0)
			#fig.colorbar(im, ax=ax)

			#ax = fig.add_subplot(122,projection=self.wcs,slices=('x', 'y', 50))
			fig = plt.figure(figsize=(8,8))
			ax = plt.subplot(projection=self.wcs,slices=('x', 'y', 50))
			ax.set_title(title)
			if nan:
				im = ax.imshow(copy, cmap=cmap, title=title)
			else:
				im = ax.imshow(copy, vmin=lowerlim, vmax=upperlim, cmap=cmap)
			fig.colorbar(im, ax=ax)

			plt.savefig('figures/test.png', bbox_inches='tight') 
			plt.show()

			print(np.nanmean(copy))

			return copy

		plot(vel, error=vel_err, nan=False, upperlim=100, lowerlim=-100, velshift=-160, cmap='RdBu', title='Velocity (km/s)', mask=velmask)
		plot(veldisp, error=veldisp_err, nan=False, upperlim=300, title=r'$\sigma$ (km/s)', mask=velmask)
		#plot(vel_err, nan=False, cmap='RdBu', title='Velocity error (km/s)') #mask=velmask, 
		#plot(veldisp_err, nan=False, upperlim=200, title=r'$\sigma$ error (km/s)') #mask=velmask, 

		return

	def make_emline_map(self, datanorm, wvlnorm, line_name, velmask='velmask.out', xlim=20., overwrite=False):
		""" Fits gas emission lines and makes emission line maps.

			Arguments:
				datanorm, wvlnorm (3D arrays): data and wavelength arrays to be fit
				line_name (string): name of line to fit
				velmask (2D bool array): mask marking bins where total S/N > 8 (1 = good, 0 = bad)
				xlim (float): in Angstroms, half of wavelength range about line center to fit
				overwrite (bool): if 'True', overwrite any existing files

			Returns:
				lineflux, width (2D arrays): output line flux and width maps
		"""

		print('Making emission line map of '+line_name+'...')

		# Check if data already exists
		flux_file = 'output/'+self.galaxyname+'/'+line_name+'_flux.out'
		width_file = 'output/'+self.galaxyname+'/'+line_name+'_width.out'
		snr_file = 'output/'+self.galaxyname+'/'+line_name+'_snr.out'
		if overwrite==False and os.path.exists(flux_file) and os.path.exists(width_file) and os.path.exists(snr_file):
			return np.loadtxt(flux_file), np.loadtxt(width_file), np.loadtxt(snr_file)

		# Define where to measure emission lines
		if velmask=='velmask.out':
			velmask = np.loadtxt('output/'+self.galaxyname+'/'+velmask)
		else:
			velmask = self.velmask

		# Get central line wavelength
		line = wvldict[line_name]

		# Mask out any bad pixels
		datanorm = np.ma.array(datanorm, mask=self.mask_cropped)
		wvlnorm = np.ma.array(wvlnorm, mask=self.mask_cropped)

		# Prep arrays to hold outputs
		lineflux = np.zeros(np.shape(datanorm[0,:,:]))
		width = np.zeros(np.shape(datanorm[0,:,:]))
		snr = np.zeros(np.shape(datanorm[0,:,:]))

		# Loop over all spaxels
		for i in range(len(datanorm[0,:,0])):
			for j in range(len(datanorm[0,0,:])):
				if velmask[i,j]:

					# Crop flux, wvl arrays to only contain the area around the line
					idx = np.where((wvlnorm[:,i,j] > (line-xlim)) & (wvlnorm[:,i,j] < (line+xlim)))[0]
					wvl = wvlnorm[:,i,j][idx]
					flux = datanorm[:,i,j][idx]

					# Fit line with Gaussian + linear background
					gaussian_model = models.Gaussian1D(np.max(flux), line, 2) + models.Linear1D(0,0)
					fitter = fitting.LevMarLSQFitter()
					gaussian_fit = fitter(gaussian_model, wvl, flux)

					# Compute integral
					integral = np.sqrt(2.*np.pi)*gaussian_fit[0].amplitude*gaussian_fit[0].stddev

					if integral > 0. and gaussian_fit[0].stddev.value < xlim/2. and np.abs(gaussian_fit[0].mean - line) < xlim/2.:

						lineflux[i,j] = integral
						width[i,j] = gaussian_fit[0].stddev.value

						# Compute SNR
						emidx = np.where((wvlnorm[:,i,j] > (gaussian_fit[0].mean-2.5*gaussian_fit[0].stddev)) & (wvlnorm[:,i,j] < (gaussian_fit[0].mean+2.5*gaussian_fit[0].stddev)))[0]
						emflux = datanorm[:,i,j][emidx]

						cont1idx = np.where((wvlnorm[:,i,j] < (gaussian_fit[0].mean-5*gaussian_fit[0].stddev)))[0]
						cont2idx = np.where((wvlnorm[:,i,j] > (gaussian_fit[0].mean+5*gaussian_fit[0].stddev)))[0]
						contflux1 = datanorm[:,i,j][cont1idx]
						contflux2 = datanorm[:,i,j][cont2idx]

						signal = np.sum(emflux - np.mean(np.hstack((contflux1,contflux2)))) / np.sqrt(len(emflux))
						noisecont = (np.std(contflux1) + np.std(contflux2)) / 2. # Continuum noise
						pois = np.random.poisson(size=len(emflux))
						noisepois = np.std(pois/np.sum(pois)*np.sqrt(emflux)) # Poisson noise
						noise = np.sqrt(noisecont**2. + noisepois**2.)

						if signal/noise > 0.:
							snr[i,j] = signal/noise

		# Save data
		np.savetxt(flux_file, lineflux)
		np.savetxt(width_file, width)
		np.savetxt(snr_file, snr)

		return lineflux, width, snr

	def reddening(self, kinematics=True, verbose=False, overwrite=False):
		""" Compute and apply reddening correction to each spaxel.

			Arguments:
				kinematics (bool): if 'True', remove best-fit stellar template (must run stellarkinematics first)
				verbose (bool): if 'True', make diagnostic plots
		"""

		print('Computing Balmer decrement...')

		# Remove best-fit stellar template
		if kinematics:

			# Check if data already exists
			try:
				data_norm = self.data_norm
				kinematics_wvl = self.kinematics_wvl
			except AttributeError:
				data_norm = np.load('output/'+self.galaxyname+'/'+'data_norm.npy')
				kinematics_wvl = np.load('output/'+self.galaxyname+'/'+'kinwvl.npy')

		else:
			data_norm = self.data_cropped
			kinematics_wvl = np.broadcast_to(self.wvl_cropped,self.data_cropped.T.shape).T

		# Get Balmer line maps
		resultHbeta = self.make_emline_map(data_norm, kinematics_wvl, 'Hbeta', overwrite=overwrite)
		resultHgamma = self.make_emline_map(data_norm, kinematics_wvl, 'Hgamma', overwrite=overwrite)

		# Compute relevant quantities
		Hbeta = np.copy(resultHbeta[0])
		Hgamma = np.copy(resultHgamma[0])
		balmer = Hgamma/Hbeta

		if verbose:

			fig = plt.figure(figsize=(8,8))
			ax = plt.subplot(projection=self.wcs,slices=('x', 'y', 50))
			plt.imshow(resultHbeta[2], vmin=1)
			plt.colorbar()
			plt.savefig('figures/HbetaSNRtest.png', bbox_inches='tight')
			plt.show()

			fig = plt.figure(figsize=(15,6))

			ax = fig.add_subplot(131,projection=self.wcs,slices=('x', 'y', 50))
			ax.set_title(r'H$\beta$')
			im=ax.imshow(Hbeta, vmin=0, vmax=1)
			fig.colorbar(im, ax=ax)

			ax = fig.add_subplot(132,projection=self.wcs,slices=('x', 'y', 50))
			ax.set_title(r'H$\gamma$')
			im=ax.imshow(Hgamma, vmin=0, vmax=1)
			fig.colorbar(im, ax=ax)

			#balmer[np.isnan(balmer)] = 0.
			ax = fig.add_subplot(133,projection=self.wcs,slices=('x', 'y', 50))
			ax.set_title(r'$\mathrm{H}\gamma/\mathrm{H}\beta$')
			im=ax.imshow(balmer, vmax=0.8)
			fig.colorbar(im, ax=ax)

			plt.show()

		print('Computing E(B-V)...')

		# Intrinsic Hgamma/Hbeta ratio (assuming Case B recombination, T=10^4K, electron density 100/cm^3)
		balmer0 = 0.468

		# Compute E(B-V)
		Ebv = np.log10(balmer/balmer0)/(-0.4*(k_lambda(wvldict['Hgamma'])-k_lambda(wvldict['Hbeta'])))

		if verbose:
			plt.figure(figsize=(8,8))
			plt.subplot(projection=self.wcs,slices=('x', 'y', 50))
			plt.imshow(Ebv, cmap='viridis', interpolation='nearest', vmin=0, vmax=5)
			plt.colorbar(label=r'$E(B-V)$')
			plt.savefig('figures/EBVtest.png', bbox_inches='tight')
			plt.show()

		return

def main():

	c = Cube('reines65', folder='/Users/miadelosreyes/Documents/Research/VoidDwarfs/data/', verbose=False, wcscorr=[174.17801 - 174.1787083, 26.727126 - 26.7263583], z=0.0331)
	#c.testcovar(threshold=100, verbose=True)
	#c.binspaxels(verbose=True, targetsn=5., alpha=2.8)
	#c.stellarkinematics(verbose=True, overwrite=True, snr_mask=1)
	#c.plotkinematics(instdisp=True)
	c.reddening(verbose=False, overwrite=False)

	return

if __name__ == "__main__":
	main()
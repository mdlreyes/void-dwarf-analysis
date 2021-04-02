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
from params import params

# Packages for binning
import kcwialign
import vorbin
from vorbin.voronoi_2d_binning import voronoi_2d_binning

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

		# Open mask cube
		with fits.open(folder+filename+'_mcubes.fits') as mcube:
			self.mask = mcube[0].data
		#self.mask = np.zeros_like(data)
		#badidx = np.where((np.isclose(data,0)) & (np.isclose(var,0)))
		#self.mask[badidx] = True
		#badidx = np.where((data < 0.) | (~np.isfinite(var)))
		#self.mask[badidx] = True

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
		self.wvlsection = np.where((self.wvl_zcorr > sn_wvl[0]) & (self.wvl_zcorr < sn_wvl[1]))[0] # Wavelength range for S/N fitting
		self.goodwvl = np.where((self.wvl_zcorr > wvlrange[0]) & (self.wvl_zcorr < wvlrange[1]))[0] # Wavelength range for stellar template fitting
		self.wvl_cropped = self.wvl_zcorr[self.goodwvl]
		self.data_cropped = self.data[self.goodwvl, :, :]
		self.mask_cropped = self.mask[self.goodwvl, :, :]

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
					noise[i,j] = np.std(self.data[self.wvlsection,i,j] - np.asarray(poly(self.wvl_zcorr[self.wvlsection]))**2.)

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
		x = []
		y = []
		s = []
		n = []
		for i in range(ysize):
			for j in range(xsize):
				if sntest[i,j] > 1.:

					# Convert from RA/Dec to image coords
					x.append(self.ra0 - i*self.rad)
					y.append(self.dec0 + j*self.decd)

					# Also append signal and noise to list
					s.append(signal[i,j])
					n.append(noise[i,j])

		self.x = np.asarray(x)
		self.y = np.asarray(y)
		s = np.asarray(s)
		n = np.asarray(n)

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

			# Get RA/Dec of all the pixels in a bin
			xarray = np.asarray(-(self.x[idx]-self.ra0)/self.rad)
			yarray = np.asarray((self.y[idx]-self.dec0)/self.decd)

			# Loop over all pixels in the bin and append the spectrum and errors from each pixel to lists
			binned_spec = []
			binned_err = []
			for i in range(len(xarray)):
				binned_spec.append(self.data[self.goodwvl,np.int(round(xarray[i])),np.int(round(yarray[i]))])
				binned_err.append(self.var[self.goodwvl,np.int(round(xarray[i])),np.int(round(yarray[i]))])

				# Also record the bin ID for each pixel
				self.binIDarray[np.int(round(xarray[i])),np.int(round(yarray[i]))] = binID

			# Loop again over all pixels in the bin and put the new mean spectra in the list
			if verbose:
				if self.sn[binID] > 1.:
					for i in range(len(xarray)):
						stacked_data[:,np.int(round(xarray[i])),np.int(round(yarray[i]))] = np.ma.mean(binned_spec)

			# Compute covariance correction
			if len(xarray) >= self.threshold:
				correction = self.norm * (1 + self.alpha * np.log(self.threshold))
			else:
				correction = self.norm * (1 + self.alpha * np.log(len(xarray)))

			self.stacked_spec[binID] = np.ma.mean(binned_spec, axis=0)
			self.stacked_errs[binID] = np.ma.mean(binned_err, axis=0) * correction**2. / len(xarray)

		# For testing purposes, save arrays of bin IDs and bin errors
		np.save('output/'+self.galaxyname+'/binIDarray', self.binIDarray)
		np.save('output/'+self.galaxyname+'/binerrs', self.stacked_errs)

		# Find luminosity-weighted center
		xcenter = np.sum(self.x * s)/np.sum(s)
		ycenter = np.sum(self.y * s)/np.sum(s)
		print('center:', xcenter, ycenter)

		# Find bin where center is located
		self.centeridx = (np.sqrt((xNode-xcenter)**2. + (yNode-ycenter)**2.)).argmin()

		# Plot test figures
		if verbose:

			fig = plt.figure(figsize=(12,6))
			ax = fig.add_subplot(121,projection=self.wcs,slices=('x', 'y', 50))
			ax.set_title('Original white-light image')
			im=ax.imshow(np.ma.sum(self.data[self.wvlsection,:,:], axis=0), vmin=0)
			fig.colorbar(im, ax=ax)

			ax = fig.add_subplot(122,projection=self.wcs,slices=('x', 'y', 50))
			ax.set_title('Binned white-light image')
			im=ax.imshow(np.ma.sum(stacked_data[self.wvlsection,:,:], axis=0), vmin=0)
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
			print("     dV    dsigma")
			print("".join("%8.2g" % f for f in pp.error*np.sqrt(pp.chi2)))

			print('Best-fitting redshift z:', (self.z + 1)*(1 + pp.sol[0]/c) - 1)

		return np.asarray([pp.sol[0], pp.sol[1], pp.error[0]*np.sqrt(pp.chi2), pp.error[1]*np.sqrt(pp.chi2), pp.chi2]), np.exp(logLam1), pp.bestfit, galaxy, noise

	def stellarkinematics(self, verbose=False, removekinematics=True, overwrite=False, snr_mask=1, plottest=False, vsigma=True):
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
			self.kinematics_fit_bin = np.zeros(np.shape(self.stacked_spec))
			self.kinematics_wvl_bin = np.zeros(np.shape(self.stacked_spec))

			print('Doing stellar kinematics fit...')

			# If computing global v/sigma, prep the outputs
			if vsigma:
				vtotalsq = 0.
				verrtotal = 0.
				sigtotalsq = 0.
				sigerrtotal = 0.
				weightedvelerr = 0.
				weightedsigerr = 0.
				totalflux = 0.

			# Loop over all bins
			for binID in tqdm(range(len(self.bins))):

				# Do stellar kinematic fit
				params, fitwvl, fit, obs, obserr = self.ppxf_fit(self.stacked_spec[binID], self.stacked_errs[binID], verbose=False)

				if binID == self.centeridx:
					systvel = params[0]
					print('Systematic velocity:'+"\t".join("%.2f" % f for f in [params[0],params[2]]))

				# Put fit for each bin into an array
				self.kinematics_fit_bin[binID] = fit
				self.kinematics_wvl_bin[binID] = fitwvl

				# Get all IDs in that bin
				idx = np.where(self.binNum==self.bins[binID])[0]

				# Get RA/Dec of all the pixels in a bin
				xarray = np.asarray(-(self.x[idx]-self.ra0)/self.rad)
				yarray = np.asarray((self.y[idx]-self.dec0)/self.decd)

				if plottest:
					if params[2] > np.abs(params[0]) or params[3] > params[1]:
						print(binID, xarray[0], yarray[0])
						print(params)
						plt.plot(fitwvl, fit, 'r-')
						plt.fill_between(fitwvl, obs-np.sqrt(obserr),obs+np.sqrt(obserr), color='gray', alpha=0.5)
						plt.show()

				flux = 0.  # only gets used if systvel is not None

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

					# If doing v/sigma calculation, compute total continuum flux in bin
					if vsigma:
						
						# First get wavelength range, excluding emission lines
						contwvl = np.ones_like(self.wvl_zcorr, dtype='bool')
						contwvl[self.wvl_zcorr < 3700.] = False
						contwvl[self.wvl_zcorr > 5100.] = False

						for wvl in wvldict.keys():
							wvlrange = np.where((self.wvl_zcorr > (wvldict[wvl]-10.)) & (self.wvl_zcorr > (wvldict[wvl]+10.)))[0]
							contwvl[wvlrange] = False

						# Now add up all data in the correct wvl range
						flux += np.sum(self.data[contwvl,np.int(round(xarray[i])),np.int(round(yarray[i]))])

				# Compute v/sigma and specific angular momentum (Eq 1 from Ferre-Mateu+2021)
				if vsigma:

					vtotalsq += flux * (params[0]-systvel)**2. * 1/(params[2]**2.)
					verrtotal += 1./(params[2]**2.)  # Do weighted average
					sigtotalsq += flux * params[1]**2. * 1/(params[3]**2.)
					sigerrtotal += 1./(params[3]**2.)  # Do weighted average
					weightedvelerr += flux * params[2]**2.
					weightedsigerr += flux * params[3]**2.
					totalflux += flux

			# Do final computation of v/sigma
			if vsigma:
				vtotalsq /= verrtotal
				sigtotalsq /= sigerrtotal
				weightedvelerr /= totalflux
				weightedsigerr /= totalflux
				kinematics = [np.sqrt(vtotalsq), np.sqrt(sigtotalsq), np.sqrt(weightedvelerr), np.sqrt(weightedsigerr)]
				print('Global kinematics:')
				print("\t".join("%.2f" % f for f in kinematics))

			# Correct for systemic velocity
			self.vel += -systvel

			np.savetxt('output/'+self.galaxyname+'/velocity.out', self.vel)
			np.savetxt('output/'+self.galaxyname+'/veldisp.out', self.veldisp)
			np.savetxt('output/'+self.galaxyname+'/vel_err.out', self.vel_err)
			np.savetxt('output/'+self.galaxyname+'/veldisp_err.out', self.veldisp_err)
			np.savetxt('output/'+self.galaxyname+'/velmask.out', self.velmask)

			np.save('output/'+self.galaxyname+'/kinfit', self.kinematics_fit)
			np.save('output/'+self.galaxyname+'/kinwvl', self.kinematics_wvl)
			np.save('output/'+self.galaxyname+'/kinfit_bin', self.kinematics_fit_bin)
			np.save('output/'+self.galaxyname+'/kinwvl_bin', self.kinematics_wvl_bin)

			if verbose:
				fig = plt.figure(figsize=(8,8))
				ax = plt.subplot(projection=self.wcs,slices=('x', 'y', 50))
				plt.title('Velocity (test)')
				plt.imshow(self.vel)
				plt.colorbar()
				plt.show()

				fig = plt.figure(figsize=(8,8))
				ax = plt.subplot(projection=self.wcs,slices=('x', 'y', 50))
				plt.title('Velocity mask')
				plt.imshow(self.velmask)
				plt.colorbar()
				plt.show()

		# Else, just open existing kinematics fit files
		else:
			self.prepstellarfit()
			self.kinematics_fit = np.load('output/'+self.galaxyname+'/kinfit.npy')
			self.kinematics_wvl = np.load('output/'+self.galaxyname+'/kinwvl.npy')
			self.kinematics_fit_bin = np.load('output/'+self.galaxyname+'/kinfit_bin.npy')
			self.kinematics_wvl_bin = np.load('output/'+self.galaxyname+'/kinwvl_bin.npy')

		# Remove best-fit stellar template from each spaxel
		if removekinematics:

			print('Normalizing data by best-fit stellar template for each spaxel...')

			# Define sizes of array
			xsize = np.shape(self.data[0,:,:])[0]
			ysize = np.shape(self.data[0,:,:])[1]

			# Prep array for output
			self.data_norm = np.zeros(np.shape(self.data_cropped))

			for i in range(xsize):
				for j in range(ysize):

					# Smooth observed spectrum to match template
					spectrum = self.data_cropped[:,i,j]
					galspec = ndimage.gaussian_filter1d(spectrum, self.sigma)

					# Save log-rebinned spectrum
					galaxy, logLam1, velscale = util.log_rebin(self.lamRange1, galspec)
					self.data_norm[:,i,j] = galaxy

			# Subtract kinematics fit from data and save file
			self.data_norm = self.data_norm - self.kinematics_fit*np.median(self.data_norm, axis=0)
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
				galaxy, logLam1, velscale = util.log_rebin(self.lamRange1, galspec)

				# Subtract kinematics fit from binned data
				self.data_norm_bin[binID] = galaxy - self.kinematics_fit_bin[binID]*np.median(galaxy)

			# Save file
			np.save('output/'+self.galaxyname+'/data_norm_bin', self.data_norm_bin)

			# Plot image for testing
			if verbose:

				# Plot example spectrum
				plt.figure(figsize=(12,5))
				idx = 60
				idy = 40
				plt.title('single spaxel')
				plt.plot(self.kinematics_wvl[:,idx,idy], self.data_norm[:,idx,idy])
				plt.xlabel(r'$\lambda (\AA)$', fontsize=16)
				plt.ylabel('Flux', fontsize=16)
				plt.xlim(3500,5100)

				# Plot example spectrum
				plt.figure(figsize=(12,5))
				plt.title('single bin')
				idx = 10
				plt.plot(self.kinematics_wvl_bin[idx], self.data_norm_bin[idx])
				plt.xlabel(r'$\lambda (\AA)$', fontsize=16)
				plt.ylabel('Flux', fontsize=16)
				plt.xlim(3500,5100)

				# Plot error
				plt.show()

				# Plot error
				plt.show()

		return

	def plotkinematics(self, vel='velocity.out', veldisp='veldisp.out', vel_err='vel_err.out', 
		veldisp_err='veldisp_err.out', velmask='velmask.out', 
		instdisp=False, vellimit=None, veldisplimit=None, ploterrs=False):
		""" Make kinematic plots.

			Arguments:
				vel, veldisp, vel_err, veldisp_err (2D arrays): kinematics measurements
					(If set to 'None', will use output directly from stellarkinematics())
				velmask (2D bool array): mask marking bins where total S/N > some value (1 = good, 0 = bad)
				instdisp (bool): if 'True', subtract (in quadrature) instrument dispersion from vel dispersion
				vellimit, veldisplimit (float): limits for velocity and velocity dispersion maps
					(velocity map goes from [-vellimit, vellimit], dispersion map goes from [0, veldisplimit])
				ploterrs (bool): if 'True', also plot and save velocity/dispersion error maps
		"""

		print('Plotting stellar kinematics maps...')

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

		def plot(array, error=None, sn=None, limits=None, velshift=None, limit=None, mask=None, cmap='viridis', title=None, plotname=''): 

			# Copy array
			copy = np.copy(array)
			#copy[copy==0] = np.nan

			# Do S/N cut on data
			if error is not None and sn is not None:
				idx = np.where(np.abs(copy/error) < sn)
				copy[idx] = np.nan

			# Mask any bad data
			if mask is not None:
				mask = np.array(mask, dtype=bool)
				copy[~mask] = np.nan

			# Do velocity shift
			'''
			if velshift is not None:
				# If velocity shift was input but not defined (i.e., don't use systemic velocity)
				if np.isclose(velshift, 0.):
					velshift = np.nanmean(copy)
				
				# Round velocity shift to the nearest 10 km/s
				velshift = -round(velshift,-1)
				print('Velocity shift:', velshift)
				copy += velshift
			'''

			fig = plt.figure(figsize=(8,8))
			ax = plt.subplot(projection=self.wcs,slices=('x', 'y', 50))
			ax.text(0.03, 0.95, self.galaxyname, transform=ax.transAxes, fontsize=14)
			plt.grid(color='black', ls='dotted')
			if limits is None:
				im = ax.imshow(copy, cmap=cmap)
			else:
				im = ax.imshow(copy, vmin=limits[0], vmax=limits[1], cmap=cmap)
			fig.colorbar(im, ax=ax, label=title)

			plt.savefig('figures/'+self.galaxyname+'/'+plotname+'.png', bbox_inches='tight') 
			plt.show()

			return copy

		plot(vel, error=vel_err, limits=[-vellimit,vellimit], cmap='coolwarm', title='Velocity (km/s)', mask=velmask, plotname='vel')
		plot(veldisp, error=veldisp_err, limits=[0, veldisplimit], title=r'$\sigma$ (km/s)', mask=velmask, plotname='veldisp')
		if ploterrs:
			plot(vel_err, title='Velocity error (km/s)', plotname='velerr') #mask=velmask, 
			plot(veldisp_err, title=r'$\sigma$ error (km/s)', plotname='veldisperr') #mask=velmask, 

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
			plt.show()

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
				xarray = np.asarray(-(self.x[idx]-self.ra0)/self.rad)
				yarray = np.asarray((self.y[idx]-self.dec0)/self.decd)

				# Loop again over all pixels in the bin and put the correct emline values in the array
				for i in range(len(xarray)):
					Hbeta_unbinned[np.int(round(xarray[i])),np.int(round(yarray[i]))] = resultHbeta[0][binID]
					Hgamma_unbinned[np.int(round(xarray[i])),np.int(round(yarray[i]))] = resultHgamma[0][binID]
					ebv_unbinned[:, np.int(round(xarray[i])),np.int(round(yarray[i]))] = ebv[:, binID]

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
			plt.show()

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

			plt.show()
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
			plt.show()

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
			plt.show()

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
		c.stellarkinematics(overwrite=True, snr_mask=param['snr_mask'], verbose=param['verbose'], vsigma=True)

	# Make kinematics plots
	c.plotkinematics(instdisp=param['instdisp'], vellimit=param['vellimit'], veldisplimit=param['veldisplimit'], ploterrs=True)

	# TODO: Re-bin, this time using emission line S/N
	#c.binspaxels(verbose=False, targetsn=10, params=covparams, emline='Hbeta')

	# TODO: Correct for reddening
	#c.reddening(verbose=True, overwrite=True, binned=True)

	# TODO: Compute metallicity
	#c.metallicity_Te(overwrite=True)

	return

def main():

	#runredux('reines65', folder='/Users/miadelosreyes/Documents/Research/VoidDwarfs/analysis/stackedcubes/')

	c = Cube('reines65', folder='/Users/miadelosreyes/Documents/Research/VoidDwarfs/analysis/stackedcubes/', verbose=False, wcscorr=[174.17801 - 174.1787083, 26.727126 - 26.7263583], z=0.0331, EBV=0.0217)
	c.binspaxels(verbose=False, targetsn=15, params=[0.108,1.65,80], emline=None)
	c.stellarkinematics(verbose=False, overwrite=True, snr_mask=1, plottest=False, vsigma=True)
	c.plotkinematics(instdisp=False, vellimit=100, veldisplimit=150, ploterrs=True)
	#c.reddening(verbose=True, overwrite=True, binned=True)

	return

if __name__ == "__main__":
	main()

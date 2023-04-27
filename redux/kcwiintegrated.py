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
from astropy.wcs.utils import proj_plane_pixel_scales, skycoord_to_pixel
import os
from params import params
from params_integrated import params_integrated

# Astropy packages
from astropy import units as u
from astropy.modeling import models, fitting
from photutils.isophote import EllipseGeometry, Ellipse
from photutils.aperture import EllipticalAperture
from astropy.coordinates import Angle, SkyCoord
from regions import PixCoord, EllipsePixelRegion, RectangleSkyRegion, PointPixelRegion, RegionVisual, CircleSkyRegion
from astropy.cosmology import FlatLambdaCDM  # needed to compute redshifts
cosmo = FlatLambdaCDM(H0=67.4, Om0=0.315)  # using Planck (2018) params
from astropy.io import ascii

# Packages for stellar continuum fitting
import ppxf as ppxf_package
from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
from os import path
import glob
from scipy import ndimage

# Packages for emission line fitting
from utils.k_lambda import k_lambda
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

		# Define galaxy name
		self.folder = folder
		self.galaxyname = filename

		print('Initializing cube '+self.galaxyname+'...')

		# Make output folders
		if not os.path.exists('output/'+self.galaxyname):
			os.makedirs('output/'+self.galaxyname)
		if not os.path.exists('figures/'+self.galaxyname):
			os.makedirs('figures/'+self.galaxyname)

		# Get auxiliary data
		self.voiddata = ascii.read('../data/sample_FINAL_new.csv').filled(-999.0)
		self.galaxyidx = np.where(self.voiddata['ID']==self.galaxyname)

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

		'''
		# Make mask
		self.mask = np.zeros_like(data)
		badidx = np.where((np.isclose(data,0)) & (np.isclose(var,0)))
		self.mask[badidx] = True
		badidx = np.where((data < 0.) | (~np.isfinite(var)))
		self.mask[badidx] = True
		'''

		# Mask the data and variance cubes
		self.data = np.ma.array(data, mask=self.mask)
		self.var = np.ma.array(var, mask=self.mask)

		# Apply WCS shift to correct for pointing errors
		if len(wcscorr)==2:
			self.header['CRVAL1'] += wcscorr[0]
			self.header['CRVAL2'] += wcscorr[1]
		self.wcs = WCS(icube[0].header)

		#print(self.header)

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
					('sersic' for Sersic fitting, 'isophote' for elliptical isophotes, 'dbsp' to mimic long-slit region)

			Outputs:
				Re (float): half-light radius
		"""

		print('Fitting aperture to white-light image...')
		
		# Open params

		if mode=='sersic':
			# Fit with 2D Sersic profile
			y, x = np.mgrid[:self.ysize, :self.xsize]
			p_init = models.Sersic2D(amplitude=param['sersic_amp'], r_eff=param['sersic_reff'], n=param['sersic_n'], x_0=param['sersic_x0'], y_0=param['sersic_y0'], ellip=param['sersic_ellip'], theta=param['sersic_theta'])
			fit_p = fitting.LevMarLSQFitter()
			p = fit_p(p_init, x, y, self.bband)
			#print(p)

			#e = models.Ellipse2D(amplitude=1., x_0=p.x_0.value, y_0=p.y_0.value, a=p.r_eff.value/2., b=p.r_eff.value*(1-p.ellip.value)/2., theta=p.theta.value)
			#print(e.evaluate(x, y))

			reg = EllipsePixelRegion(PixCoord(p.x_0.value, p.y_0.value), width=p.r_eff.value, height=p.r_eff.value*(1-p.ellip.value),
                         angle=Angle(p.theta.value*180./(np.pi), 'deg'))
			ellipsemask = reg.to_mask().multiply(self.bband)

			# test ellipse mask
			newdata = np.zeros_like(self.bband)
			for i in range(self.ysize):
				for j in range(self.xsize):
					if reg.contains(PixCoord(j,i)):
						newdata[i,j] = self.bband[i,j]

			if plot:
				fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10,5))
				axs[0].imshow(self.bband, origin='lower', cmap='viridis', vmin=0, vmax=13)
				patch = reg.plot(ax=axs[0], facecolor='none', edgecolor='red', lw=2)
				axs[1].imshow(p(x, y), origin='lower', cmap='viridis', vmin=0, vmax=13)
				#e1 = mpatches.Ellipse((p.x_0.value, p.y_0.value), p.r_eff.value, p.r_eff.value*(1-p.ellip.value), p.theta.value*180./(np.pi), edgecolor='red', facecolor='none')
				patch = reg.plot(ax=axs[1], facecolor='none', edgecolor='red', lw=2)
				axs[2].imshow(newdata, origin='lower', cmap='viridis', vmin=0, vmax=13)
				plt.show()

			return reg

		if mode=='dbsp':

			spatialwcs = self.wcs.slice(0,1)

			# Get slit region as sky region
			center = SkyCoord(ra=self.voiddata['RA'][self.galaxyidx], dec=self.voiddata['Dec'][self.galaxyidx], unit=(u.hourangle, u.deg))[0]
			width = 1 * u.arcsec
			height = 128 * u.arcsec
			angle = self.voiddata['pa'][self.galaxyidx][0] * u.deg
			regsky = RectangleSkyRegion(center, width, height, angle)
			reg = regsky.to_pixel(spatialwcs)

			# test ellipse mask
			newdata = np.zeros_like(self.bband)
			for i in range(self.ysize):
				for j in range(self.xsize):
					if reg.contains(PixCoord(j,i)):
						newdata[i,j] = self.bband[i,j]

			if plot:
				fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
				axs[0].imshow(self.bband, origin='lower', cmap='viridis', vmin=0, vmax=13)
				patch = reg.plot(ax=axs[0], facecolor='none', edgecolor='red', lw=2)
				axs[1].imshow(newdata, origin='lower', cmap='viridis', vmin=0, vmax=13)
				plt.show()

			return reg

		if mode=='desi':

			spatialwcs = self.wcs.slice(0,1)

			# Get fiber region as sky region
			center = SkyCoord(ra=self.voiddata['RA'][self.galaxyidx], dec=self.voiddata['Dec'][self.galaxyidx], unit=(u.hourangle, u.deg))[0]
			radius = 0.76 * u.arcsec # approximate DESI fiber radius (note: will vary slightly based on location on focal plane)
			regsky = CircleSkyRegion(center, radius)
			reg = regsky.to_pixel(spatialwcs)

			# test ellipse mask
			newdata = np.zeros_like(self.bband)
			for i in range(self.ysize):
				for j in range(self.xsize):
					if reg.contains(PixCoord(j,i)):
						newdata[i,j] = self.bband[i,j]

			if plot:
				fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
				axs[0].imshow(self.bband, origin='lower', cmap='viridis', vmin=0, vmax=13)
				patch = reg.plot(ax=axs[0], facecolor='none', edgecolor='red', lw=2)
				axs[1].imshow(newdata, origin='lower', cmap='viridis', vmin=0, vmax=13)
				plt.show()

			return reg

		if mode=='dbsp2d':

			spatialwcs = self.wcs.slice(0,1)

			# Find max flux
			ind = np.unravel_index(np.argmax(self.bband, axis=None), self.bband.shape)
			center = PixCoord(ind[1],ind[0])
			center = center.to_sky(self.wcs)

			# Get slit region as sky region
			#center = SkyCoord(ra=self.voiddata['RA'][self.galaxyidx], dec=self.voiddata['Dec'][self.galaxyidx], unit=(u.hourangle, u.deg))[0]
			#center = SkyCoord(ra=self.voiddata['RA'][self.galaxyidx], dec=self.voiddata['Dec'][self.galaxyidx], unit=(u.deg, u.deg))[0]
			width = 1 * u.arcsec
			height = 128 * u.arcsec
			angle = self.voiddata['pa'][self.galaxyidx][0] * u.deg
			regsky = RectangleSkyRegion(center, width, height, angle)
			reg = regsky.to_pixel(spatialwcs)

			polygon = reg.to_polygon()
			x = polygon.vertices.xy[0]
			y = polygon.vertices.xy[1]
			lineslope = (y[-1]-y[0])/(x[-1]-x[0])
			lineradius = y[-1]-y[0]
			self.linecenter = PixCoord.from_sky(center, self.wcs)

			self.Nbins = 128

			centerxarray = reg.center.xy[0] + np.arange(-int(self.Nbins/2),int(self.Nbins/2),1)*(reg.width/self.Nbins)
			centeryarray = reg.center.xy[1] + np.arange(-int(self.Nbins/2),int(self.Nbins/2),1)*(reg.height/self.Nbins)

			# test slit mask
			newdata = np.zeros_like(self.bband)
			for i in range(self.ysize):
				for j in range(self.xsize):
					if reg.contains(PixCoord(j,i)):
						newdata[i,j] = self.bband[i,j]

			# Get all sliceregions
			regions = []
			for i in range(self.Nbins):
				pixcenter = PixCoord(centerxarray[i],centeryarray[i]).rotate(self.linecenter,reg.angle)
				slicesky = RectangleSkyRegion(pixcenter.to_sky(self.wcs), 1*u.arcsec, (128/self.Nbins)*u.arcsec, (self.voiddata['pa'][self.galaxyidx][0]+90)*u.deg)
				slicereg = slicesky.to_pixel(spatialwcs)
				regions.append(slicereg)

			# test slice mask
			idx = 64
			newdata_slice = np.zeros_like(self.bband)
			n_goodpix = 0
			for i in range(self.ysize):
				for j in range(self.xsize):
					if regions[idx].contains(PixCoord(j,i)):
						newdata_slice[i,j] = self.bband[i,j]
						n_goodpix += 1
			mask = regions[idx].to_mask()
			checkdata = mask.cutout(self.bband)
			
			if plot:
				fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
				axs[0].imshow(self.bband, origin='lower', cmap='viridis', vmin=0, vmax=13)
				patch = reg.plot(ax=axs[0], facecolor='none', edgecolor='red', lw=2)
				#for i in range(Nbins):
				#	pixcenter = PixCoord(centerxarray[i],centeryarray[i]).rotate(self.linecenter,reg.angle)
				#	pointpix = PointPixelRegion(pixcenter, visual=RegionVisual(symbol='o'))
				#	patch = pointpix.plot(ax=axs[0])
				axs[1].imshow(newdata, origin='lower', cmap='viridis', vmin=0, vmax=13)
				for i in range(self.Nbins):
					pixcenter = PixCoord(centerxarray[i],centeryarray[i]).rotate(self.linecenter,reg.angle)
					slicesky = RectangleSkyRegion(pixcenter.to_sky(self.wcs), 1*u.arcsec, 1*u.arcsec, (self.voiddata['pa'][self.galaxyidx][0]+90)*u.deg)
					slicereg = slicesky.to_pixel(spatialwcs)
					patch = slicereg.plot(ax=axs[1], facecolor='none', edgecolor='red', lw=1)
				#axs[2].imshow(newdata_slice, origin='lower', cmap='viridis', vmin=0, vmax=13)
				plt.show()

			return regions

		# TODO: finish elliptical isophote fitting
		if mode=='isophote':
			pass	

		return
	
	def integrate(self, plot=False, mode='snr', covparams=None, weight='flux', targetsn=10):
		""" Get integrated spectrum

			Args:
				plot (bool): if 'True', make plots
				mode (str): method to use to decide which pixels to stack
					'snr': add pixels with S/N > 1,
					'isophote': add pixels within elliptical isophote (from ellipsefit),
					'sersic': add pixels within best-fit Sersic profile (from ellipsefit)
				covparams (float tuple): (alpha, norm, threshold) parameters for cov correction
				weight (str): how to weight coadds (options: 'flux', 'ivar')
				targetsn (float): target S/N for 2d binning
		"""

		# Apply covariance correction
		if covparams is not None:
			alpha, norm, threshold = covparams
			self.covcorr = norm * (1 + alpha * np.log(threshold))
			self.var *= self.covcorr**2.

		def coadd(newdata, newvar, weight=weight):
			""" Coadd spectra - use modified version of coadd function (https://spectrum.readthedocs.io/en/latest/_modules/spectrum/coadd.html) """

			# First, make sure there is no flux defined if there is no error.
			newvar = np.ma.fix_invalid(newvar)
			if np.ma.is_masked(newvar):
				newdata[newvar.mask] = np.ma.masked
			# This can be simplified considerably as soon as masked quantities exist.
			newdata = np.ma.fix_invalid(newdata)

			# Flux weighting
			if weight=='flux':
				sn_wvl=[4750.,4800.]
				goodwvl_sn = np.where((self.wvl_zcorr > sn_wvl[0]) & (self.wvl_zcorr < sn_wvl[1]))[0]
				flux = np.nansum(newdata[goodwvl_sn,:,:], axis=0)
				flux = np.repeat(flux[np.newaxis, :, :], np.shape(newdata)[0], axis=0)
				totalspec = np.ma.average(newdata, axis=(1,2), weights=flux).filled(np.nan) 

			# Do coaddition
			elif weight=='var':
				totalspec = np.ma.average(newdata, axis=(1,2), weights = 1./newvar).filled(np.nan) 

			totalvar = 1. / np.ma.sum(1./newvar, axis=(1,2)).filled(np.nan)

			assert not np.ma.isMaskedArray(totalspec)
			assert not np.ma.isMaskedArray(totalvar)

			# Compute final signal
			signalfinal = np.mean(totalspec[self.snwvl])

			# Compute noise as detrended standard deviation
			linfit = np.polyfit(self.wvl_zcorr[self.snwvl],totalspec[self.snwvl],deg=1)
			poly = np.poly1d(linfit)
			noisefinal = np.std(totalspec[self.snwvl] - np.asarray(poly(self.wvl_zcorr[self.snwvl])))
			sn = signalfinal/noisefinal

			return totalspec, totalvar, sn

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

			self.totalspec, self.totalvar, self.sn = coadd(newdata, newvar)

		if mode in ['sersic','dbsp', 'desi']:
			# Create mask
			newdata = np.zeros_like(self.data)
			newvar =  np.zeros_like(self.data)
			n_goodpix = 0
			reg = self.ellipsefit(plot=False, mode=mode)
			for i in range(self.ysize):
				for j in range(self.xsize):
					if reg.contains(PixCoord(j,i)):
						newdata[:,i,j] = self.data[:,i,j]
						newvar[:,i,j] = self.var[:,i,j]
						n_goodpix += 1
			#print(n_goodpix)

			self.totalspec, self.totalvar, self.sn = coadd(newdata, newvar)
			self.mode = mode

		if mode in ['snr','sersic','dbsp', 'desi'] and plot:
			#print(self.sn)
			plt.plot(self.wvl_zcorr[self.goodwvl], self.totalspec[self.goodwvl], color='k', lw=1)
			plt.fill_between(self.wvl_zcorr[self.goodwvl], (self.totalspec-np.sqrt(self.totalvar))[self.goodwvl], \
				(self.totalspec+np.sqrt(self.totalvar))[self.goodwvl], color='r', alpha=0.5)
			plt.show()

		if mode=='dbsp2d':
			# Create mask
			regions = self.ellipsefit(plot=False, mode=mode)
			reg_spectra = np.zeros((len(regions), np.shape(self.data)[0]))
			reg_var = np.zeros((len(regions), np.shape(self.data)[0]))
			reg_sn = np.zeros(len(regions))
			reg_x = np.zeros(len(regions))
			reg_y = np.zeros(len(regions))

			for regidx, reg in enumerate(regions):
				# Check if the region even overlaps the data
				mask = reg.to_mask()
				checkdata = mask.cutout(self.bband)
				
				if checkdata is not None:
					newdata = np.zeros_like(self.data)
					newvar =  np.zeros_like(self.data)
					n_goodpix = 0
					for i in range(self.ysize):
						for j in range(self.xsize):
							if reg.contains(PixCoord(j,i)):
								newdata[:,i,j] = self.data[:,i,j]
								newvar[:,i,j] = self.var[:,i,j]
								n_goodpix += 1
					totalspec, totalvar, sn = coadd(newdata,newvar)

					reg_spectra[regidx,:] = totalspec
					reg_var[regidx,:] = totalvar
					reg_sn[regidx] = sn
					reg_x[regidx] = reg.center.x
					reg_y[regidx] = reg.center.y

			def bin2dspec(reg_spectra, reg_var, reg_sn, reg_x, reg_y, plot=False, covparams=None, targetsn=targetsn):
				""" Bin 2d spectra by S/N. """

				#reg_spectra, reg_var, reg_sn, reg_x, reg_y = self.integrate(mode='dbsp2d', plot=plot, covparams=covparams, weight='flux')
				#print(np.array2string(reg_sn, separator=', '))

				# First cut out any regions with S/N < 3
				goodidx = np.where(reg_sn > 3)[0]
				reg_sn = reg_sn[goodidx]
				reg_spectra = reg_spectra[goodidx,:]
				reg_var = reg_var[goodidx,:]
				reg_x = reg_x[goodidx]
				reg_y = reg_y[goodidx]

				def coadd(idx):

					# Get spectra
					spec = reg_spectra[idx,:]
					var = reg_var[idx,:]

					# Do flux-weighting
					sn_wvl=[4750.,4800.]
					goodwvl_sn = np.where((self.wvl_zcorr > sn_wvl[0]) & (self.wvl_zcorr < sn_wvl[1]))[0]
					flux = np.nansum(spec[:,goodwvl_sn], axis=1)
					totalspec = np.ma.average(spec, axis=0, weights=flux).filled(np.nan)

					totalvar = 1. / np.ma.sum(1./var, axis=0).filled(np.nan)

					assert not np.ma.isMaskedArray(totalspec)
					assert not np.ma.isMaskedArray(totalvar)

					# Compute final signal
					signalfinal = np.mean(totalspec[self.snwvl])

					# Compute noise as detrended standard deviation
					linfit = np.polyfit(self.wvl_zcorr[self.snwvl],totalspec[self.snwvl],deg=1)
					poly = np.poly1d(linfit)
					noisefinal = np.std(totalspec[self.snwvl] - np.asarray(poly(self.wvl_zcorr[self.snwvl])))
					sn = signalfinal/noisefinal

					return totalspec, totalvar, sn

				pixellist = list(range(len(reg_sn)))
				bins = []
				currentbin = []

				# Start from pixel with highest S/N
				pixel = np.argmax(reg_sn)

				while len(pixellist) > 0:

					# Add current pixel to current bin
					currentbin.append(pixel)
					pixellist.remove(pixel)

					# Find nearest unbinned pixel
					if len(pixellist) > 0:
						newpixelidx = np.argmin(np.abs(np.asarray(pixellist) - pixel))
						newpixel = pixellist[newpixelidx]
					else:
						newpixel = None

					# If we've reached the end and can't bin, just add the currentbin on to the nearest bin anyway
					if newpixel is None or ((newpixel is not None) and (np.abs(pixel - newpixel) > 1)):
						bins[-1] += currentbin
						pixel = newpixel
						currentbin = []
						continue
				
					# Check if target S/N has been reached:
					_, _, computesn = coadd(currentbin)
					if computesn > targetsn:

						# If so, we're done with this bin
						bins.append(currentbin)

						# Reset to new pixel and new currentbin
						pixel = newpixel
						currentbin = []

					else:
						# If target S/N hasn't been reached, currentbin doesn't change
						pixel = newpixel
						pass

				#print(bins)
				#print(len(bins))

				# Save binned spectra
				binned_spec = np.zeros((len(bins), np.shape(self.data)[0]))
				binned_var = np.zeros((len(bins), np.shape(self.data)[0]))
				binned_sn = np.zeros(len(bins))
				binned_centers = np.zeros(len(bins), dtype=object)
				for binidx in range(len(bins)):
					binned_spec[binidx,:], binned_var[binidx,:], binned_sn[binidx] = coadd(bins[binidx])
					binned_x = np.average(reg_x[bins[binidx]])
					binned_y = np.average(reg_y[bins[binidx]])
					binned_centers[binidx] = PixCoord(binned_x,binned_y)

				#print(self.Nbins, len(reg_x), len(bins))
				#print(binned_sn)

					#if plot:
					#	print(bins[binidx])
					#	plt.plot(self.wvl_zcorr[self.goodwvl], binned_spec[binidx,self.goodwvl], color='k', lw=1, label='S/N: {:.2f}'.format(binned_sn[binidx]))
					#	plt.legend()
					#	plt.show()
				
				fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
				axs[0].imshow(self.bband, origin='lower', cmap='viridis', vmin=0, vmax=13)
				for i in range(len(reg_x)):
					pixcenter = PixCoord(reg_x[i], reg_y[i])
					slicesky = RectangleSkyRegion(pixcenter.to_sky(self.wcs), (128./self.Nbins)*u.arcsec, 1*u.arcsec, (self.voiddata['pa'][self.galaxyidx][0]+90)*u.deg)
					slicereg = slicesky.to_pixel(self.wcs.slice(0,1))
					patch = slicereg.plot(ax=axs[0], facecolor='none', edgecolor='red', lw=1)
					#pointpix = PointPixelRegion(pixcenter, visual=RegionVisual(symbol='o',color='red'))
					#patch = pointpix.plot(ax=axs[0])
				axs[1].imshow(self.bband, origin='lower', cmap='viridis', vmin=0, vmax=13)
				for i in range(len(bins)):
					pixcenter = binned_centers[i]
					slicesky = RectangleSkyRegion(pixcenter.to_sky(self.wcs), len(bins[i])*(128./self.Nbins)*u.arcsec, 1*u.arcsec, (self.voiddata['pa'][self.galaxyidx][0]+90)*u.deg)
					slicereg = slicesky.to_pixel(self.wcs.slice(0,1))
					patch = slicereg.plot(ax=axs[1], facecolor='none', edgecolor='red', lw=1)
				plt.savefig('figures/kinematics2d/'+self.galaxyname+'_intspec.pdf', bbox_inches='tight')

				if plot:
					plt.show()
				else:
					plt.close()
				
				print('Done with binning')

				return binned_spec, binned_var, binned_sn, binned_centers
			
			self.binned_spec, self.binned_var, self.binned_sn, self.binned_centers = bin2dspec(reg_spectra, reg_var, reg_sn, reg_x, reg_y, plot=plot)

		return

	def prepstellarfit(self, spectrum, wvl):
		""" Prepare stacked cube for spectral fitting with pPXF. Only need to run this once per galaxy!

			Arguments:
				spectrum, wvl (1D arrays): observed spectrum to use as template (default: use stacked spectrum)
		"""

		print('Preparing for stellar kinematics fit...')

		# Define path to pPXF directory
		ppxf_dir = path.dirname(path.realpath(ppxf_package.__file__))
		print(ppxf_dir)

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
		if spectrum.size != self.wvl_zcorr[self.goodwvl].size:
			raise ValueError('Size of spectrum array %s does not match size of wvl array %s' % (spectrum.size, self.wvl_zcorr[self.goodwvl].size))
		if noise.size != self.wvl_zcorr[self.goodwvl].size:
			raise ValueError('Size of noise array %s does not match size of wvl array %s' % (noise.size, self.wvl_zcorr[self.goodwvl].size))

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

	def stellarfit(self, plot=True, specdim=1, nangle = 1, removekinematics=False):
		""" Fit stellar continuum of integrated spectra, return stellar kinematics,
			and subtract continuum/absorption features.

			Args:
				plot (bool): if 'True', make plots
				specdim (int): number of dimensions of spectra (1 = treat like single integrated spectrum, 2 = compute spectrum from bins)
				nangle (int): number of PAs to try
				removekinematics (bool): if 'True', continuum normalize

			Returns:
				kinematics (array): [vel, veldisp, vel_err, veldisp_err, wvlarray, fitspectrum, obsspectrum, obsspectrum_err]
		"""

		# Integrated spectrum
		if specdim == 1:
			self.prepstellarfit(self.totalspec[self.goodwvl], self.wvl_zcorr[self.goodwvl])

			print(self.totalvar[self.goodwvl])
			params, fitwvl, fit, obs, obserr = self.ppxf_fit(self.totalspec[self.goodwvl], self.totalvar[self.goodwvl], verbose=True)

			print("v: {:.2f}\t{:.2f}".format(params[0], params[2]))
			print("sigma: {:.2f}\t{:.2f}".format(params[1], params[3]))

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
			plt.text(3750, 1.35, 'Integrated light: S/N={:.1f}'.format(self.sn), fontsize=15)
			plt.ylim(0.5,1.5)
			plt.xlim(3700,5100)
			plt.savefig('figures/'+self.galaxyname+'/'+'intspec_'+self.mode+'.pdf', bbox_inches='tight') 
			if plot:
				plt.show()
			else:
				plt.close()

			# Subtract stellar contribution from spectrum
			print('Normalizing data by best-fit stellar template...')

			# Smooth observed spectrum to match template
			spectrum = self.totalspec[self.goodwvl]
			galspec = ndimage.gaussian_filter1d(spectrum, self.sigma)

			# Save log-rebinned spectrum
			galaxy, _, _ = util.log_rebin(self.lamRange1, galspec)

			if removekinematics:
				# Subtract kinematics fit
				self.spectrum_norm = galaxy - fit*np.median(galaxy)
				self.kinematics_wvl = fitwvl

				#plt.plot(self.kinematics_wvl, self.spectrum_norm, 'k-')
				#plt.show()

				np.save('output/'+self.galaxyname+'/intspec_norm', self.spectrum_norm)
				np.save('output/'+self.galaxyname+'/intspec_wvl', self.kinematics_wvl)

		# Binned 2d spectrum
		if specdim == 2:
			# Prep the stellar fit
			self.prepstellarfit(self.binned_spec[0][self.goodwvl], self.wvl_zcorr[self.goodwvl])

			# Create data structures to hold final outputs
			self.vel = np.zeros(len(self.binned_sn))
			self.veldisp = np.zeros(len(self.binned_sn))
			self.vel_err = np.zeros(len(self.binned_sn))
			self.veldisp_err = np.zeros(len(self.binned_sn))
			self.kinematics_fit_bin = np.zeros(np.shape(self.binned_spec[:,self.goodwvl]))
			self.kinematics_wvl_bin = np.zeros(np.shape(self.binned_spec[:,self.goodwvl]))

			# Loop over all bins
			for binID in tqdm(range(len(self.binned_sn))):

				# Do stellar kinematic fit
				params, fitwvl, fit, obs, obserr = self.ppxf_fit(self.binned_spec[binID][self.goodwvl], self.binned_var[binID][self.goodwvl], verbose=False)

				if np.isclose(params[2],0.) or np.isclose(params[3],0.): # or (params[2] > 70):
					params = [np.nan, np.nan, np.nan, np.nan, np.nan]

				if plot:
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
						plt.text(3750, 1.35, 'Central bin: S/N={:.1f}'.format(self.binned_sn[binID]), fontsize=15)
						plt.ylim(0.5,1.5)
						plt.xlim(3700,5100)
						plt.savefig('figures/kinematics2d/'+self.galaxyname+'_centerspec1d.pdf', bbox_inches='tight') 
						plt.show()

				# Save data from each bin
				self.vel[binID] = params[0]
				self.veldisp[binID] = params[1]
				self.vel_err[binID] = params[2]
				self.veldisp_err[binID] = params[3]

				# Put fit for each bin into an array
				self.kinematics_fit_bin[binID] = fit
				self.kinematics_wvl_bin[binID] = fitwvl
			
			# Compute global quantities

			# Compute systemic velocity
			sn_wvl=[4750.,4800.]
			goodwvl_sn = np.where((self.wvl_zcorr > sn_wvl[0]) & (self.wvl_zcorr < sn_wvl[1]))[0]
			goodidx = np.where(~np.isnan(self.vel) & ~np.isnan(self.veldisp))
			flux = np.nansum(self.binned_spec[:,goodwvl_sn], axis=1)
			systvel = np.average(self.vel[goodidx], weights=1./(self.vel_err[goodidx]**2.))
			print("systvel: {:.2f}".format(systvel))

			# Subtract systemic velocity
			self.vel += -systvel

			# Compute vmax
			goodidx = np.where((self.vel_err > 0.) & (self.veldisp_err > 0))
			maxv = np.median([np.percentile(self.vel[goodidx], 95), np.max(self.vel[goodidx])])
			minv = np.median([np.min(self.vel[goodidx]), np.percentile(self.vel[goodidx], 5)])
			maxv_err = np.max(self.vel[goodidx]) - np.percentile(self.vel[goodidx], 95)
			minv_err = np.percentile(self.vel[goodidx], 5) - np.min(self.vel[goodidx])
			vmax = 0.5*(maxv - minv)
			vmax_err = 0.5*np.sqrt(maxv_err**2. + minv_err**2.)
			print(r"vmax: {:.2f} \pm {:.2f}".format(vmax, vmax_err))

			# Compute sigma
			goodidx = np.where(~np.isnan(self.vel) & ~np.isnan(self.veldisp) & (self.veldisp_err > 1e-5) & (self.veldisp > 1.)) # & (self.veldisp > self.veldisp_err))
			sigma = np.average(self.veldisp[goodidx], weights=flux[goodidx])
			sigma_err = np.sqrt(np.sum( self.veldisp_err[goodidx]**2. * (flux[goodidx]/np.sum(flux[goodidx]))**2. ))
			print("sigma: {:.2f} \pm {:.2f}".format(sigma, sigma_err))

			# Compute v/sigma
			vsigma = vmax/sigma
			vsigma_err = np.sqrt((vmax_err/vmax)**2. + (sigma_err/sigma)**2.) * vsigma
			print("vsigma: {:.2f} \pm {:.2f}".format(vsigma, vsigma_err))
			
			# Compute distances from center
			distances = np.zeros(len(self.binned_centers))
			center = SkyCoord(ra=self.voiddata['RA'][self.galaxyidx], dec=self.voiddata['Dec'][self.galaxyidx], unit=(u.hourangle, u.deg))[0]
			for i in range(len(distances)):
				bincenter_sky = self.binned_centers[i].to_sky(self.wcs)
				theta = center.separation(bincenter_sky).to(u.arcsec)
				d_A = cosmo.angular_diameter_distance(z=self.z)
				distances[i] = (theta * d_A).to(u.kpc, u.dimensionless_angles()).value # in kpc

			# Plot 1d kinematic info
			fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,5))

			# Velocities
			pos_vel = np.where(self.vel > 0.)[0]
			neg_vel = np.where(self.vel < 0.)[0]
			axs[0].plot(distances[pos_vel], np.abs(self.vel[pos_vel]), 'ko')
			axs[0].plot(distances[neg_vel], np.abs(self.vel[neg_vel]), color='k', marker='o', fillstyle='none', linestyle='None')
			axs[0].axhline(vmax, color='r', linestyle='--', lw=2)
			axs[0].axhspan(vmax-vmax_err, vmax+vmax_err, color='r', alpha=0.3)
			axs[0].set_ylabel(r'$v_{i}$ (km/s)', fontsize=18)

			# Velocity dispersions
			axs[1].plot(distances, self.veldisp, 'ko')
			axs[1].axhline(sigma, color='r', linestyle='--', lw=2)
			axs[1].axhspan(sigma-sigma_err, sigma+sigma_err, color='r', alpha=0.3)
			axs[1].set_ylabel(r'$\sigma_{\star,i}$ (km/s)', fontsize=18)

			for ax in axs:
				ax.tick_params(axis='both', labelsize=14)
				ax.set_xlabel('Distance from center (kpc)', fontsize=18)

			plt.savefig('figures/kinematics2d/'+self.galaxyname+'_kinematics2d.png', bbox_inches='tight')
			plt.close()

			if removekinematics:
				# Subtract stellar contribution from spectrum
				print('Normalizing data by best-fit stellar template...')
				
				# Prep array for output
				self.spectrum_norm = np.zeros(np.shape(self.binned_spec[:,self.goodwvl]))
				self.kinematics_wvl = fitwvl

				# Loop over all bins
				for binID in range(len(self.binned_sn)):

					# Smooth observed spectrum to match template
					spectrum = self.binned_spec[binID,self.goodwvl]
					galspec = ndimage.gaussian_filter1d(spectrum, self.sigma)

					# Save log-rebinned spectrum
					galaxy, _, _ = util.log_rebin(self.lamRange1, galspec)

					# Subtract kinematics fit
					self.spectrum_norm[binID] = galaxy - self.kinematics_fit_bin[binID]*np.median(galaxy)

				#plt.plot(self.kinematics_wvl, self.spectrum_norm[0], 'k-')
				#plt.show()

				# Save file
				np.save('output/'+self.galaxyname+'/2dspec_norm', self.spectrum_norm)
				np.save('output/'+self.galaxyname+'/2dspec_wvl', self.kinematics_wvl)

		return

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
	c.ellipsefit(plot=True, mode='dbsp2d')
	c.integrate(plot=True, covparams=param['covparams'], mode='dbsp2d', targetsn=param['targetsn'])
	c.stellarfit(plot=True, specdim=2)

	#c.reddening(verbose=False)
	#c.metallicity_Te(verbose=False)

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
			integratedpipeline(galaxy, folder='/Users/miadelosreyes/Documents/Research/VoidDwarfs/redux/stackedcubes/')
		except:
			print('Failed on '+galaxy)

	return

def main():

	#runallgalaxies()

	integratedpipeline('825059', folder='/Users/miadelosreyes/Documents/Research/VoidDwarfs/redux/stackedcubes/')

	#c = Cube('reines65', folder='/Users/miadelosreyes/Documents/Research/VoidDwarfs/redux/stackedcubes/', verbose=False, wcscorr=[174.17801 - 174.1787083, 26.727126 - 26.7263583], z=0.0331, EBV=0.0217)
	#c.ellipsefit(plot=True, mode='dbsp')
	#c.integrate(plot=False, covparams=None, mode='sersic') #[0.108,1.65,80])
	#c.stellarfit(plot=False)
	#c.reddening(verbose=False)
	#c.metallicity_Te(verbose=True)

	return

if __name__ == "__main__":
	main()
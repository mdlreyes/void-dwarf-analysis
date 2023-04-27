# kcwiintegrated.py
# Script to compute integrated properties
# (Modified version to test putting DBSP slits at different angles on galaxies.)
# from KCWI data cubes
#
# Created: 3 Feb 2023
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
import csv

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

	def ellipsefit(self, plot=True, angleoffset=0):
		""" Fit cube with elliptical isophote

			Args:
				plot (bool): if 'True', make plots
				angleoffset (float): offset angle (only works for mode=='dbsp2d')

			Outputs:
				Re (float): half-light radius
		"""

		print('Fitting aperture to white-light image...')

		# Get DBSP-like slit region

		spatialwcs = self.wcs.slice(0,1)

		# Find max flux
		ind = np.unravel_index(np.argmax(self.bband, axis=None), self.bband.shape)
		center = PixCoord(ind[1],ind[0])
		center = center.to_sky(self.wcs)

		# Get slit region as sky region
		#center = SkyCoord(ra=self.voiddata['RA'][self.galaxyidx], dec=self.voiddata['Dec'][self.galaxyidx], unit=(u.hourangle, u.deg))[0]
		width = 1 * u.arcsec
		height = 128 * u.arcsec
		angle = (self.voiddata['pa'][self.galaxyidx][0] + angleoffset) * u.deg
		regsky = RectangleSkyRegion(center, width, height, angle)
		reg = regsky.to_pixel(spatialwcs)

		# Center of slit in pixel coords
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
		
		if plot:
			fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
			axs[0].imshow(self.bband, origin='lower', cmap='viridis', vmin=0, vmax=13)
			patch = reg.plot(ax=axs[0], facecolor='none', edgecolor='red', lw=2)
			axs[1].imshow(newdata, origin='lower', cmap='viridis', vmin=0, vmax=13)
			for i in range(self.Nbins):
				pixcenter = PixCoord(centerxarray[i],centeryarray[i]).rotate(self.linecenter,reg.angle)
				slicesky = RectangleSkyRegion(pixcenter.to_sky(self.wcs), 1*u.arcsec, 1*u.arcsec, (self.voiddata['pa'][self.galaxyidx][0]+90)*u.deg)
				slicereg = slicesky.to_pixel(spatialwcs)
				patch = slicereg.plot(ax=axs[1], facecolor='none', edgecolor='red', lw=1)
			plt.show()

		return regions
	
	def integrate(self, plot=False, covparams=None, weight='flux', targetsn=10, angleoffset=0):
		""" Get integrated spectrum

			Args:
				plot (bool): if 'True', make plots
				covparams (float tuple): (alpha, norm, threshold) parameters for cov correction
				weight (str): how to weight coadds (options: 'flux', 'ivar')
				targetsn (float): target S/N for 2d binning
				angleoffset (float): offset slit angle (only valid for mode=='dbsp2d')
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

		# Create mask
		regions = self.ellipsefit(plot=False, angleoffset=angleoffset)
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

		def bin2dspec(reg_spectra, reg_var, reg_sn, reg_x, reg_y, plot=False, targetsn=targetsn):
			""" Bin 2d spectra by S/N. """

			# First cut out any regions with S/N < 3
			goodidx = np.where(reg_sn > 3)[0]
			reg_sn = reg_sn[goodidx]
			reg_spectra = reg_spectra[goodidx,:]
			reg_var = reg_var[goodidx,:]
			reg_x = reg_x[goodidx]
			reg_y = reg_y[goodidx]

			def bin_coadd(idx):

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
				_, _, computesn = bin_coadd(currentbin)
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

			# Save binned spectra
			binned_spec = np.zeros((len(bins), np.shape(self.data)[0]))
			binned_var = np.zeros((len(bins), np.shape(self.data)[0]))
			binned_sn = np.zeros(len(bins))
			binned_centers = np.zeros(len(bins), dtype=object)
			for binidx in range(len(bins)):
				binned_spec[binidx,:], binned_var[binidx,:], binned_sn[binidx] = bin_coadd(bins[binidx])
				binned_x = np.average(reg_x[bins[binidx]])
				binned_y = np.average(reg_y[bins[binidx]])
				binned_centers[binidx] = PixCoord(binned_x,binned_y)

			# Make plots
			fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
			axs[0].imshow(self.bband, origin='lower', cmap='viridis', vmin=0, vmax=13)
			for i in range(len(reg_x)):
				pixcenter = PixCoord(reg_x[i], reg_y[i])
				slicesky = RectangleSkyRegion(pixcenter.to_sky(self.wcs), (128./self.Nbins)*u.arcsec, 1*u.arcsec, (self.voiddata['pa'][self.galaxyidx][0]+90)*u.deg)
				slicereg = slicesky.to_pixel(self.wcs.slice(0,1))
				patch = slicereg.plot(ax=axs[0], facecolor='none', edgecolor='red', lw=1)
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
		
		binned_spec, binned_var, binned_sn, binned_centers = bin2dspec(reg_spectra, reg_var, reg_sn, reg_x, reg_y, plot=plot)

		return binned_spec, binned_var, binned_sn, binned_centers

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

	def stellarfit(self, plot=True, nangles=1, removekinematics=False, targetsn=10):
		""" Fit stellar continuum of integrated spectra, return stellar kinematics,
			and subtract continuum/absorption features.

			Args:
				plot (bool): if 'True', make plots
				specdim (int): number of dimensions of spectra (1 = treat like single integrated spectrum, 2 = compute spectrum from bins)
				nangles (int): number of PAs to try
				removekinematics (bool): if 'True', continuum normalize spectrum

			Returns:
				kinematics (array): [vel, veldisp, vel_err, veldisp_err, wvlarray, fitspectrum, obsspectrum, obsspectrum_err]
		"""

		angle_vrot = np.zeros(nangles)
		angles = np.linspace(0, 180, nangles, endpoint=False)

		# Loop over each slit angle
		for anglenum in range(nangles):

			try:

				# Integrate spectra
				binned_spec, binned_var, binned_sn, _ = self.integrate(plot=False, covparams=None, weight='flux', targetsn=targetsn, angleoffset=angles[anglenum])

				# Prep the stellar fit
				self.prepstellarfit(binned_spec[0][self.goodwvl], self.wvl_zcorr[self.goodwvl])

				# Create data structures to hold final outputs
				vel = np.zeros(len(binned_sn))
				veldisp = np.zeros(len(binned_sn))
				vel_err = np.zeros(len(binned_sn))
				veldisp_err = np.zeros(len(binned_sn))

				# Loop over all bins
				for binID in tqdm(range(len(binned_sn))):

					# Do stellar kinematic fit
					params, fitwvl, fit, _, _ = self.ppxf_fit(binned_spec[binID][self.goodwvl], binned_var[binID][self.goodwvl], verbose=False)

					if np.isclose(params[2],0.) or np.isclose(params[3],0.): # or (params[2] > 70):
						params = [np.nan, np.nan, np.nan, np.nan, np.nan]

					# Save data from each bin
					vel[binID] = params[0]
					veldisp[binID] = params[1]
					vel_err[binID] = params[2]
					veldisp_err[binID] = params[3]

				# Compute global quantities

				# Compute systemic velocity
				sn_wvl=[4750.,4800.]
				goodwvl_sn = np.where((self.wvl_zcorr > sn_wvl[0]) & (self.wvl_zcorr < sn_wvl[1]))[0]
				goodidx = np.where(~np.isnan(vel) & ~np.isnan(veldisp))
				flux = np.nansum(binned_spec[:,goodwvl_sn], axis=1)
				systvel = np.average(vel[goodidx], weights=1./(vel_err[goodidx]**2.))

				# Subtract systemic velocity
				vel += -systvel

				# Compute vmax
				goodidx = np.where((vel_err > 0.) & (veldisp_err > 0))
				maxv = np.median([np.percentile(vel[goodidx], 95), np.max(vel[goodidx])])
				minv = np.median([np.min(vel[goodidx]), np.percentile(vel[goodidx], 5)])
				vmax = 0.5*(maxv - minv)
				print(vmax)
				angle_vrot[anglenum] = vmax

			except:
				pass

		# Compute which slit had the max vrot
		bestanglenum = np.argmax(angle_vrot)
		print(bestanglenum)

		# So now redo everything but for the best slit
		self.binned_spec, self.binned_var, self.binned_sn, self.binned_centers = self.integrate(plot=False, covparams=None, weight='flux', targetsn=targetsn, angleoffset=angles[bestanglenum])
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
		try:
			goodidx = np.where(~np.isnan(self.vel) & ~np.isnan(self.veldisp) & (self.veldisp_err > 1e-5) & (self.veldisp > 1.)) # & (self.veldisp > self.veldisp_err))
			sigma = np.average(self.veldisp[goodidx], weights=flux[goodidx])
			sigma_err = np.sqrt(np.sum( self.veldisp_err[goodidx]**2. * (flux[goodidx]/np.sum(flux[goodidx]))**2. ))
			print("sigma: {:.2f} \pm {:.2f}".format(sigma, sigma_err))

			# Compute v/sigma
			vsigma = vmax/sigma
			vsigma_err = np.sqrt((vmax_err/vmax)**2. + (sigma_err/sigma)**2.) * vsigma
			print("vsigma: {:.2f} \pm {:.2f}".format(vsigma, vsigma_err))

		except:
			sigma = np.nan
			sigma_err = np.nan
			vsigma = np.nan
			vsigma_err = np.nan
		
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

			# Save file
			np.save('output/'+self.galaxyname+'/2dspec_norm', self.spectrum_norm)
			np.save('output/'+self.galaxyname+'/2dspec_wvl', self.kinematics_wvl)

		# Return data
		data = [self.galaxyname, systvel, vmax, vmax_err, sigma, sigma_err, vsigma, vsigma_err, angles[bestanglenum]]

		return data

def integratedpipeline(galaxyname, folder='/raid/madlr/voids/analysis/stackedcubes/', mode='sersic', nangles=1):
	""" Run full pipeline to get: 
		- Systemic stellar kinematics
		- Gas-phase abundances

	Arguments:
		galaxyname (str): name of galaxy
		folder (str): folder where stacked data cubes are stored
		mode (str): which mode to use to integrate
		nangles (int): number of angles to place slit
	"""

	# Open params
	param = params[galaxyname]

	# Open cube
	c = Cube(galaxyname, folder=folder, verbose=param['verbose'], wcscorr=param['wcscorr'], z=param['z'], EBV=param['EBV'])
	#c.ellipsefit(plot=True, mode='desi')
	#if mode=='dbsp2' and nangles==1:
	#	c.integrate(plot=True, covparams=param['covparams'], mode='desi', targetsn=param['targetsn'])
	data = c.stellarfit(plot=False, nangles=nangles, targetsn=param['targetsn'])

	#c.reddening(verbose=False)
	#c.metallicity_Te(verbose=False)

	return data

def runallgalaxies(nangles):

	# List of all galaxies
	galaxylist = ['reines65','1180506','281238','1142116','1876887','1904061','2502521','821857',
			'1126100','1158932','1782069','1785212','866934','825059','1063413','1074435',
			'1228631','1246626','955106','1280160','control757','control801','control872',
			'control842','PiscesA','PiscesB','control751','control775','control658']

	# Open file to write in
	header = ['ID','systvel','vmax','vmax_err','sigma','sigma_err','vsigma','vsigma_err']
	with open('output/vsigma_%d.txt'%nangles, 'w') as f:
		writer = csv.writer(f)
		writer.writerow(header)

	# Run reduction pipeline for each galaxy
	for galaxy in galaxylist:
		try:
			data = integratedpipeline(galaxy, folder='/Users/miadelosreyes/Documents/Research/VoidDwarfs/redux/stackedcubes/', nangles=nangles)
			with open('output/vsigma_%d.txt'%nangles, 'a') as f:
				writer = csv.writer(f)
				writer.writerow(data)
		except:
			print('Failed on '+galaxy)

	return

def main():

	runallgalaxies(4)

	#data = integratedpipeline('1904061', folder='/Users/miadelosreyes/Documents/Research/VoidDwarfs/redux/stackedcubes/', nangles=2)
	#print(data)

	#c = Cube('reines65', folder='/Users/miadelosreyes/Documents/Research/VoidDwarfs/redux/stackedcubes/', verbose=False, wcscorr=[174.17801 - 174.1787083, 26.727126 - 26.7263583], z=0.0331, EBV=0.0217)
	#c.ellipsefit(plot=True, mode='dbsp')
	#c.integrate(plot=False, covparams=None, mode='sersic') #[0.108,1.65,80])
	#c.stellarfit(plot=False)
	#c.reddening(verbose=False)
	#c.metallicity_Te(verbose=True)

	return

if __name__ == "__main__":
	main()
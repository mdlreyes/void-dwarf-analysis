# kcwiredux.py
# Script to analyze KCWI data cubes
# COMPUTES STELLAR KINEMATICS ONLY
# Includes MC error estimation for v/sigma
# USE ON SHERLOCK
#
# Created: 9 Aug 2020
######################################

import numpy as np
from numpy.random import default_rng
from astropy.io import fits
from astropy.wcs import WCS
import os
import sys
from params import params

# Packages for binning
#import kcwialign
import vorbin
from vorbin.voronoi_2d_binning import voronoi_2d_binning

# Packages for stellar continuum fitting
import ppxf as ppxf_package
from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
import glob
from scipy import ndimage

# Packages for emission line fitting
from k_lambda import k_lambda

# Packages for parallelization
import concurrent.futures
import functools
from tqdm import tqdm

# Wavelength dictionary for standard lines (from NIST when possible)
wvldict = {'Hbeta':4861.35, 'Hgamma':4340.472, 'Hdelta':4101.734, 'Hepsilon':3970.075,
		'OII3727':3727.320, 'OII3729':3729.225, 'OII3727_doublet':3728., 'OIII4363':4363.209, 'OIII4959':4959., 'OIII5007':5006.8}

# Function to parallelize
def binfit(binID, stacked_spec_new, goodwvl_sn, templates, sigma, lamRange1, logLam2, lamRange2):
	"""Do PPXF prep and run PPXF"""

	# Prep the observed spectrum
	galspec = ndimage.gaussian_filter1d(stacked_spec_new[binID], sigma)
	galaxy, logLam1, velscale = util.log_rebin(lamRange1, galspec)
	galaxy = galaxy/np.median(galaxy)

	# Shift the template to fit the starting wavelength of the galaxy spectrum
	c = 299792.458
	dv = (logLam2[0] - logLam1[0])*c  # km/s

	goodPixels = util.determine_goodpixels(logLam1, lamRange2, 0)

	# Here the actual fit starts. The best fit is plotted on the screen
	start = [0., 200.]  # (km/s), starting guess for [V, sigma]

	pp = ppxf(templates, galaxy, np.full_like(galaxy, 1e-3), velscale, start,
				goodpixels=goodPixels, moments=2, degree=6, vsyst=dv, clean=False, quiet=True)

	return binID, np.asarray([pp.sol[0], pp.sol[1], pp.error[0]*np.sqrt(pp.chi2)])

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

	Methods:

	"""

	def __init__(self, filename, folder='/oak/stanford/orgs/kipac/users/mdlreyes/data/stackedcubes/', wcscorr=None, z=0., sn_wvl=[4750.,4800.], wvlrange=[3700., 5100.], EBV=0.):

		"""Opens datacube and sets base attributes.

			Args:
				filename (str): Name of cube to open.
				folder (str): Path in which data are stored.
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
		print(self.galaxyname)
		print('Initializing cube...')

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

	def binspaxels(self, params=[1., 1., 60.], targetsn=10.):
		""" Bin spaxels spatially to increase S/N

			Args:
				params (float list): parameters [alpha, norm, thresh] to use for covar correction
				targetsn (float): target value of S/N
		"""

		print('Binning cube...')

		# Define sizes of array
		xsize = np.shape(self.data[0,:,:])[1]
		ysize = np.shape(self.data[0,:,:])[0]

		# Compute continuum signal/noise
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
		np.save('output/'+self.galaxyname+'/contsnr', np.ma.getdata(sntest))

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
		self.binNum, xNode, yNode, _, _, self.sn, _, _ = voronoi_2d_binning(self.x, self.y, s, n, targetsn, sn_func=snfunc, plot=0, quiet=1)

		# Get list of bins
		self.bins = np.unique(self.binNum)

		# Make cubes to hold stacked data
		self.stacked_spec = np.zeros((len(self.bins), len(self.goodwvl)))
		self.stacked_errs = np.zeros((len(self.bins), len(self.goodwvl)))

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
			for i in range(len(xarray)):
				binned_spec.append(self.data[self.goodwvl,yarray[i],xarray[i]])
				binned_var.append(self.var[self.goodwvl,yarray[i],xarray[i]])

				# Also record the bin ID for each pixel
				self.binIDarray[yarray[i], xarray[i]] = binID

			# Compute covariance correction
			if len(xarray) >= self.threshold:
				correction = self.norm * (1 + self.alpha * np.log(self.threshold))
			else:
				correction = self.norm * (1 + self.alpha * np.log(len(xarray)))

			self.stacked_spec[binID] = np.ma.mean(binned_spec, axis=0) #np.ma.average(binned_spec, axis=0, weights=1/np.power(binned_var,2))
			self.stacked_errs[binID] = np.ma.mean(binned_var, axis=0).filled(1e-5) * correction**2. / len(xarray) #np.sqrt(1./np.ma.sum(1./np.asarray(binned_var), axis=0))

		# For testing purposes, save arrays of bin IDs and bin errors
		np.save('output/'+self.galaxyname+'/binIDarray', self.binIDarray)
		np.save('output/'+self.galaxyname+'/binerrs', self.stacked_errs)

		print('Nbins:', len(self.bins))

		return

	def prepstellarfit(self, spectrum=None, wvl=None):
		""" Prepare stacked cube for spectral fitting with pPXF. Only need to run this once per galaxy!

			Arguments:
				spectrum, wvl (1D arrays): observed spectrum to use as template (default: use stacked spectrum)
		"""

		print('Preparing for stellar kinematics fit...')

		# Define spectrum
		spectrum = self.stacked_spec[0] if spectrum is None else spectrum
		wvl = self.wvl_cropped if wvl is None else wvl

		# Define wavelength range
		self.lamRange1 = [wvl[0],wvl[-1]]
		fwhm_gal = 2.4/(1+self.z)  # KCWI instrumental FWHM of ~2.4A

		# Rebin spectrum into log scale to get initial velocity scale
		_, _, velscale = util.log_rebin(self.lamRange1, spectrum)

		# Read the list of filenames from the E-MILES SSP library
		#vazdekis = glob.glob('/oak/stanford/orgs/kipac/users/mdlreyes/data/miles_models/Mku*.fits')
		vazdekis = glob.glob('/oak/stanford/orgs/kipac/users/mdlreyes/data/miles_stellar/s*.fits')
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

		sspNew, _ = util.log_rebin(lamRange_temp, ssp[good_lam], velscale=velscale)[:2]
		self.templates = np.empty((sspNew.size, len(vazdekis)))

		# Convolve observed spectrum with quadratic difference between observed and template resolution.
		# (This is valid if shapes of instrumental spectral profiles are well approximated by Gaussians.)
		fwhm_dif = np.sqrt(np.abs(fwhm_gal**2 - fwhm_tem**2))
		self.sigma = fwhm_dif/2.355/h2['CDELT1'] # Sigma difference in pixels
		galspec = ndimage.gaussian_filter1d(spectrum, self.sigma)

		# Now logarithmically rebin this new observed spectrum
		_, _, velscale = util.log_rebin(self.lamRange1, galspec, velscale=velscale)

		# Open and normalize all the templates
		for j, file in enumerate(vazdekis):
			hdu = fits.open(file)
			ssp = np.squeeze(hdu[0].data)
			sspNew, self.logLam2, _ = util.log_rebin(lamRange_temp, ssp[good_lam], velscale=velscale)
			self.templates[:, j] = sspNew/np.median(sspNew)  # Normalizes templates

		return

	def stellarkinematics(self, snr_mask=1, Niter=100):
		""" Do stellar kinematics fitting with pPXF. Note: must run prepstellarfit() first!

			Arguments:
				snr_mask (float): produce velmask.out file marking any bins where S/N > snr_mask
				Niter (int): number of iterations
				processes (int): number of processors to use
		"""

		# Prep the stellar fit
		self.prepstellarfit()

		# Create data structures to hold outputs over all iterations
		vel = np.zeros((len(self.bins), Niter))
		veldisp = np.zeros((len(self.bins), Niter))

		# Create data structures to hold final outputs
		self.velmask =np.zeros(len(self.bins), dtype=bool)

		print('Doing stellar kinematics fit...')

		# If computing global v/sigma, prep the outputs
		systvel = np.zeros(Niter)
		vmax = np.zeros(Niter)

		rng = default_rng()

		# Instantiate pool
		#p = multiprocessing.Pool(processes=4) #int(os.environ['SLURM_JOB_CPUS_PER_NODE']))

		newsize = (Niter, self.stacked_errs.shape[0], self.stacked_errs.shape[1])
		spec_new = np.broadcast_to(self.stacked_spec, newsize)
		errs_new = np.broadcast_to(np.sqrt(self.stacked_errs), newsize)
		self.stacked_spec_new = rng.normal(spec_new, errs_new)
		
		# Iterate in order to estimate MC errors
		for iteration in tqdm(range(Niter)):

			# Define function to parallelize
			func = functools.partial(binfit, stacked_spec_new=self.stacked_spec_new[iteration,:,:], 
									goodwvl_sn=self.goodwvl_sn, templates=self.templates, sigma=self.sigma, 
									lamRange1=self.lamRange1, logLam2=self.logLam2, lamRange2=self.lamRange2)

			# Begin the parallelization
			#for result in p.imap(func, range(len(self.bins))):
			with concurrent.futures.ProcessPoolExecutor(max_workers=int(os.environ['SLURM_CPUS_PER_TASK'])) as executor:
				for result in executor.map(func, range(len(self.bins))):
				
					# Get output
					binID, params = result

					# Store kinematic params
					vel[binID,iteration] = params[0]
					veldisp[binID,iteration] = params[1]
					if iteration==0 and self.sn[binID] > snr_mask:
						self.velmask[binID] = 1

			print(iteration)
			
		# Compute median values to store in final outputs
		self.vel = np.nanmedian(vel, axis=1)
		self.veldisp = np.nanmedian(veldisp, axis=1)
		self.vel_err_lo = self.vel - np.nanpercentile(vel, 16, axis=1)
		self.vel_err_up = np.nanpercentile(vel, 84, axis=1) - self.vel
		self.veldisp_err_lo = self.veldisp - np.nanpercentile(veldisp, 16, axis=1)
		self.veldisp_err_up = np.nanpercentile(veldisp, 84, axis=1) - self.veldisp

		# Compute systemic velocity
		flux = np.nansum(self.stacked_spec[:,self.goodwvl_sn], axis=1)
		systvel = np.average(self.vel, weights=flux)
		print("systvel: {:.2f}".format(systvel))

		# Compute sigma
		sigma = np.sqrt(np.average(veldisp**2., axis=0, weights=flux))

		# Compute vmax
		goodvel = vel[self.velmask]-systvel
		vmax = np.max(np.abs(goodvel), axis=0)

		# Compute MC errors
		labels = ['sigma', 'vmax', 'vsigma']
		for i, variable in enumerate([sigma, vmax, vmax/sigma]):
			median = np.nanpercentile(variable, 50)
			lower_err = median - np.nanpercentile(variable, 16)
			upper_err = np.nanpercentile(variable, 84) - median
			print(labels[i]+": {:.2f}\t{:.2f}\t{:.2f}".format(median, lower_err, upper_err))

		# Correct for systemic velocity
		self.vel += -np.nanpercentile(systvel, 50)

		np.savetxt('output/'+self.galaxyname+'/velocity.out', self.vel)
		np.savetxt('output/'+self.galaxyname+'/veldisp.out', self.veldisp)
		np.savetxt('output/'+self.galaxyname+'/vel_err.out', np.maximum(self.vel_err_lo, self.vel_err_up))
		np.savetxt('output/'+self.galaxyname+'/veldisp_err.out', np.maximum(self.veldisp_err_lo, self.veldisp_err_up))
		np.savetxt('output/'+self.galaxyname+'/velmask.out', self.velmask)

		# Make sure pool is closed!
		#p.close()
		#p.join()

		return

def runredux(galaxyname, folder='/oak/stanford/orgs/kipac/users/mdlreyes/data/stackedcubes/', Niter=1000):
	""" Run full redux pipeline.

	Arguments:
		overwrite (bool): if 'True', overwrite existing data files
		binned (bool): if 'True', use binned spectra instead of individual spaxels
		makeplots (bool): if 'True', just run the steps required to make plots
			(note: only works if all steps have been run before!)
	"""

	# Open params
	param = params[galaxyname]

	# Open cube
	c = Cube(galaxyname, folder=folder, wcscorr=param['wcscorr'], z=param['z'], EBV=param['EBV'])

	# Bin spaxels by continuum S/N, accounting for covariance
	c.binspaxels(targetsn=param['targetsn'], params=param['covparams'])

	# Do continuum fitting to get stellar kinematics
	c.stellarkinematics(snr_mask=param['snr_mask'], Niter=Niter)

	return

def main():

	if len(sys.argv) != 3:
		sys.exit("not enough args")
	Niter = int(sys.argv[1])
	galaxynum = int(sys.argv[2])

	# List of all galaxies
	#galaxylist = ['reines65','281238','control801','1074435','1158932','821857','866934','2502521','control751','control775']
	#galaxylist = ['1142116','1876887','1126100','1280160','control757','control872']
	galaxylist = ['1782069','1063413','1228631','955106','1246626','825059','control842','1785212']
	#galaxylist = ['PiscesA','PiscesB','control658','1904061','1180506']

	# Run test galaxy
	runredux(galaxylist[galaxynum], Niter=Niter)

	return

if __name__ == "__main__":
	main()

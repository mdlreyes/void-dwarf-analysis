# kcwialign.py
# Script to align, stack, and do
# covariance corrections to KCWI cubes
#
# Created: 16 March 2021
######################################

#Backend for python3 on mahler
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import os
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from tqdm import tqdm
from scipy.optimize import curve_fit
from reproject import reproject_interp
from cwitools import utils
from cwitools.scripts import cwi_crop, cwi_measure_wcs, cwi_apply_wcs, cwi_coadd

# Do some formatting stuff with matplotlib
from matplotlib import rc
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, MaxNLocator
rc('font', family='serif')
rc('axes', labelsize=14) 
rc('xtick', labelsize=12)
rc('ytick', labelsize=12)
rc('xtick.major', size=10)
rc('ytick.major', size=10)
rc('legend', fontsize=12, frameon=False)
rc('text',usetex=True)
rc('xtick',direction='in')
rc('ytick',direction='in')

def makefake(filename):
    
    with fits.open(filename+'_vcubes.fits') as vhdu:
        
        # Draw random numbers with same shape as vhdu
        vals = np.random.normal(size=vhdu[0].data.shape)
        vhdu[0].data = vals**2.

        print('Writing to '+filename+'_vcubes.test.fits')
        vhdu.writeto(filename+'_vcubes.test.fits', overwrite=True)

    with fits.open(filename+'_icubes.fits') as ihdu:

        # Add new fake variances to unity intensity
        ihdu[0].data = 1. + vals

        print('Writing to '+filename+'_icubes.test.fits')
        ihdu.writeto(filename+'_icubes.test.fits', overwrite=True) 

    return

def stack(galaxyname, mode, radec, box, outfile=None, plot=True, fakes=True, cubed=False, listdir='', drizzle=0.7):
    """Aligns and stacks datacubes.

        Args:
            galaxyname (str): Base filename
            mode (str): Mode to do spatial correlation ('src_fit' or 'xcor')
            radec (float tuple): RA and Dec of fitting box for alignment
            box (int): Size of fitting box 
            outfile (str): Filename to save output stacked file 
                        (if None, assume same name as galaxyname)
            plot (bool): If 'True', plot aligned cubes to check them
            fakes (bool): If 'True', also run cropping/stacking on fake cubes
                        (cubes with unity flux, Gaussian error)
            cubed (bool): If 'True', also run cropping/stacking on *cubed.fits
            listdir (str): Folder where '.list' and '.wcs' files are stored
            drizzle (float): pixfrac parameter for drizzling
	"""

    # Make fake datacubes
    if fakes:

        #Parse cube list
        cdict = utils.parse_cubelist(listdir+galaxyname+'.list')

        #Load input files
        in_files = utils.find_files(cdict["ID_LIST"], cdict["DATA_DIRECTORY"], 'icubes.fits', depth=cdict["SEARCH_DEPTH"])
        print('Making fake versions of', in_files)

        # Make fake datacubes
        for infile in in_files:
            makefake(infile[:-12])

    # Crop white space around datacubes
    ctype=["icubes.fits", "mcubes.fits", "vcubes.fits", "ocubes.fits"]
    if fakes:
        ctype += ["icubes.test.fits", "vcubes.test.fits"]
    if cubed:
        ctype += ["icubed.fits", "mcubed.fits", "vcubed.fits", "ocubed.fits"]
    cwi_crop(
        listdir+galaxyname+".list",
        ctype=ctype,
        xcrop=(5, 28),
        ycrop=(15, 80)
    )

    # Measure the coordinate system to create a 'WCS correction table'
    print('Measuring WCS correction')
    cwi_measure_wcs(
        listdir+galaxyname+".list",
        ctype="icubes.c.fits",
        xymode=mode,
        radec=radec,
        box=5,
        zmode="none",
        plot=False
    )

    # Apply the new WCS table to the cropped data cubes
    print('Applying WCS correction')
    ctype=["icubes.c.fits", "mcubes.c.fits", "vcubes.c.fits"]
    cwi_apply_wcs(
        listdir+galaxyname+".wcs",
        ctypes=ctype
    )
    if fakes:
        ctype = ["icubes.test.c.fits", "vcubes.test.c.fits"]
        cwi_apply_wcs(
            listdir+galaxyname+".wcs",
            ctypes=ctype
        )
    if cubed:
        ctype = ["icubed.c.fits", "vcubed.c.fits"]
        cwi_apply_wcs(
            listdir+galaxyname+".wcs",
            ctypes=ctype
        )

    if plot:
        #Load input files
        in_files = utils.find_files(cdict["ID_LIST"], cdict["DATA_DIRECTORY"], 'icubes.c.wc.fits', depth=cdict["SEARCH_DEPTH"])
        int_fits = [fits.open(x) for x in in_files]

        # Make images
        plt.subplot(projection=WCS(int_fits[0][0].header),slices=('x', 'y', 50))
        ims = [plt.imshow(np.nansum(int_fits[0][0].data,axis=0))]
        for idx, intfit in enumerate(int_fits):
            if idx != 0:
                array, footprint = reproject_interp(intfit, int_fits[0][0].header)
                ims.append( plt.imshow(np.nansum(array,axis=0)) )
                ims[idx].set_visible(False)

        # Try plotting frames
        def toggle_images(event):
            'toggle the visible state of the images'

            if event.key != 't':
                return
            b = [im.get_visible() for im in ims]
            for idx, im in enumerate(ims):
                if b[idx]:
                    im.set_visible(False)
                    ims[(idx + 1) % len(b)].set_visible(True)
            plt.draw()

            return

        plt.connect('key_press_event', toggle_images)
        plt.show()

    if outfile is None:
        outfile = 'stackedcubes/'+galaxyname

    # Coadd the WCS-corrected data cubes
    print('Coadding datacubes')
    cwi_coadd(listdir+galaxyname+".list", ctype='icubes.c.wc.fits', masks='mcubes.c.wc.fits', var='vcubes.c.wc.fits', 
        pa=None, px_thresh=0.5, exp_thresh=0.1, verbose=False, drizzle=0.8, out=outfile+".fits")
    os.rename(outfile+".fits", outfile+"_icubes.fits")
    os.rename(outfile+".var.fits", outfile+'_vcubes.fits')

    # Coadd the WCS-corrected fake data cubes
    if fakes:
        print('Coadding fake datacubes')
        cwi_coadd(listdir+galaxyname+".list", ctype='icubes.test.c.wc.fits', masks='mcubes.c.wc.fits', var='vcubes.test.c.wc.fits', 
            pa=None, px_thresh=0.5, exp_thresh=0.1, verbose=False, drizzle=0.7, out=outfile+".test.fits")
        os.rename(outfile+".test.fits", outfile+"_test_icubes.fits")
        os.rename(outfile+".test.var.fits", outfile+"_test_vcubes.fits")

    # Coadd the WCS-corrected *cubed.fits files
    if cubed:
        print('Coadding *cubed.fits')
        cwi_coadd(listdir+galaxyname+".list", ctype='icubed.c.wc.fits', masks='mcubes.c.wc.fits', var='vcubed.c.wc.fits', 
            pa=None, px_thresh=0.5, exp_thresh=0.1, verbose=False, drizzle=0.7, out=outfile+".cubed.fits")
        os.rename(outfile+".cubed.fits", outfile+"_icubed.fits")
        os.rename(outfile+".cubed.var.fits", outfile+"_vcubed.fits")

    return

def getdata(filename='', maskfile=None, z=0., plot=True, maskout=True):
    """Get data from intensity and variance cubes
    
    Args:
        filename (str): Base file name of fits file
        maskfile (str): Full name of mask file
        z (float): Redshift (doesn't make a difference for synthetic cubes)
        plot (bool): If 'True', plot white light image
        maskout (bool): If 'True', output a mask based on where data and var are 0
    """

    # Main intensity cube
    icube = fits.open(filename+'_icubes.fits')
    data = icube[0].data
    wcs = WCS(icube[0].header)
    print(data.shape)

    # Error cube
    vcube = fits.open(filename+'_vcubes.fits')
    var = vcube[0].data

    # Mask cube
    makemask = False
    if maskfile is not None:
        mcube = fits.open(maskfile)
        mask = mcube[0].data
        if mask.shape != data.shape:
            makemask = True
    if (maskfile is None) or makemask:
        # Mask cube
        mask = np.zeros_like(data)
        badidx = np.where((np.isclose(data,0)) & (np.isclose(var,0)))
        mask[badidx] = True
        masked_data = np.ma.array(data, mask=mask)
        masked_var = np.ma.array(var, mask=mask)

        if maskout:
            with fits.open(filename+'_icubes.fits') as ihdu:
                # Write mask to FITS
                ihdu[0].data = mask
                print('Writing to '+filename+'_mcubes.fits')
                ihdu.writeto(filename+'_mcubes.fits', overwrite=True) 

    # Mask the data and error cubes
    masked_data = np.ma.array(data, mask=mask)
    masked_var = np.ma.array(var, mask=mask)

    if plot:
        fig = plt.figure(figsize=(8,8))
        ax = plt.subplot(projection=wcs,slices=('x', 'y', 50))
        plt.title('White light image')
        plt.imshow(np.nansum(data, axis=0), vmin=0)
        plt.colorbar()
        plt.savefig(filename+'.png')
        plt.show()

    # Get zeropoints and deltas for coordinates and wavelength
    ra0 = icube[0].header['CRVAL1']
    dec0 = icube[0].header['CRVAL2']
    wvl0 = icube[0].header['CRVAL3']

    rad = icube[0].header['CD1_1'] # RA degrees per col
    decd = icube[0].header['CD2_2'] # Dec degrees per row
    wvld = icube[0].header['CD3_3'] # wvl Angstroms per pixel

    # Create wavelength array
    N = len(masked_data[:,0,0])
    wvl = np.arange(wvl0,wvl0+N*wvld,wvld)
    wvl_zcorr = wvl / (1.+z)

    return data, var, mask, wcs, wvl_zcorr

def covar_curve(ksizes, alpha, norm, thresh):
    """Two-component model to describe increase in noise due to covariance
    The model is divided into two regimes, based on the 'threshold' parameter:
    for ksizes <= threshold:
    noise / ideal_noise = norm * (1 + alpha * ln(ksizes))
    for ksizes > threshold:
    noise / ideal_noise = beta = norm * (1 + alpha * ln(threshold))

    Args:
        ksizes (np.array): Array of 2D kernel or bin sizes (i.e. areas)
        alpha, norm, threshold (float): Parameters of function
    Returns:
        factor (np.array): The ratio of true noise to 'ideal' noise
    """
    res = norm * (1 + alpha * np.log(ksizes))
    res[ksizes > thresh] = norm * (1 + alpha * np.log(thresh))
    return res

def estimatecovar(galaxyname, folder='stackedcubes/old/', maskfile=None, plot=True, n_w=20, bin_grid=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]):
    """Estimate covariance from test datacubes.

        Args:
            galaxyname (str): Name of galaxy to estimate covariance for
            folder (str): Folder where data are located
            maskfile (str): Full name of mask cube
            plot (bool): If 'True', plot aligned cubes to check them
            n_w (int): Number of independent wavelength slices to try
            bin_grid (int list): List of bin sizes to try
    """

    # Open files
    data, var, mask, wcs, wvl_zcorr = getdata(folder+galaxyname+'_test', maskfile=maskfile, plot=False)

    # Apply mask
    var[mask != 0] = 0
    data[mask != 0] = 0

    # Filter data for bad values
    data = np.nan_to_num(data, nan=0, posinf=0, neginf=0)
    var = np.nan_to_num(var, nan=0, posinf=0, neginf=0)

    # Bin spaxels into boxcar of size N^2
    def resize(cube_in, var_in, mask_in, binsize):

        cube = cube_in.copy()
        mask = mask_in.copy()
        var = var_in.copy()

        if binsize == 1:
            return cube.reshape((cube.shape[0],cube.shape[1]*cube.shape[2])), \
                    var.reshape((var.shape[0],var.shape[1]*var.shape[2])), \
                    mask.reshape((mask.shape[0],mask.shape[1]*mask.shape[2]))

        binsize = int(binsize)

        # Adjust size of cube to be even multiple of binsize
        shape = cube.shape
        if shape[1] % binsize != 0:
            cube = cube[:, 0:(shape[1] - shape[1] % binsize), :]
            mask = mask[:, 0:(shape[1] - shape[1] % binsize), :]
            var = var[:, 0:(shape[1] - shape[1] % binsize), :]
        if shape[2] % binsize != 0:
            cube = cube[:, :, 0:(shape[2] - shape[2] % binsize)]
            mask = mask[:, :, 0:(shape[2] - shape[2] % binsize)]
            var = var[:, :, 0:(shape[2] - shape[2] % binsize)]

        # Update shape
        shape = cube.shape

        # Create new shape for the purpose of rebinning
        shape_new = (
            shape[0],
            int(shape[1] / binsize),
            binsize,
            int(shape[2]/binsize),
            binsize
            )
        cube_reshape = cube.reshape(shape_new)
        var_reshape = var.reshape(shape_new)
        mask_reshape = mask.reshape(shape_new)

        # If a bin contains mask == 1, set whole binned mask voxel to 1
        for k in range(shape_new[0]):
            for i in range(shape_new[1]):
                for j in range(shape_new[3]):
                    msk_bin = mask_reshape[k, i, :, j, :]
                    if 1 in msk_bin:
                        mask_reshape[k, i, :, j, :] = 1

        #Recover final, binned versions
        cube_binned = cube_reshape.sum(axis=(-1,2))
        var_binned = var_reshape.sum(axis=(-1,2))
        mask_binned = mask_reshape.max(-1).max(2)

        cube_binned = cube_binned.reshape((cube_binned.shape[0],cube_binned.shape[1]*cube_binned.shape[2]))
        var_binned = var_binned.reshape((var_binned.shape[0],var_binned.shape[1]*var_binned.shape[2]))
        mask_binned = mask_binned.reshape((mask_binned.shape[0],mask_binned.shape[1]*mask_binned.shape[2]))

        return cube_binned, var_binned, mask_binned

    # Calculate noise ratio as a function of spatial bin size
    bin_sizes = []
    noise_ratios = []
    bin_grid = np.array(bin_grid)
    
    # Get indices of a set of 'n_w' evenly-spaced wavelength layers throughout cube
    # This is done to extract a sub-cube made up of independent z-layers
    z_indices = np.arange(1, data.shape[0] - n_w, n_w).astype(int)

    # 'z_shift' shifts these indices along by 1 each time, selecting a different
    # sub-cube made up of independent z-layers
    for z_shift in tqdm(range(n_w-1)):
        for bin_i in np.flip(bin_grid):

            #Extract sub cube and sub mask
            subcube = data[z_indices + z_shift, :, :].copy()
            submask = mask[z_indices + z_shift, :, :].copy()
            subvar = var[z_indices + z_shift, :, :].copy()

            #Bin
            cube_b, var_b, mask_b = resize(subcube, subvar, submask, bin_i)

            #Get binary mask of useable vox (mask is dtype int)
            use_vox = (mask_b == 0)

            #Skip if fewer than 10 useable voxels remain in binned cube
            if np.count_nonzero(use_vox) < 10:
                continue

            # Measure the error in the binned cube
            actual_err = np.std(cube_b[use_vox])
            propagated_err = np.sqrt(np.mean(var_b[use_vox]))

            # Append bin size and noise ratio to lists
            if np.isfinite(actual_err / propagated_err):
                bin_sizes.append(bin_i)
                noise_ratios.append(actual_err / propagated_err)

    # Get arrays for plotting
    bin_sizes = np.array(bin_sizes)
    kernel_areas = bin_sizes**2.
    noise_ratios = np.array(noise_ratios)

    # Fit model curve to results
    popt, pcov = curve_fit(covar_curve, kernel_areas, noise_ratios, bounds=([0.1, 1., 30.], [10., 2., 150.]))

    # Plot results
    plt.plot(kernel_areas, noise_ratios, 'ko')
    kareas_smooth = np.linspace(kernel_areas.min(), kernel_areas.max(), 1000)
    plt.plot(kareas_smooth, covar_curve(kareas_smooth, *popt), 'r-',
        label=r'Fit: $\alpha$=%5.3f, $\beta$=%5.3f, $N_{\mathrm{threshold}}$=%5.3f' % tuple(popt))
    plt.xlabel(r'Bin size $N$', fontsize=16)
    plt.ylabel(r'$\eta$', fontsize=16)
    plt.legend(fontsize=14)
    plt.savefig('figures/'+galaxyname+'/covartest.pdf', bbox_inches='tight')
    if plot:
        plt.show()
    else:
        plt.close()
    
    print('Results:', popt)

    return popt

if __name__ == "__main__":

    stack('reines65', 'xcor', radec=(174.1787932, 26.72628063), box=5, listdir='lists/', cubed=False)
    #getdata('stackedcubes/reines65', plot=True, maskout=True)  # Make sure stacking worked, make final mask
    estimatecovar('reines65', plot=True)

# getmasses.py
# Script to find stellar masses from SDSS+WISE MAGPHYS catalog
# (https://irfu.cea.fr/Pisp/yu-yen.chang/sw.html)
#
# Created: 12 Sept 2022
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

# Import packages
import numpy as np
from astropy.io import ascii
from astropy.coordinates import SkyCoord, Distance
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM 
cosmo = FlatLambdaCDM(H0=67.4, Om0=0.315)  # using Planck (2018) params
from numpy.random import default_rng

def matchcatalog():
    """Try to find stellar masses in other catalogs"""
    # Read in datafiles
    kcwi_data = ascii.read('../data/sample_FINAL_new.csv').filled(-999.0)
    sdss_input = ascii.read('../data/sdss_wise_masses_input.txt', format='cds')
    sdss_output = ascii.read('../data/sdss_wise_masses_output.txt', format='cds')
    gswlc = ascii.read('../data/gswlc_masses.csv')

    # Get catalog matches
    c_kcwi = SkyCoord(ra=kcwi_data['RA'], dec=kcwi_data['Dec'], unit=(u.hourangle, u.deg))  # coords of KCWI data
    c_sdss = SkyCoord(ra=sdss_input['RAdeg'], dec=sdss_input['DEdeg'])  # coords of SDSS+WISE data
    c_gswl = SkyCoord(ra=gswlc['ra']*u.deg, dec=gswlc['decl']*u.deg)

    #for i in range(len(c_kcwi)):
    #    print(kcwi_data['ID'][i], c_kcwi[i].to_string('decimal'))

    # Set max separation
    max_sep = 2.0 * u.arcsec

    # Match SDSS+WISE catalog
    idx, sep, _ = c_kcwi.match_to_catalog_sky(c_sdss)
    sep_constraint = sep < max_sep

    kcwi_matches_id = kcwi_data['ID'][sep_constraint]
    sdss_matches_id = sdss_input['ID'][idx[sep_constraint]]

    print('SDSS matches')
    for i in range(len(kcwi_matches_id)):
        print(kcwi_matches_id[i], sdss_matches_id[i], kcwi_data['massSDSS'][sep_constraint][i], sdss_output['lmass50'][idx[sep_constraint]][i], sdss_output['lmass16'][idx[sep_constraint]][i], sdss_output['lmass84'][idx[sep_constraint]][i])

    # Match GSWLC catalog
    idx, sep, _ = c_kcwi.match_to_catalog_sky(c_gswl)
    sep_constraint = sep < max_sep

    kcwi_matches_id = kcwi_data['ID'][sep_constraint]
    gswl_matches_id = gswlc['objid'][idx[sep_constraint]]

    print('GSWLC matches')
    for i in range(len(kcwi_matches_id)):
        print(kcwi_matches_id[i], gswl_matches_id[i], kcwi_data['massSDSS'][sep_constraint][i], c_gswl['logmstar'][idx[sep_constraint]][i])

    return

def computeWISEmasses():
    """Compute stellar masses from WISE W1 and W2 bands"""

    # Get data
    kcwi_data = ascii.read('../data/sample_FINAL_new.csv').filled(-999.0)

    print('WISE masses')
    rng = default_rng()
    for i in range(len(kcwi_data)):

        if kcwi_data['w1mpro'][i] > -990:
            # Get luminosity distance
            dist = Distance(z=kcwi_data['z'][i], cosmology=cosmo, unit=u.pc)

            Niter = 10000
            Mstar = np.zeros(Niter)
            for iter in range(Niter):
                # Perturb magnitudes
                w1m = rng.normal(kcwi_data['w1mpro'][i], kcwi_data['w1sigmpro'][i])
                w2m = rng.normal(kcwi_data['w2mpro'][i], kcwi_data['w2sigmpro'][i])

                # Compute absolute magnitude of W1 band
                abs_W1 = w1m - 5.*np.log10(float(dist.value)) + 5

                # Using equation 1 from Cluver+2014 (GAMA)
                Msun = 3.24
                L_W1 = np.power(10., -0.4*(abs_W1-Msun))
                log10ML = -2.54*(w1m-w2m) - 0.17

                Mstar[iter] = np.log10(L_W1*np.power(10., log10ML))

            # Propagate errors
            median = np.percentile(Mstar, 50)
            lowerr = median - np.percentile(Mstar, 16)
            uperr = np.percentile(Mstar, 84) - median
            print(median, lowerr, uperr)

    return

def comparemasses():
    """Plot stellar mass estimates against each other."""

    # Get data
    kcwi_data = ascii.read('../data/sample_FINAL_new.csv').filled(-999.0)

    # Create figure
    plt.figure(figsize=(7,7))

    # Open WISE data
    x_void = kcwi_data['massSDSS']
    wise_void = kcwi_data['massWISE']
    wise_goodidx = np.where((x_void > -990.) & (wise_void > -990.))
    x_void_err = [np.asarray(kcwi_data['massSDSS'] - kcwi_data['massSDSS_lo'])[wise_goodidx], np.asarray(kcwi_data['massSDSS_hi'] - kcwi_data['massSDSS'])[wise_goodidx]]
    wise_void_err = [kcwi_data['massWISE_lo'][wise_goodidx], kcwi_data['massWISE_hi'][wise_goodidx]]

    # Now do the SDSS+WISE data
    sdsswise_void = kcwi_data['massSDSSWISE']
    sdsswise_goodidx = np.where((x_void > -990.) & (sdsswise_void > -990.))
    xwise_void_err = [np.asarray(kcwi_data['massSDSS'] - kcwi_data['massSDSS_lo'])[sdsswise_goodidx], np.asarray(kcwi_data['massSDSS_hi'] - kcwi_data['massSDSS'])[sdsswise_goodidx]]
    sdsswise_void_err = [np.asarray(kcwi_data['massSDSSWISE'] - kcwi_data['massSDSSWISE_lo'])[sdsswise_goodidx], np.asarray(kcwi_data['massSDSSWISE_hi'] - kcwi_data['massSDSSWISE'])[sdsswise_goodidx]]

    plt.errorbar(x_void[wise_goodidx], wise_void[wise_goodidx], xerr=x_void_err, yerr=wise_void_err, mfc='C0', mec='C0', ecolor='C0', linestyle='None', marker='o', markersize=6, linewidth=1, label='WISE calibration masses')
    plt.errorbar(x_void[sdsswise_goodidx], sdsswise_void[sdsswise_goodidx], xerr=xwise_void_err, yerr=sdsswise_void_err, mfc='C1', mec='C1', ecolor='C1', linestyle='None', marker='o', markersize=6, linewidth=1, label='SDSS+WISE MAGPHYS masses')

    # Plot formatting
    minval = 6.5
    maxval = 9.75    
    plt.plot([minval, maxval],[minval,maxval],'r--')
    plt.xlim(minval,maxval)
    plt.ylim(minval,maxval)
    plt.xlabel(r'$\log_{10}M_{\star,\mathrm{SDSS}}~[M_{\odot}]$', fontsize=16)
    plt.ylabel(r'$\log_{10}M_{\star,\mathrm{comparison}}~[M_{\odot}]$', fontsize=16)
    plt.legend(fontsize=12)
    plt.savefig('plots/masscomparison.png', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    #matchcatalog()
    #computeWISEmasses()
    comparemasses()
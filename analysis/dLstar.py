# dLstar.py
# Script to compute distances to other galaxies
#
# Created: 13 April 2022
######################################

import numpy as np
import matplotlib.pyplot as plt

# Import astropy modules
from astropy.io import ascii  # only needed for SFH test stuff
from astropy.cosmology import FlatLambdaCDM  # needed to compute redshifts
cosmo = FlatLambdaCDM(H0=67.4, Om0=0.315)  # using Planck (2018) params
import astropy.units as u
from astropy.coordinates import SkyCoord,Distance
from calc_kcor import calc_kcor

def coords_to_deg(coord):
    c1 = SkyCoord(coord, unit=(u.hourangle, u.deg))
    print(str(c1.ra.degree)+','+str(c1.dec.degree))
    return (c1.ra.degree,c1.dec.degree)

def dist(c1,z1,c2,z2):
    d1 = Distance(z=z1, cosmology=cosmo)
    c1 = SkyCoord(c1[0]*u.deg, c1[1]*u.deg, frame='icrs', distance=d1)

    d2 = Distance(z=z2, cosmology=cosmo)
    #print(d2)
    #print(c2)
    c2 = SkyCoord(c2[0]*u.deg, c2[1]*u.deg, frame='icrs', distance=d2)

    sep = c1.separation_3d(c2)
    #print('Separation', sep.to(u.kpc))
    return sep.to(u.kpc)

def dist_onsky(c1,c2,z):
    c1 = SkyCoord(c1[0]*u.deg, c1[1]*u.deg, frame='icrs')
    c2 = SkyCoord(c2[0]*u.deg, c2[1]*u.deg, frame='icrs')

    theta = c1.separation(c2).arcsecond * u.arcsec
    d_A = cosmo.angular_diameter_distance(z=z)
    onskydist = (theta*d_A).to(u.kpc, u.dimensionless_angles())

    return onskydist

def computemag(mag,z2):
    d2 = Distance(z=z2, cosmology=cosmo)
    absmag = mag - 5*np.log10(d2.to(u.kpc)/(1.*u.kpc))
    print(absmag,'mag')

    L = 10.**(0.4*(5.23-absmag))
    print(np.log10(L))
    return

def computeWISEmass(w1m,w2m,z):
    # Get luminosity distance
    dist = Distance(z=z, cosmology=cosmo, unit=u.pc)

    # Compute absolute magnitude of W1 band
    abs_W1 = w1m - 5.*np.log10(float(dist.value)) + 5

    # Using equation 1 from Cluver+2014 (GAMA)
    Msun = 3.24
    L_W1 = np.power(10., -0.4*(abs_W1-Msun))
    log10ML = -2.54*(w1m-w2m) - 0.17

    logMstar = np.log10(L_W1*np.power(10., log10ML))

    return logMstar

def computeBellmass(g,r,z):
    # Get luminosity distance
    dist = Distance(z=z, cosmology=cosmo, unit=u.Mpc)

    # Compute Bell+03 stellar mass
    gr = g - r
    Mr = r - (np.log10(float(dist.value)) * 5.0 + 25.0) - calc_kcor("r", z, "g - r", gr) # Compute absolute magnitude of r band
    logMstar = 1.254 + 1.0976 * gr - 0.4 * Mr

    return logMstar

if __name__=="__main__":
    coord = '02 26 28.29	+01 09 37.92'
    c1 = coords_to_deg(coord)
    z1 = 0.00513761

    sqlfile = ascii.read('sql.csv')
    dist2d = np.zeros(len(sqlfile['specObjID']))
    dist3d = np.zeros(len(sqlfile['specObjID']))
    for i in range(len(sqlfile['specObjID'])):
        c2 = (sqlfile['ra'][i], sqlfile['dec'][i])
        z2 = sqlfile['redshift'][i]
        dist2d[i] = dist_onsky(c1,c2,z1).value
        dist3d[i] = dist(c1,z1,c2,z2).value

    # Find first minimum
    minidx = np.argmin(dist3d)
    print(sqlfile['ra'][minidx],sqlfile['dec'][minidx],sqlfile['redshift'][minidx],sqlfile['distance'][minidx],dist3d[minidx])

    # Test
    #np.set_printoptions(suppress=True)
    #print(dist3d)

    # Correct the original coordinates, then look for nearest neighbor
    c1 = (sqlfile['ra'][minidx],sqlfile['dec'][minidx])
    z1 = sqlfile['redshift'][minidx]
    dist2d_Lstar = np.zeros(len(sqlfile['specObjID']))
    dist2d_Lstar_wise = np.zeros(len(sqlfile['specObjID']))
    dist3d_Lstar = np.zeros(len(sqlfile['specObjID']))
    dist3d_Lstar_wise = np.zeros(len(sqlfile['specObjID']))
    dist3d = np.zeros(len(sqlfile['specObjID']))

    #mass_wise = np.zeros(len(sqlfile['specObjID']))
    #mass_bell = np.zeros(len(sqlfile['specObjID']))
    for i in range(len(sqlfile['specObjID'])):
        c2 = (sqlfile['ra'][i], sqlfile['dec'][i])
        z2 = sqlfile['redshift'][i]
        dist3d[i] = dist(c1,z1,c2,z2).value  # Nearest neighbor
        if computeWISEmass(sqlfile['w1_mag'][i], sqlfile['w2_mag'][i], z2) > 10.:
            dist2d_Lstar_wise[i] = dist_onsky(c1,c2,z1).value  # Nearest massive neighbor (from WISE)
            dist3d_Lstar_wise[i] = dist(c1,z1,c2,z2).value  # Nearest massive neighbor (from WISE)
        if computeBellmass(sqlfile['g'][i], sqlfile['r'][i], z2) > 10.:
            dist2d_Lstar[i] = dist_onsky(c1,c2,z1).value  # Nearest massive neighbor (from WISE)
            dist3d_Lstar[i] = dist(c1,z1,c2,z2).value  # Nearest massive neighbor (from Bell+03)

        #mass_wise[i] = computeWISEmass(sqlfile['w1_mag'][i], sqlfile['w2_mag'][i], z2)
        #mass_bell[i] = computeBellmass(sqlfile['g'][i], sqlfile['r'][i], z2)
    print('dnearest', np.min(dist3d[dist3d > 0.]))
    print('dLstar', np.min(dist3d_Lstar[dist3d_Lstar > 0.]))
    print('dLstar_wise', np.min(dist3d_Lstar_wise[dist3d_Lstar_wise > 0.]))
    print('proj_dLstar', np.min(dist2d_Lstar[dist2d_Lstar > 0.]))
    print('proj_dLstar_wise', np.min(dist2d_Lstar_wise[dist2d_Lstar_wise > 0.]))

    '''
    masses = np.concatenate((mass_wise, mass_bell))
    minmass = np.nanmin(masses[np.isfinite(masses)])
    maxmass = np.nanmax(masses[np.isfinite(masses)])
    minmass = 0
    maxmass = 25
    print(minmass, maxmass)
    plt.plot(mass_wise, mass_bell, 'ko')
    plt.plot([minmass,maxmass],[minmass,maxmass],'r--')
    plt.xlim(minmass,maxmass)
    plt.ylim(minmass,maxmass)
    plt.xlabel('WISE')
    plt.ylabel('Bell')
    plt.show()
    '''

    #computemag(15.65,z2)
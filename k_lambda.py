#!/usr/bin/env python

# Modification by RLT: added Reddy et al curve

#
# NAME:
#   K_LAMBDA()
#
# PURPOSE:
#   Return a variety of extinction curves.
#
# INPUTS:
#   wave  - wavelength vector [Angstroms]
#
# OPTIONAL INPUTS:
#   r_v   - total to selective extinction ratio (default 3.1, but
#           poorly known for the LMC and SMC, and inappropriate
#           for the Calzetti attenuation curve)# also hard-wired
#           to be 3.1 for Li & Draine (2001) since it was only
#           calibrated for the diffuse MW ISM
#
# KEYWORD PARAMETERS:
#   calzetti - Calzetti (2000) continuum attenuation curve for IUE
#              starburst galaxies
#   charlot  - Charlot & Fall (2000) attenuation curve
#   ccm      - Cardelli, Clayton, & Mathis (1989) [Milky Way]
#   odonnell - CCM with O'Donnell (1994) coefficients [Milky Way]
#   fm       - Fitzpatrick (1999) [Milky Way]
#   avglmc   - average Large Magellanic Cloud from Gordon et
#              al. 2003, ApJ, 594, 279# R_V = 3.41+/-0.06
#   lmc2     - Large Magellanic Cloud Supershell (LMC2) field,
#              including 30 Dor from Gordon et al. 2003, ApJ, 594,
#              279# R_V = 2.76+/-0.09
#   smc      - Small Magellanic Cloud Bar region from Gordon et
#              al. 2003, ApJ, 594, 279# R_V = 2.74+/-0.13
#   li       - Li & Draine (2001) Milky Way extinction curve
#   silent   - suppress warning messages
#   reddy    - Reddy et al. 2015+2016 curve **added by RLT**
# OUTPUTS:
#   k_lambda - defined as A(lambda)/E(B-V)
#
# OPTIONAL OUTPUTS:
#
# COMMENTS:
#   This is literally just copied from John Moustakas' k_lambda.pro
#
# EXAMPLE:
#   Plot the Calzetti and the CCM extinction curves in the
#   optical and UV.
#
#   IDL> wave = findgen(6000)+1000.0 # Angstrom
#   IDL> plot, wave, k_lambda(wave,/calzetti), xsty=3, ysty=3
#   IDL> oplot, wave, k_lambda(wave,/ccm)
#
# MODIFICATION HISTORY:
#   J. Moustakas, 2002 September 19, U of A
#   jm03mar2uofa - added the Charlot & Fall (2000) curve
#   jm03sep2uofa - general updates
#   jm03sep28uofa - added Seaton (1979) extinction curve
#   jm04jan26uofa - updated SMC, LMC2, and AVGLMC curves from
#                   Gordon et al. (2003)
#   jm04apr26uofa - the Charlot & Fall R_V=5.9, not 3.1 
#   jm06feb24uofa - added SILENT keyword
#   jm07feb19nyu  - added Li & Draine (2001) extinction curve 
#   jm09aug19ucsd - ensure that the Calzetti, O'Donnell and
#     SMC extinction curves behave properly at very long and short 
#     wavelengths  
#   jm11aug05ucsd - made READ_GORDON_2003() an internal support
#     routine 
#   R. Trainor - transcribed into python
#
# Copyright (C) 2002-2004, 2006-2007, 2009, 2011, John Moustakas
# Copyright (C) 2016, Ryan Trainor
# 
# This program is free software# you can redistribute it and/or modify 
# it under the terms of the GNU General Public License as published by 
# the Free Software Foundation# either version 2 of the License, or
# (at your option) any later version. 
# 
# This program is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY# without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details. 
#

import os
import numpy as np
import scipy.interpolate as interpolate
from numpy.polynomial.polynomial import polyval
#from readcol import readcol
from astropy.io import ascii

#ROOT='/Users/trainor/Astro/python/dust/'

def vac2air(wave):
    '''Converts vacuum wavelengths to air via the IAU standard conversion
    from Morton (1991, ApJS, 77, 119).
    '''
    return wave / (1.0 + 2.735182E-4 + 131.4182 / wave**2 + 2.76249E8 / wave**4)

    

def read_gordon_2003(wave=None, dlaw='smc',r_v=None):
    ''' internal support routine for K_LAMBDA()
    dlaw=['smc','lmc2','avglmc']
    
    jm04jan26uofa - written
    jm09aug20ucsd - force k(lambda) to be positive at long wavelengths
    rt16may13ucb  - translate to python
    '''

    #path = os.environ['PYTHONPATH']+'/dust/etc/'
    path = '/Users/rtheios/python/dust/etc/'
    file = '2003_gordon.dat'

    data = np.genfromtxt(path+file)
    l = data[:,0]; x = data[:,1]; Al_AV_smc = data[:,2]; err = data[:,3]; Al_AV_lmc2 = data[:,4]; err = data[:,5]; Al_AV_avglmc = data[:,6]; err = data[:,7]

    #l,x,Al_AV_smc,err,Al_AV_lmc2,err,Al_Av_avglmc,err=ascii.read(path+file)

    R_V_smc    = 2.74
    R_V_lmc2   = 2.76
    R_V_avglmc = 3.41

    if dlaw=='smc':
        if not r_v: r_v = R_V_smc
        good = Al_AV_smc != -9.999
        Al_AV = Al_AV_smc[good]
        l = l[good]
    if dlaw=='lmc2':
        if not r_v: r_v = R_V_lmc2
        good = Al_AV_lmc2 != -9.999
        Al_AV = Al_AV_lmc2[good]
        l = l[good]
    if dlaw=='avglmc':
        if not r_v: r_v = R_V_avglmc
        good = Al_AV_avglmc != -9.999
        Al_AV = Al_AV_avglmc[good]
        l = l[good]
    
    lam = 1e4*l
    k_lambda = r_v*Al_AV

    if wave != None:
        idx=np.argsort(lam)
        spline=interpolate.splrep(lam[idx],k_lambda[idx],k=3)
        k_lambda=interpolate.splev(wave,spline)
    else: wave = lam
    
    return k_lambda
 
def k_lambda(wave, dlaw='odonnell', r_v=None, silent=False, air=False):
    '''Returns a variety of extinction curves. Adapted by Ryan Trainor from
    John Moustakas' IDL program k_lambda.pro.

    Wavelengths are assumed to be Angstroms. If air=True, input is assumed
    to be wavelengths in air, otherwise vacuum wavelengths are converted to
    air via the IAU standard conversion from Morton (1991, ApJS, 77, 119)
    because I assumed k_lambda.pro used air wavelengths as input. Please
    email trainor@berkeley.edu if you know this to be incorrect.

    Allowed extinction laws are set by dlaw keyword (default is 'odonnell'):
    'calzetti'  - Calzetti (2000) continuum attenuation curve for IUE
                  starburst galaxies
    'charlot'   - Charlot & Fall (2000) attenuation curve
    'ccm'       - Cardelli, Clayton, & Mathis (1989) [Milky Way]
    'odonnell'  - CCM with O'Donnell (1994) coefficients [Milky Way]
    'fm_MW'     - Fitzpatrick (1999) [Milky Way]
    'fm_lmc2'   - Fitzpatrick (1999) [LMC2]
    'fm_avglmc' - Fitzpatrick (1999) [LMC average]
    'avglmc'    - average Large Magellanic Cloud from Gordon et
                  al. 2003, ApJ, 594, 279# R_V = 3.41+/-0.06
    'lmc2'      - Large Magellanic Cloud Supershell (LMC2) field,
                  including 30 Dor from Gordon et al. 2003, ApJ, 594,
                  279# R_V = 2.76+/-0.09
    'smc'       - Small Magellanic Cloud Bar region from Gordon et
                  al. 2003, ApJ, 594, 279# R_V = 2.74+/-0.13
    'li'        - Li & Draine (2001) Milky Way extinction curve
    'reddy'     - Reddy et al. (2015+2016)
    '''
    k_lambda = -1.0

    if hasattr(wave, "__len__"):
        wave = np.asarray(wave)
    else:
        wave = np.asarray([wave])

    if not air:
        wave = vac2air(wave)
            
    zero = wave <= 0.0
    if zero.sum() !=0:
        print('WAVE contains zero or negative values!')
        return k_lambda
    
    # ---------------------------------------------------------------------------    
    # Calzetti et al. (2000)
    if dlaw=='calzetti':
        if not r_v: r_v = 4.05
        w1 = (wave >= 6300)&(wave <= 22000)
        c1 = w1.sum()
        #w2 = (wave >=  912)&(wave <  6300)
        w2 = (wave<6300)&(wave>=1500)
        c2 = w2.sum()
        w3 = (wave<1500)&(wave>=500)
        c3 = w3.sum()
        x  = 10000.0/wave        # wavelength in inverse microns

        k_lambda = 0.0*wave

        if c1 > 0:
            k_lambda[w1] = 2.659*(-1.857 + 1.040*x[w1])+r_v
        if c2 > 0:
            k_lambda[w2] = 2.659*(polyval(x[w2], [-2.156, 1.509e0, -0.198e0, 0.011e0])) + r_v
        if c3 > 0:
            k_lambda[w3] = 4.126+0.931*x[w3]
            
    # if necessary, extrapolate to longer and shorter wavelengths
        inrange = (wave >= 912.0) & (wave <= 22000.0)
        ninrange = inrange.sum()
        lowave = wave < 500.0
        nlowave = lowave.sum()
        hiwave = wave > 22000.0
        nhiwave = hiwave.sum()
        if (nlowave != 0) | (nhiwave != 0):
            if not silent: print('Extrapolating the Calzetti k(lambda)')
        if (nlowave != 0): k_lambda[lowave] = k_lambda[inrange[0]]
        if (nhiwave != 0):
            k_lambda=np.minimum(np.interp(wave[hiwave],wave[inrange],k_lambda[inrange]),0)

    # ---------------------------------------------------------------------------    
    # Charlot & Fall (2000)
    if dlaw=='charlot':
        if not r_v: r_v = 5.9 # S. Charlot, private communication
        k_lambda = r_v*(wave/5500.0)**(-0.7)

    # ---------------------------------------------------------------------------    
    # Cardelli, Clayton, & Mathis (1989)
    if (dlaw=='ccm')|(dlaw=='odonnell'):
        if not r_v: r_v = 3.1

        x = 10000./ wave         # convert to inverse microns 
        npts = len(x)
        a = np.zeros(npts)  
        b = np.zeros(npts)

        good = (x > 0.3) & (x < 1.1) # infrared
        ngood = good.sum()
        if ngood > 0:
            a[good] =  0.574 * x[good]**(1.61)
            b[good] = -0.527 * x[good]**(1.61)

        good = (x >= 1.1) & (x < 3.3) # optical/NIR
        ngood = good.sum()
        if ngood > 0:

            y = x[good] - 1.82

            if dlaw=='odonnell': # new coefficients from O'Donnell (1994)

                c1 = [ 1. , 0.104,   -0.609,    0.701,  1.137,
                    -1.718,   -0.827,    1.647, -0.505 ]
                c2 = [ 0.,  1.952,    2.908,   -3.989, -7.985,    
                    11.102,    5.491,  -10.805,  3.347 ]

            else: # original coefficients

                c1 = [ 1. , 0.17699, -0.50447, -0.02427,  0.72085,
                    0.01979, -0.77530,  0.32999 ]
                c2 = [ 0.,  1.41338,  2.28305,  1.07233, -5.38434,
                    -0.62251,  5.30260, -2.09002 ]

            a[good] = polyval(y,c1)
            b[good] = polyval(y,c2)

        good = (x >= 3.3) & (x < 8) # mid-UV
        ngood = good.sum()
        if ngood > 0:

            y = x[good]
            F_a = np.zeros(ngood)
            F_b = np.zeros(ngood)
            good1 = y > 5.9
            ngood1 = good1.sum()
            if ngood1 > 0:
                y1 = y[good1] - 5.9
                F_a[good1] = -0.04473 * y1**2 - 0.009779 * y1**3
                F_b[good1] =   0.2130 * y1**2  +  0.1207 * y1**3
          
            a[good] =  1.752 - 0.316*y - (0.104 / ( (y-4.67)**2 + 0.341 )) + F_a
            b[good] = -3.090 + 1.825*y + (1.206 / ( (y-4.62)**2 + 0.263 )) + F_b

        good = (x >= 8) & (x <= 11) # far-UV
        if ngood > 0:

            y = x[good] - 8.
            c1 = [ -1.073, -0.628,  0.137, -0.070 ]
            c2 = [ 13.670,  4.257, -0.420,  0.374 ]
            a[good] = polyval(y, c1)
            b[good] = polyval(y, c2)

        k_lambda = r_v * (a + b/r_v)

        # if necessary, make K_LAMBDA a smooth function by extrapolating
        inrange = (wave >= 912.0) & (wave <= 10000.0/0.3)
        ninrange = inrange.sum()
        lowave = wave < 912.0
        nlowave = lowave.sum()
        hiwave = wave > 10000.0/0.3
        nhiwave = hiwave.sum()
        if (nlowave != 0) | (nhiwave != 0):
            if not silent: print("Extrapolating the O'Donnell k(lambda)")
        if (nlowave != 0): k_lambda[lowave] = k_lambda[inrange[0]]
        if (nhiwave != 0): k_lambda[hiwave] = np.minimum(np.interp(wave[hiwave],k_lambda[inrange],wave[inrange]),0)

    # ---------------------------------------------------------------------------    
    # Seaton (1979) - old Milky Way
    if dlaw=='seaton':

        if not r_v: r_v = 3.1

        x = 10000.0/wave         # convert to inverse microns 
        k_lambda = x*0.0

        # Table 2 (adopted from Nandy et al. (1975)       
       
        xprime = np.arange(18)*0.1+1.0
        kprime = [1.36,1.44,1.84,2.04,2.24,2.44,2.66,2.88,3.14,
                    3.36,3.56,3.77,3.96,4.15,4.26,4.40,4.52,4.64]
              
        w1 = x < 2.70
        w2 = (x >= 2.70) & (x <= 3.65)
        w3 = (x > 3.65) & (x <= 7.14)
        w4 = (x > 7.14) & (x <= 10.0)

        if w1.sum() > 0: k_lambda[w1] = np.interp(x[w1],xprime,kprime)
        if w2.sum() > 0: k_lambda[w2] = 1.56+1.048*x[w2]+1.01/((x[w2]-4.60)**2+0.280)
        if w3.sum() > 0: k_lambda[w3] = 2.29+0.848*x[w3]+1.01/((x[w3]-4.60)**2+0.280)
        if w4.sum() > 0: k_lambda[w4] = 16.17-3.20*x[w4]+0.2975*x[w4]**2

        k_lambda = r_v*k_lambda/3.2

    # ---------------------------------------------------------------------------    
    # Fitzpatrick (1999) - new Milky Way, AVGLMC, and LMC2
    if (dlaw=='fm_mw')|(dlaw=='fm_lmc2')|(dlaw=='fm_avglmc'):

        if not r_v: r_v = 3.1

        x = 10000./ wave         # convert to inverse microns 
        k_lambda = x*0.

        if dlaw=='fm_lmc2': x0,gamma,c4,c3,c2,c1 = (4.626,1.05,0.42,1.92,1.31,-2.16)
        elif dlaw=='fm_avglmc': x0,gamma,c4,c3,c2,c1 = (4.596,0.91,0.64,2.73,1.11,-1.28)
        else:
            x0,gamma,c4,c3,c2 = (4.596,0.99,3.23,0.41,-0.824 + 4.717/r_v)
            c1 = 2.030 - 3.007*c2

        # Compute UV portion of A(lambda)/E(B-V) curve using FM fitting function and 
        # R-dependent coefficients
       
        xcutuv = 10000.0/2700.0
        xspluv = 10000.0/np.array([2700.0,2600.0])
        iuv = x >= xcutuv
        if iuv.sum() > 0: xuv = np.append([xspluv,x[iuv]])
        else:  xuv = xspluv

        yuv = c1  + c2*xuv
        yuv = yuv + c3*xuv**2/((xuv**2-x0**2)**2 +(xuv*gamma)**2)
        yuv = yuv + c4*(0.5392*((xuv>5.9)-5.9)**2+0.05644*((xuv>5.9)-5.9)**3)
        yuv = yuv + r_v
        yspluv  = yuv[:2]       # save spline points

        if iuv.sum() > 0: k_lambda[iuv] = yuv[2:] # remove spline points
       
        # Compute optical portion of A(lambda)/E(B-V) curve
        # using cubic spline anchored in UV, optical, and IR

        xsplopir = [0,10000.0/np.array([26500.0,12200.0,6000.0,5470.0,4670.0,4110.0])]
        ysplir   = np.array([0.0,0.26469,0.82925])*r_v/3.1 
        ysplop   = [polyval(r_v, [-4.22809e-01, 1.00270, 2.13572e-04] ),
            polyval(r_v, [-5.13540e-02, 1.00216, -7.35778e-05] ),
            polyval(r_v, [ 7.00127e-01, 1.00184, -3.32598e-05] ),
            polyval(r_v, [ 1.19456, 1.01707, -5.46959e-03, 7.97809e-04, 
            -4.45636e-05] ) ]
       
        ysplopir = [ysplir,ysplop]

        iopir = x < xcutuv
        if iopir.sum() > 0:
            spline=interpolate.splrep(np.append([xsplopir,xspluv]),np.append([ysplopir,yspluv]),k=3)
            k_lambda[iopir] = interpolate.splev(x[iopir],spline)
       
    # ---------------------------------------------------------------------------    
    # LMC, LMC Average
    if (dlaw=='lmc2')|(dlaw=='avglmc'): 
        k_lambda = read_gordon_2003(wave,dlaw=dlaw,r_v=r_v)
        
    # ---------------------------------------------------------------------------    
    # SMC
    if (dlaw=='smc'):
        if not r_v: r_v = 2.74
        # ll = wave / 1.0e4
        k_lambda = np.zeros(len(wave))
        # case1 = np.where((ll >= 0.07) & (ll <= 0.15))
        # case2 = np.where(ll > 0.15)
        # k_lambda[case1] = (2.28002/ll[case1]-2.14851)*r_v/2.74 # new values from NAR, 1/13/2017
        # if len(k_lambda[case2]) > 0:
        #     k_lambda[case2] = read_gordon_2003(wave[case2],dlaw='smc',r_v=r_v)
        file = '/Users/rtheios/kbss/ksmc_updated_faruv.txt'
        l = np.genfromtxt(file, usecols=(0))
        kl = np.genfromtxt(file, usecols=(1))
        
        k_spl = interpolate.splrep(l, kl, k=3)
        k_lambda = interpolate.splev(wave, k_spl)
            
    if dlaw=='ssmc':
        if not r_v: r_v = 2.74
        ll = wave / 1.0e4
        k_lambda = np.zeros(len(wave))
        case1 = np.where((ll >= 0.07) & (ll <= 0.1575))
        case2 = np.where(ll > 0.1575)
        # k_lambda[case1] = (2.29726964/ll[case1]-2.39493337)*r_v/2.74 # RLT 'smoothing' of curve
        k_lambda[case1] = (2.28002/ll[case1]-2.28541)*r_v/2.74
        if len(k_lambda[case2]) > 0:
            k_lambda[case2] = read_gordon_2003(wave[case2],dlaw='smc',r_v=r_v)
        

    # ---------------------------------------------------------------------------    
    # Li & Draine (2001)
    if dlaw=='li':
        path = os.environ['PYTHONPATH']+'/dust/etc/'
        li_wave,albedo,g,sigma_ext,kappa = readcol(path+'li_draine01.dat',twod=False,comment='#')
        if not r_v: r_v = 3.1
        spline = interpolate.splrep(li_wave*1e4,sigma_ext,k=3)
        k_lambda = r_v*2.146e21*interpolate.splev(wave,spline)
    
    # ---------------------------------------------------------------------------    
    # Reddy et al. (2015+2016)
    if dlaw=='reddy':
        
        ll = wave/1.e04
        k = np.zeros(len(wave))
    
        for i in range(len(ll)):
            if ll[i] <= 0.15:
                k[i] =  2.191 + 0.974/ll[i]
            else:
                if ll[i] > 0.15 and ll[i] < 0.60:
                    k[i] = -5.726 + 4.004/ll[i] - 0.525/ll[i]**2 + 0.029/ll[i]**3 + 2.505
                else:
                    k[i] = -2.672 - 0.010/ll[i] + 1.532/ll[i]**2 - 0.412/ll[i]**3 + 2.505

        k_lambda = k
    
    # ---------------------------------------------------------------------------    
    # 1/lambda with Rv = 3.1
    if dlaw=='invlam':
        
        if not r_v: r_v = 3.1
        ll = np.array(wave)/1.e04
        k_lambda = 1.0/ll * (0.55*r_v)
    
    # ---------------------------------------------------------------------------    
    # Salim et al. 2018 
    if dlaw=='salim_lm': # Mstar <= 10^10 Msun
        
        ll = np.array(wave)/1.0e04
        k = np.zeros(len(wave))
        
        for i in range(len(ll)):
            #if ll[i] >= 0.09 and ll[i] < 2.05:
            if ll[i] < 2.05:
                k[i] = -3.80 + 2.25/ll[i] - 0.073/ll[i]**2 + 0.0092/ll[i]**3 + (2.74 * 0.035**2 * ll[i]**2)/((ll[i]**2 - 0.2175**2)**2 + ll[i]**2 * 0.035**2) + 2.72
            elif ll[i] >= 2.05:
                k[i] = 0
        
        k_lambda = k
        
    if dlaw=='salim_hm': # Mstar > 10^10 Msun
        
        ll = wave/1.0e04
        k = np.zeros(len(wave))
        
        for i in range(len(ll)):
            #if ll[i] >= 0.09 and ll[i] < 2.09:
            if ll[i] < 2.09:
                k[i] = -4.12 + 2.56/ll[i] - 0.152/ll[i]**2 + 0.0104/ll[i]**3 + (2.11 * 0.035**2 * ll[i]**2)/((ll[i]**2 - 0.2175**2)**2 + ll[i]**2 * 0.035**2) + 2.93
            elif ll[i] >= 2.09:
                k[i] = 0
        
        k_lambda = k
        
    # if len(k_lambda) == 1: k_lambda = k_lambda[0]
    return k_lambda
    
def ebv_neb(BD_obs,BD_int=2.89,dlaw='ccm',r_v=None,min0=True,**kwargs):
    '''Calculated E(B-V)_neb for a given Balmer decrement ratio (BD)
    given an intrinsic BD and extinction law.

    If min0==True, negative values of E(B-V) are set equal to zero.
    '''
    k_a = k_lambda(6562.85,dlaw=dlaw,r_v=r_v)
    k_b = k_lambda(4861.36,dlaw=dlaw,r_v=r_v)

    if min0:
        return np.maximum(-2.5*np.log10(BD_obs/BD_int)/(1-k_b/k_a)/k_a,0)
    else:
        return -2.5*np.log10(BD_obs/BD_int)/(1-k_b/k_a)/k_a

def ebv2Alam(wave,ebv,dlaw='ccm',r_v=None,**kwargs):
    '''Calculates the extinction in magnitudes at a rest-wavelength
    (or series of wavelengths) wave, which is assumed to be the vacuum wavelength.
    '''
    if hasattr(ebv,"__len__")&hasattr(wave,"__len__"):
        Alam=[]
        for ebvi in ebv:
            Alam.append(k_lambda(wave,dlaw=dlaw,r_v=r_v,**kwargs)*ebvi)
        return np.array(Alam)
    else:
        return k_lambda(wave,dlaw=dlaw,r_v=r_v,**kwargs)*ebv

def ebv2ext(wave,ebv,dlaw='ccm',r_v=None,**kwargs):
    '''Calculates the extinction as a multiplicative factor at a rest-wavelength
    (or series of wavelengths) wave, which is assumed to be the vacuum wavelength.
    '''
    return 10**(-0.2*ebv2Alam(wave=wave,ebv=ebv,dlaw=dlaw,r_v=r_v,**kwargs))
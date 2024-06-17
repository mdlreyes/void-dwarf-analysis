# makeplots.py
# Script to make plots
#
# Created: 5 April 2021
######################################

#Backend for python3 on mahler
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Change fonts
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

# Astropy packages
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM  # needed to compute redshifts
cosmo = FlatLambdaCDM(H0=67.4, Om0=0.315)  # using Planck (2018) params
from astropy.coordinates import Distance

# Import other packages
import numpy as np
from astropy.io import ascii
import cmasher as cmr
import pandas as pd

def vsigma_plot(param='mass', plot_path='plots/', mass='sdss', inclination=False, plotline=True, onsky=False):
    """ Plots v/sigma as a function of another parameter

        Args:
            param (str): Parameter to plot on x-axis 
                        (options: 'mass')
            plot_path (str): Path to store output plots
            mass (str): Either SDSS ('sdss') or WISE ('wise') stellar masses 
	"""

    # Read in data from Wheeler+17
    wheeler17 = ascii.read('../data/wheeler17_tab1.txt')

    y_wheeler = wheeler17['vsigma']
    y_wheeler_err = np.asarray([np.asarray(wheeler17['evsigma_down']), np.asarray(wheeler17['evsigma_up'])])

    if param=='mass':
        x_wheeler = np.log10(wheeler17['Mstar']*1.e6)

    if param=='dLstar':
        x_wheeler = wheeler17['dLstar']

    if param=='ellipticity':
        x_wheeler = wheeler17['Ellipticity']

    # Make masks for Wheeler+17 data
    ufd_idx = [True if i.startswith('UF') else False for i in wheeler17['Category']]
    iso_idx = [True if i.startswith('Iso') else False for i in wheeler17['Category']]
    sat_idx = [False if ufd_idx[i] or iso_idx[i] else True for i in range(len(ufd_idx))]

    # Read in data from void dwarfs
    voiddata = ascii.read('../data/sample_FINAL_new.csv').filled(-999.0)
    #print(voiddata)

    vmax_version = 'mc'
    
    y_void = voiddata['vsigma_'+vmax_version]
    y_void_err = np.vstack((voiddata['vsigma_'+vmax_version+'_err_lo'],voiddata['vsigma_'+vmax_version+'_err_up']))
    print(np.shape(y_void_err))

    #y_void_err_lowerlim = y_void_lowerlim * np.sqrt((voiddata['v_Binney_err']/voiddata['v_Binney'])**2. + (voiddata['sigma_Binney_err']/voiddata['sigma_Binney'])**2.)

    if param=='mass':
        if mass != 'wise':
            x_void = voiddata['massSDSS']
            x_void_err = np.asarray([np.asarray(voiddata['massSDSS'] - voiddata['massSDSS_lo']), np.asarray(voiddata['massSDSS_hi'] - voiddata['massSDSS'])])
        else:
            x_void = voiddata['massWISE']
            x_void_err = np.asarray([voiddata['massWISE_lo'],voiddata['massWISE_hi']])

        #labels = [None,None,None,None,None,None]
        #else:
        labels = ['Local Group ultra-faint','Local Group satellite','Local Group isolated','Void','Field','Larger than KCWI FoV']

    if param=='dLstar':
        if onsky:
            x_void = voiddata['projdLstar_wise']
        else:
            x_void = voiddata['dLstar_wise']
        x_void[x_void<-990] = 3000
        x_void_err = np.zeros((2,len(x_void)))

    if param=='ellipticity':
        x_void = voiddata['ellipticity']
        x_void_err = np.zeros((2,len(x_void)))

    if param=='redshift':
        x_void = voiddata['z']
        x_void_err = np.zeros((2,len(x_void)))

    # Make masks for void data
    extended_idx = [True if (voiddata['ID'][i] in ['1228631', 'Pisces A', 'Pisces B', 'control872']) else False for i in range(len(x_void))]
    good_idx = [True if ~np.any(np.isclose([x_void[i], voiddata['vmax_'+vmax_version][i], voiddata['sigma'][i]], -999)) and y_void[i] > 0. else False for i in range(len(x_void))]
    void_idx = [True if (voiddata['Type'][i]=='void' and good_idx[i]) else False for i in range(len(x_void))]
    control_idx = [True if (voiddata['Type'][i]=='control' and good_idx[i]) else False for i in range(len(x_void))]

    # Try messing with inclination
    if inclination:
        y_void /= np.sin(voiddata['inclination_rad'])

    # Try messing with averaging
    #avg_factor = np.zeros(len(voiddata['z']))
    test_angle = (cosmo.angular_diameter_distance(voiddata['z']).to(u.kpc)/206265).value
    avg_factor = test_angle/np.min(test_angle)
    avg_factor = (avg_factor)**(1/4.)
    #print(avg_factor)
    #y_void *= 2

    # Make figure
    fig = plt.figure(figsize=(10,5))
    ax = plt.subplot()

    # Plot data from Wheeler+17
    if param != 'redshift':
        ax.errorbar(x_wheeler[ufd_idx], y_wheeler[ufd_idx], yerr=y_wheeler_err[:,ufd_idx], label=labels[0],
                    color=plt.cm.Set2(0), linestyle='None', marker='^', markersize=8, linewidth=1, alpha=0.8)

        ax.errorbar(x_wheeler[sat_idx], y_wheeler[sat_idx], yerr=y_wheeler_err[:,sat_idx], label=labels[1],
                    color=plt.cm.Set2(1), linestyle='None', marker='s', markersize=7, linewidth=1, alpha=0.8)

        ax.errorbar(x_wheeler[iso_idx], y_wheeler[iso_idx], yerr=y_wheeler_err[:,iso_idx], label=labels[2],
                    mfc='white', mec=plt.cm.Set2(1), ecolor=plt.cm.Set2(1), linestyle='None', marker='s', markersize=8, linewidth=1)

    # Plot my data
    '''
    ax.errorbar(x_void[void_idx], y_void[void_idx], xerr=x_void_err[:,void_idx], yerr=y_void_err[:,void_idx], label=labels[3],
               mfc='white', mec=plt.cm.Dark2(2), ecolor=plt.cm.Dark2(2), linestyle='None', marker='o', markersize=8, linewidth=1)    
     
    ax.errorbar(x_void[control_idx], y_void[control_idx], xerr=x_void_err[:,control_idx], yerr=y_void_err[:,control_idx], label=labels[4],
               color=plt.cm.Dark2(2), linestyle='None', marker='o', markersize=8, linewidth=1)
    
    ax.errorbar(x_void[extended_idx], y_void[extended_idx], label=labels[5],
               color='r', linestyle='None', marker='x', markersize=6, linewidth=1)
    '''
    ax.errorbar(x_void[good_idx], y_void[good_idx], xerr=x_void_err[:,good_idx], yerr=y_void_err[:,good_idx], label='de los Reyes et al. (2023)',
               color=plt.cm.Dark2(2), linestyle='None', marker='o', markersize=8, linewidth=1)

    # Do some math to get best-fit line for mass
    if param=='mass' or param=='dLstar' and plotline:

        Niter = 100000
        poly = np.zeros((Niter,2))
        for i in range(Niter):

            # Randomly perturb data
            y_wheeler_err_avg = (y_wheeler_err[0] + y_wheeler_err[1])/2.
            y_wheeler_new = np.random.normal(y_wheeler, y_wheeler_err_avg)
            y_void_new = np.random.normal(y_void[good_idx], np.average(y_void_err[:,good_idx], axis=0))

            x = np.concatenate((x_wheeler[sat_idx],x_wheeler[iso_idx],x_void[good_idx]))
            if param=='dLstar':
                x = np.log10(x)
            y = np.concatenate((y_wheeler_new[sat_idx],y_wheeler_new[iso_idx],y_void_new))
        
            poly[i,:] = np.polyfit(x, y, 1)

        poly_med = np.median(poly, axis=0)
        poly_lo = np.percentile(poly, 16, axis=0)
        poly_hi = np.percentile(poly, 84, axis=0)

        print(poly_med, poly_med-poly_lo, poly_hi-poly_med)

    if param=='mass':
        # Plot Keck2024A targets
        #mass_keck2024a = np.array([8.02931976,8.13880634,8.24504566,8.27385139,8.70394897,8.721735,8.92823124,8.9456224,9.52635288,9.86613941])
        #for i, mass in enumerate(mass_keck2024a):
        #    if i==0:
        #        plt.axvline(mass, color='C0', lw=1, label='Proposed target masses')
        #    else:
        #        plt.axvline(mass, color='C0', lw=1)

        # Plot Keck2024B params
        plt.errorbar(np.log10(6e10), 7, yerr=3, marker='s', color='k', ls='None', label='MW') # van der Marel (2003)
        plt.errorbar(np.log10(10.3e10), 220/35.7, marker='o', color='k', ls='None', label='M31') # Collins et al. (2010)
        plt.axvspan(9,10.5, color='r', alpha=0.3)

    # Add title and labels
    if param=='mass':
        plt.xlabel(r'$\log M_{\star}$ (M$_{\odot}$)', fontsize=20)
        xlim = [3.25,11.5]
        plt.xlim(xlim)
        plt.ylim([-1,10.])
        plt.plot(xlim,[1,1],':k')
        if plotline:
            plt.plot(xlim, poly_med[0]*np.array(xlim) + poly_med[1], 'r-', label=r"Best-fit line (excluding LG ultra-faints") #: $y=${:.2f}$x-${:.2f})".format(poly_med[0], np.abs(poly_med[1])))
    if param=='dLstar':
        if onsky:
            plt.xlabel(r'$d_{L_{\star}}$, projected (kpc)', fontsize=20)
        else:
            plt.xlabel(r'$d_{L_{\star}}$ (kpc)', fontsize=20)
        plt.xscale('log')
        xlim = [10,1.5e4]
        plt.xlim(xlim)
        plt.ylim([-0.05,4.])
        plt.plot([0,2e4],[0,0],'-k')
        if plotline:
            plt.plot(xlim, poly_med[0]*np.array(xlim) + poly_med[1], 'r-', label=r"Best-fit line (excluding LG ultra-faints: $y=${:.2f}$x-${:.2f})".format(poly_med[0], np.abs(poly_med[1])))
    if param=='ellipticity':
        plt.xlabel(r'Ellipticity', fontsize=20)
        plt.xlim([0,0.82])
        plt.ylim([-0.05,4.])

        # Plot Binney (1978) curve
        binney = np.genfromtxt('../data/binney.txt', delimiter=',')
        plt.plot(binney[:,0],binney[:,1],'k-')
    if param=='redshift':
        plt.xlabel(r'Redshift', fontsize=20)
        plt.xlim([0,0.02])
        plt.ylim([-0.05,4.])

    plt.legend(loc='upper left', fontsize=14, ncol=2)
    plt.ylabel(r'$v_{\mathrm{rot}}/\sigma_{\star}$', fontsize=20)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)

    if mass=='wise' and param=='mass':
        param = param+'wise'

    if inclination:
        param = param+'_inclination'

    if onsky:
        param = param+'_onsky'

    plt.savefig((plot_path+'vsigma_'+param+'_withkeck.pdf'), bbox_inches='tight')
    plt.show()

    return

def mass_metallicity(plot_path='plots/'):
    """ Plots mass-metallicity relation

        Args:
            plot_path (str): Path to store output plots
	"""

    # Read in LVL data from Berg+12
    berg12_masses = ascii.read('../data/berg12_tab1.txt', guess=False, delimiter='\t', comment='#').filled(-999)
    x_berg12 = np.asarray([float(i.split(' +or- ')[0]) for i in berg12_masses['log M_sstarf']])
    x_berg12_err = np.asarray([float(i.split(' +or- ')[1]) for i in berg12_masses['log M_sstarf']])

    berg12_Z = ascii.read('../data/berg12_tab5.txt', guess=False, delimiter='\t', comment='#').filled(-999)
    y_berg12 = np.ones(len(x_berg12))*-999
    y_berg12_err = np.ones(len(x_berg12))*-999
    for idx, name in enumerate(berg12_masses['Galaxy']):
        if name in berg12_Z['Galaxy']:
            i = np.where(berg12_Z['Galaxy'] == name)[0][0]
            metal = berg12_Z['12 + log(O/H)'][i].split(' +or- ')
            y_berg12[idx] = float(metal[0])
            y_berg12_err[idx] = float(metal[1])

    # Make masks for Berg+12 data
    berg_idx = [True if ~np.any(np.isclose([x_berg12[i], y_berg12[i]], -999)) else False for i in range(len(x_berg12))]

    # Read in data from void dwarfs
    voiddata = ascii.read('../data/sample_FINAL.csv').filled(-999.0)
    x_void = voiddata['massSDSS']
    x_void_err = np.asarray([np.asarray(voiddata['massSDSS'] - voiddata['massSDSS_lo']), np.asarray(voiddata['massSDSS_hi'] - voiddata['massSDSS'])])
    
    # Get metallicity info
    y_void = np.zeros(len(x_void))
    for idx in range(len(y_void)):
        if voiddata['Z_Te_distmean'][idx] > -990:
            y_void[idx] = voiddata['Z_Te_distmean'][idx]
        elif voiddata['Z_Te_exact'][idx] > -990:
            y_void[idx] = voiddata['Z_Te_exact'][idx]

    # Make masks for void data
    good_idx = [True if ~np.any(np.isclose([x_void[i], y_void[i]], -999)) and y_void[i] > 0. else False for i in range(len(x_void))]
    void_idx = [True if (voiddata['Type'][i]=='void' and good_idx[i]) else False for i in range(len(x_void))]
    control_idx = [True if (voiddata['Type'][i]=='control' and good_idx[i]) else False for i in range(len(x_void))]

    # Make figure
    fig = plt.figure(figsize=(8,6))
    ax = plt.subplot()

    # Plot LVL data
    ax.errorbar(x_berg12[berg_idx], y_berg12[berg_idx], xerr=x_berg12_err[berg_idx], yerr=y_berg12_err[berg_idx],
                color=plt.cm.Set2(7), linestyle='None', marker='s', markersize=5, linewidth=1, label='LVL field')

    # Plot my data
    ax.errorbar(x_void[control_idx], y_void[control_idx], #xerr=x_void_err[:,control_idx], yerr=y_void_err[control_idx],
               color=plt.cm.Set2(0), ecolor=plt.cm.Set2(0), linestyle='None', marker='o', markersize=8, linewidth=1, label='Field')    
     
    ax.errorbar(x_void[void_idx], y_void[void_idx], #xerr=x_void_err[:,void_idx], yerr=y_void_err[void_idx],
               mfc='white', mec=plt.cm.Set2(1), linestyle='None', marker='o', markersize=8, linewidth=2, label='Void')

    #ax.errorbar(x_void[lowerlim_idx], y_void_lowerlim[lowerlim_idx], label='Lower limits', 
    #           color=plt.cm.Set2(2), linestyle='None', marker='o', markersize=4, linewidth=0.5, alpha=0.5) # yerr=y_void_err_lowerlim[lowerlim_idx], lolims=True, 
    
    # Add title and labels
    plt.legend(loc='best')
    plt.xlabel(r'$\log M_{\star}$ (M$_{\odot}$)', fontsize=16)
    plt.ylabel(r'12+log(O/H)', fontsize=16)
    #plt.plot([3.25,9.5],[1,1],':k')
    #plt.xlim([3.25,9.5])

    plt.savefig((plot_path+'massmetallicity.pdf'), bbox_inches='tight')
    plt.show()

    return

def dLstar_mass(plot_path='plots/', mass='sdss'):
    """ Plots dLstar vs mass, colored by v/sigma

        Args:
            plot_path (str): Path to store output plots
            mass (str): Either SDSS ('sdss') or WISE ('wise') stellar masses 
	"""

    # Data from Keck2024A proposal
    mass_keck2024a = np.array([8.02931976,8.13880634,8.24504566,8.27385139,8.70394897,8.721735,8.92823124,8.9456224,9.52635288,9.86613941])
    dLstar_keck2024a = np.array([2075,1067,1289,4612,2436,3938,2208,491,2523,1401])

    # Read in cluster dE data from Geha+03
    geha03 = np.genfromtxt('../data/geha03_tab1.txt', delimiter='\t', names=True)
    vsigma_virgo = geha03['vrot']/geha03['sigma']
    vsigma_virgo_err = vsigma_virgo * np.sqrt((geha03['vrot_err']/geha03['vrot'])**2. + (geha03['sigma_err']/geha03['sigma'])**2.)
    print(vsigma_virgo, vsigma_virgo_err)

    # Test calculation of stellar mass from Bell et al. (2003) color-M/L relation, assuming B-V=0.8
    mass_virgo = np.log10(10**(-0.628+1.305*(0.8)) * 10**(0.4*(4.81-geha03['absmag_V0'])))
    dLstar_virgo = np.ones_like(mass_virgo)

    # Read in LV data from Wheeler+17
    wheeler17 = ascii.read('../data/wheeler17_tab1.txt')

    vsigma_wheeler = wheeler17['vsigma']
    vsigma_wheeler_err = np.asarray([np.asarray(wheeler17['evsigma_down']), np.asarray(wheeler17['evsigma_up'])])
    mass_wheeler = np.log10(wheeler17['Mstar']*1.e6)
    dLstar_wheeler = wheeler17['dLstar']

    # Read in data from void dwarfs
    voiddata = ascii.read('../data/sample_FINAL_new.csv').filled(-999.0)
    
    vsigma_void = voiddata['vsigma_mc']
    vsigma_void_uperr = voiddata['vsigma_mc_err_up']
    vsigma_void_loerr = voiddata['vsigma_mc_err_lo']

    if mass != 'wise':
        mass_void = voiddata['massSDSS']
        mass_void_err = np.asarray([np.asarray(voiddata['massSDSS'] - voiddata['massSDSS_lo']), np.asarray(voiddata['massSDSS_hi'] - voiddata['massSDSS'])])
    else:
        mass_void = voiddata['massWISE']
        mass_void_err = np.asarray([voiddata['massWISE_lo'],voiddata['massWISE_hi']])

    dLstar_void = voiddata['dLstar']
    dLstar_void[dLstar_void<-990] = 3000

    # Make masks for void data
    good_idx = [True if ~np.any(np.isclose([dLstar_void[i], mass_void[i], voiddata['vmax_mc'][i], voiddata['sigma'][i]], -999)) else False for i in range(len(mass_void))]

    # Make figure
    fig = plt.figure(figsize=(8,6))

    vmax = 2.5

    # Plot data from Wheeler+17
    sc = plt.scatter(dLstar_wheeler, mass_wheeler, label='Local Volume', c=vsigma_wheeler, cmap=cmr.bubblegum, vmin=0, vmax=vmax, marker='s', s=40, alpha=0.8)

    # Plot my data
    plt.scatter(dLstar_void[good_idx], mass_void[good_idx], label='Field', c=vsigma_void[good_idx], cmap=cmr.bubblegum, vmin=0, vmax=vmax, marker='o', s=40, alpha=0.8)

    plt.scatter(dLstar_keck2024a, mass_keck2024a, label='This proposal', c='None', edgecolor='r', marker='*', s=100)

    # Plot cluster data
    #plt.scatter(dLstar_virgo, mass_virgo, label='Cluster', c=vsigma_virgo, cmap=cmr.bubblegum, vmin=0, vmax=vmax, marker='^', s=40, alpha=0.8)
    
    #ax.errorbar(dLstar_void[good_idx], mass_void[good_idx], yerr=mass_void_err[:,good_idx], label='Field', 
    #           color=vsigma_void[good_idx], cmap=cmr.bubblegum, linestyle='None', marker='o', markersize=8, linewidth=1)

    # Add title and labels
    cbar = plt.colorbar(sc)
    cbar.set_label(r'$v_{\mathrm{rot}}/\sigma_{\star}$', fontsize=20)
    plt.ylabel(r'$\log M_{\star}$ (M$_{\odot}$)', fontsize=20)
    #plt.ylim([3.25,10.2])
    plt.xlabel(r'$d_{L_{\star}}$ (kpc)', fontsize=20)
    plt.xscale('log')
    plt.xlim([10,50000])
    plt.legend(loc='best', fontsize=14)
    #plt.ylabel(r'$v_{\mathrm{rot}}/\sigma_{\star}$', fontsize=20)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)

    outfile = 'dLstar_mass'
    if mass=='wise':
        outfile += 'wise'

    plt.savefig((plot_path+outfile+'.png'), bbox_inches='tight')
    plt.show()

    return

def vsigma_dist(plot_path='plots/'):
    '''Plot distribution of vsigma'''

    # Read in data from void dwarfs
    voiddata = ascii.read('../data/sample_FINAL_new.csv').filled(-999.0)
    
    y_void = voiddata['vsigma']
    y_void_err = voiddata['vsigma_err']

    # Make masks for void data
    good_idx = [True if ~np.any(np.isclose([voiddata['sigma'][i]], -999)) and y_void[i] > 0. else False for i in range(len(y_void))]
    void_idx = [True if (voiddata['Type'][i]=='void' and good_idx[i]) else False for i in range(len(y_void))]
    control_idx = [True if (voiddata['Type'][i]=='control' and good_idx[i]) else False for i in range(len(y_void))]

    def plotcdf(array, color, weight, alpha):
        # getting data of the histogram
        count, bins_count = np.histogram(array, bins=10)
        
        # finding the PDF and CDF of histogram
        pdf = count / sum(count)
        cdf = np.cumsum(pdf)

        plt.plot(np.max(bins_count[1:]) - bins_count[1:], cdf, label="CDF", linestyle='-', color=color, lw=weight, alpha=alpha)

    # Try messing with inclination
    for i in range(1000):
        inclination = np.random.uniform(0, np.pi/2, size=np.shape(y_void[good_idx]))
        y_void_new = y_void[good_idx]/np.sin(inclination)
        plotcdf(y_void_new, color='gray', weight=0.5, alpha=0.5)

    plotcdf(y_void[good_idx], color='k', weight=2, alpha=1)

    # Format plot
    plt.ylabel(r'$f(>v_{\mathrm{rot}}/\sigma_{\star})$', fontsize=20)
    plt.xlabel(r'$v_{\mathrm{rot}}/\sigma_{\star}$', fontsize=20)
    plt.ylim(0,1)
    plt.xlim(0,np.max(y_void[good_idx]))
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.show()

    return

def compare_1d2d(plot_path='plots/', param='vsigma'):
    """ Make plots comparing kinematics measured from IFU vs (mock) long-slit data """

    voiddata = ascii.read('../data/sample_FINAL_new.csv').filled(-999.0)

    if param=='vsigma':
        properties = ['vmax','sigma','vsigma']
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16,5))

        for i, property in enumerate(properties):
            data_ifu = voiddata[property]
            data_ifu_err = voiddata[property+'_err']
            data_2d = voiddata[property+'2d']
            data_2d_err = voiddata[property+'_err2d']
            goodidx = np.where((data_ifu > 0.) & (data_2d > 0.))[0]

            if property=='vmax':
                limits = [0,100]
                label = r'$v_{\mathrm{rot}}$ (km/s)'
            elif property=='sigma':
                limits = [0,100]
                label = r'$\sigma_{\star}$ (km/s)'
            elif property=='vsigma':
                limits = [0,3]
                label = r'$v_{\mathrm{rot}}/\sigma_{\star}$'

            ax[i].errorbar(data_ifu[goodidx], data_2d[goodidx], xerr=data_ifu_err[goodidx], yerr=data_2d_err[goodidx],
                            color='cornflowerblue', linestyle='None', marker='o', markersize=8, linewidth=1)

            ax[i].plot(limits, limits, 'k--')
            ax[i].set_xlim(limits)
            ax[i].set_ylim(limits)
            ax[i].tick_params(axis='both', labelsize=14)
            ax[i].set_xlabel('IFU '+label, fontsize=18)
            ax[i].set_ylabel('Long-slit '+label, fontsize=18)

        fig.tight_layout()
        plt.savefig(plot_path+'ifu_longslit_comparison.png', bbox_inches='tight')
        plt.show()

    if param=='sigma':
        versions = ['2d','desi']
        labels = ['Long-slit (DBSP)', 'Fiber (DESI)']
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,5))

        data_ifu = voiddata['sigma']
        data_ifu_err = voiddata['sigma_err']

        for i, version in enumerate(versions):

            data_comparison = voiddata['sigma'+version]
            errs_comparison = voiddata['sigma_err'+version]
            
            goodidx = np.where((data_ifu > 0.68) & (data_comparison > 0.68))[0] # & (errs_comparison < data_comparison))[0]

            plt.errorbar(data_ifu[goodidx], data_comparison[goodidx], xerr=data_ifu_err[goodidx], yerr=errs_comparison[goodidx],
                            color=plt.cm.Set2(i), linestyle='None', marker='o', markersize=8, linewidth=1, label=labels[i]+', $N$='+str(len(goodidx)))

            #plt.hist((data_comparison[goodidx] - data_ifu[goodidx])/data_ifu[goodidx], bins=np.linspace(-0.5, 2, 15), color=plt.cm.Set2(i), alpha=0.5, label=labels[i])

        limits = [0,80]
        plt.plot(limits, limits, 'k--')
        plt.xlim(limits)
        plt.ylim(limits)
        plt.xlabel(r'IFU $\sigma_{\star}$ (km/s)', fontsize=18)
        plt.ylabel('Comparison $\sigma_{\star}$ (km/s)', fontsize=18)

        ax.tick_params(axis='both', labelsize=14)
        #plt.xlabel('Percent difference from IFU $\sigma_{\star}$', fontsize=18)

        plt.legend(loc='best', fontsize=14)

        fig.tight_layout()
        plt.savefig(plot_path+'sigma_comparison.png', bbox_inches='tight')
        plt.show()

    return

def compare_nslits(plot_path='plots/', param='sigma'):
    """ Make plots comparing kinematics measured from IFU vs (mock) long-slit data """

    ifudata = pd.read_csv('../data/sample_FINAL_new.csv', delimiter=',', header=0, index_col='ID')

    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(21,5))
    versions = ['1 slit', '2 slits', '3 slits', '4 slits']
    for i in range(4):

        nangles = i+1

        slitdata = pd.read_csv('../redux/output/vsigma_%d.txt'%nangles, header=0, index_col='ID')

        # test
        newdata = ifudata.join(slitdata, rsuffix='_%d'%nangles)

        goodidx = np.where((~np.isnan(newdata[param])) & (~np.isnan(newdata[param+'_%d'%nangles])))[0]

        axs[i].errorbar(newdata[param], newdata[param+'_%d'%nangles], xerr=newdata[param+'_err'], yerr=newdata[param+'_err'+'_%d'%nangles],
                        color=plt.cm.Set2(i), linestyle='None', marker='o', markersize=8, linewidth=1, label=versions[i]+', $N$='+str(len(goodidx)))

        axs[i].tick_params(axis='both', labelsize=14)
        if param=='vmax':
            limits = [0,100]
            label = r'$v_{\mathrm{rot}}$ (km/s)'
        elif param=='sigma':
            limits = [0,100]
            label = r'$\sigma_{\star}$ (km/s)'
        elif param=='vsigma':
            limits = [0,3]
            label = r'$v_{\mathrm{rot}}/\sigma_{\star}$'
        
        axs[i].plot(limits, limits, 'k--')
        axs[i].set_xlim(limits)
        axs[i].set_ylim(limits)
        axs[i].set_xlabel(r'IFU '+label, fontsize=18)
        axs[0].set_ylabel('Comparison '+label, fontsize=18)
        axs[i].legend(loc='upper left', fontsize=14)

    fig.tight_layout()
    plt.savefig(plot_path+'slits_'+param+'.png', bbox_inches='tight')
    plt.show()

    return

if __name__ == "__main__":

    #vsigma_plot(param='dLstar', inclination=False, plotline=False, onsky=True)
    vsigma_plot(param='mass', mass='wise', inclination=False, plotline=False)
    #vsigma_plot(param='redshift', inclination=False, plotline=False)
    #vsigma_plot(param='ellipticity', inclination=False)
    #mass_metallicity()
    #dLstar_mass()
    #vsigma_dist()
    #compare_nslits(param='vmax')
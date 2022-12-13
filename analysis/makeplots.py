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

import cmasher as cmr

# Import other packages
import numpy as np
from astropy.io import ascii

def vsigma_plot(param='mass', plot_path='plots/', mass='sdss', inclination=False):
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

    # Correct for template dispersion (~66 km/s)
    sigma = np.sqrt(np.copy(voiddata['sigma'])**2 - 66.**2)
    #print(np.shape(sigma))
    
    y_void = voiddata['vsigma']
    y_void_err = voiddata['vsigma_err']

    #y_void_err_lowerlim = y_void_lowerlim * np.sqrt((voiddata['v_Binney_err']/voiddata['v_Binney'])**2. + (voiddata['sigma_Binney_err']/voiddata['sigma_Binney'])**2.)

    if param=='mass':
        if mass != 'wise':
            x_void = voiddata['massSDSS']
            x_void_err = np.asarray([np.asarray(voiddata['massSDSS'] - voiddata['massSDSS_lo']), np.asarray(voiddata['massSDSS_hi'] - voiddata['massSDSS'])])
        else:
            x_void = voiddata['massWISE']
            x_void_err = np.asarray([voiddata['massWISE_lo'],voiddata['massWISE_hi']])

    if param=='dLstar':
        x_void = voiddata['dLstar_wise']
        x_void[x_void<-990] = 3000
        x_void_err = np.zeros((2,len(x_void)))

    if param=='ellipticity':
        x_void = voiddata['ellipticity']
        x_void_err = np.zeros((2,len(x_void)))

    # Make masks for void data
    good_idx = [True if ~np.any(np.isclose([x_void[i], voiddata['vmax'][i], voiddata['sigma'][i]], -999)) and y_void[i] > 0. else False for i in range(len(x_void))]
    void_idx = [True if (voiddata['Type'][i]=='void' and good_idx[i]) else False for i in range(len(x_void))]
    control_idx = [True if (voiddata['Type'][i]=='control' and good_idx[i]) else False for i in range(len(x_void))]

    # Try messing with inclination
    if inclination:
        y_void /= np.sin(voiddata['inclination_rad'])

    # Make figure
    fig = plt.figure(figsize=(10,5))
    ax = plt.subplot()

    # Plot data from Wheeler+17
    ax.errorbar(x_wheeler[ufd_idx], y_wheeler[ufd_idx], yerr=y_wheeler_err[:,ufd_idx], #label='Local Group ultra-faint',
                color=plt.cm.Set2(0), linestyle='None', marker='^', markersize=8, linewidth=1, alpha=0.8)

    ax.errorbar(x_wheeler[sat_idx], y_wheeler[sat_idx], yerr=y_wheeler_err[:,sat_idx], #label='Local Group satellite',
                color=plt.cm.Set2(1), linestyle='None', marker='s', markersize=7, linewidth=1, alpha=0.8)

    ax.errorbar(x_wheeler[iso_idx], y_wheeler[iso_idx], yerr=y_wheeler_err[:,iso_idx], #label='Local Group isolated',
                mfc='white', mec=plt.cm.Set2(1), ecolor=plt.cm.Set2(1), linestyle='None', marker='s', markersize=8, linewidth=1)

    # Plot my data
    ax.errorbar(x_void[void_idx], y_void[void_idx], xerr=x_void_err[:,void_idx], yerr=y_void_err[void_idx], #label='Void',
               mfc='white', mec=plt.cm.Dark2(2), ecolor=plt.cm.Dark2(2), linestyle='None', marker='o', markersize=8, linewidth=1)    
     
    ax.errorbar(x_void[control_idx], y_void[control_idx], xerr=x_void_err[:,control_idx], yerr=y_void_err[control_idx], #label='Field', 
               color=plt.cm.Dark2(2), linestyle='None', marker='o', markersize=8, linewidth=1)

    # Do some math to get best-fit line for mass
    if param=='mass' or param=='dLstar':

        Niter = 100000
        poly = np.zeros((Niter,2))
        for i in range(Niter):

            # Randomly perturb data
            y_wheeler_err_avg = (y_wheeler_err[0] + y_wheeler_err[1])/2.
            y_wheeler_new = np.random.normal(y_wheeler, y_wheeler_err_avg)
            y_void_new = np.random.normal(y_void[good_idx], y_void_err[good_idx])

            x = np.concatenate((x_wheeler[sat_idx],x_wheeler[iso_idx],x_void[good_idx]))
            if param=='dLstar':
                x = np.log10(x)
            y = np.concatenate((y_wheeler_new[sat_idx],y_wheeler_new[iso_idx],y_void_new))
        
            poly[i,:] = np.polyfit(x, y, 1)

        poly_med = np.median(poly, axis=0)
        poly_lo = np.percentile(poly, 16, axis=0)
        poly_hi = np.percentile(poly, 84, axis=0)

        print(poly_med, poly_med-poly_lo, poly_hi-poly_med)

    # Add title and labels
    if param=='mass':
        plt.xlabel(r'$\log M_{\star}$ (M$_{\odot}$)', fontsize=20)
        xlim = [3.25,9.7]
        plt.xlim(xlim)
        plt.ylim([-0.05,4.])
        plt.plot(xlim,[1,1],':k')
        plt.plot(xlim, poly_med[0]*np.array(xlim) + poly_med[1], 'r-', label=r"Best-fit line (excluding LG ultra-faints: $y=${:.2f}$x-${:.2f})".format(poly_med[0], np.abs(poly_med[1])))
    if param=='dLstar':
        plt.xlabel(r'$d_{L_{\star}}$ (kpc)', fontsize=20)
        plt.xscale('log')
        plt.xlim([10,1.5e4])
        plt.ylim([-0.05,4.])
        plt.plot([0,2e4],[1,1],':k')
    if param=='ellipticity':
        plt.xlabel(r'Ellipticity', fontsize=20)
        plt.xlim([0,0.82])
        plt.ylim([-0.05,4.])

        # Plot Binney (1978) curve
        binney = np.genfromtxt('../data/binney.txt', delimiter=',')
        plt.plot(binney[:,0],binney[:,1],'k-')

    plt.legend(loc='upper left', fontsize=14, ncol=2)
    plt.ylabel(r'$v_{\mathrm{rot}}/\sigma_{\star}$', fontsize=20)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)

    if mass=='wise' and param=='mass':
        param = param+'wise'

    if inclination:
        param = param+'_inclination'

    plt.savefig((plot_path+'vsigma_'+param+'.pdf'), bbox_inches='tight')
    #plt.show()

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
    
    vsigma_void = voiddata['vsigma']
    vsigma_void_uperr = voiddata['vsigma_uperr']
    vsigma_void_loerr = voiddata['vsigma_loerr']

    if mass != 'wise':
        mass_void = voiddata['massSDSS']
        mass_void_err = np.asarray([np.asarray(voiddata['massSDSS'] - voiddata['massSDSS_lo']), np.asarray(voiddata['massSDSS_hi'] - voiddata['massSDSS'])])
    else:
        mass_void = voiddata['massWISE']
        mass_void_err = np.asarray([voiddata['massWISE_lo'],voiddata['massWISE_hi']])

    dLstar_void = voiddata['dLstar']
    dLstar_void[dLstar_void<-990] = 3000

    # Make masks for void data
    good_idx = [True if ~np.any(np.isclose([dLstar_void[i], mass_void[i], voiddata['vmax'][i], voiddata['sigma'][i]], -999)) else False for i in range(len(mass_void))]

    # Make figure
    fig = plt.figure(figsize=(8,6))

    vmax = 2.5

    # Plot data from Wheeler+17
    sc = plt.scatter(dLstar_wheeler, mass_wheeler, label='Local Volume', c=vsigma_wheeler, cmap=cmr.bubblegum, vmin=0, vmax=vmax, marker='s', s=40, alpha=0.8)

    # Plot my data
    plt.scatter(dLstar_void[good_idx], mass_void[good_idx], label='Field', c=vsigma_void[good_idx], cmap=cmr.bubblegum, vmin=0, vmax=vmax, marker='o', s=40, alpha=0.8)

    # Plot cluster data
    plt.scatter(dLstar_virgo, mass_virgo, label='Cluster', c=vsigma_virgo, cmap=cmr.bubblegum, vmin=0, vmax=vmax, marker='^', s=40, alpha=0.8)
    
    #ax.errorbar(dLstar_void[good_idx], mass_void[good_idx], yerr=mass_void_err[:,good_idx], label='Field', 
    #           color=vsigma_void[good_idx], cmap=cmr.bubblegum, linestyle='None', marker='o', markersize=8, linewidth=1)

    # Add title and labels
    cbar = plt.colorbar(sc)
    cbar.set_label(r'$v_{\mathrm{rot}}/\sigma_{\star}$', fontsize=20)
    plt.ylabel(r'$\log M_{\star}$ (M$_{\odot}$)', fontsize=20)
    plt.ylim([3.25,9.7])
    plt.xlabel(r'$d_{L_{\star}}$ (kpc)', fontsize=20)
    plt.xscale('log')
    plt.xlim([0.5,255000])
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
    #print(voiddata)

    # Correct for template dispersion (~66 km/s)
    sigma = np.sqrt(np.copy(voiddata['sigma'])**2 - 66.**2)
    #print(np.shape(sigma))
    
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


if __name__ == "__main__":

    #vsigma_plot(param='dLstar', inclination=False)
    vsigma_plot(param='mass', mass='wise', inclination=False)
    #vsigma_plot(param='ellipticity', inclination=False)
    #mass_metallicity()
    #dLstar_mass()
    #vsigma_dist()
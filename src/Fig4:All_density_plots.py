import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from dgutils import colors as colortools
from scipy.interpolate import interp1d
from wetting_utils import *

rm = 2.9673;

#Compute Recoil Energy
hbar = 11606  * 6.5821 * 10**(-16) #Kelvin second
hbar2by2 = hbar ** 2/2 #Kelvin^2 second^2
mass = 4* 9.3827 * 11606 * 10**8 #Kelvin/c^2
c = 2.998 * 10**(18) #Angstrom per second
E_rec = hbar2by2/(mass*rm**2)*c**2
print (E_rec)

Rs = [4.0, 5.0, 6.0, 8.0, 10.0, 12.0]
el_order = ['Cs','Rb','K','Mg','Ne','Au','Ar','hwall']

_r = np.linspace(0,0.98,1000)
with plt.style.context('../include/aps.mplstyle'):

    figsize = plt.rcParams['figure.figsize']
    fig,axs = plt.subplots(nrows=3,ncols=2,sharex=True, sharey=False, figsize=(figsize[0],2*figsize[1]), 
                          constrained_layout=True)
    ax = list(np.atleast_1d(axs).ravel())

    for iR,R in enumerate(Rs):
        for element in elements:
            c = element_colors[element]
            fname = ψ_filename(R, element)
            with np.load(fname, 'rb') as f:
                rval = f['arr_0']
                radial = f['arr_1']
            norm = normalize_psi_PIMC(radial, rval)
            
            #ax[iR].plot(rval/R, radial, 'o', mfc=c, mec=c,label=element)
            ax[iR].plot(rval/R, radial, 'o:', mfc=colortools.get_alpha_hex(c,0.7), mec=c, color=c,
                        label=element, linewidth=0.5)
            #ρ_cubic = interp1d(rval/R, radial, kind='cubic')
            #ax[iR].plot(_r, ρ_cubic(_r), '-', color=colortools.get_alpha_hex(c,0.5, real=True),linewidth=0.5)

        ax[iR].text(0.99, 0.97, f"$R = {R}\;$"+r"${\rm \AA}$", transform=ax[iR].transAxes,ha="right", va="top")

        if iR%2==0:
            ax[iR].set_ylabel(r'$\varrho(r)\;\; [{\rm \AA}^{-3}]$')

        if iR>= len(Rs)-2:
            ax[iR].set_xlabel(r'$r/R$')

    ax[-1].set_xlim(0,0.98)

    # some manual tweaking of y-limits
    ax[3].set_ylim(0, 0.0032)
    ax[4].set_ylim(0, 0.0023)
    ax[5].set_ylim(0, 0.00185)

    handles, labels = ax[0].get_legend_handles_labels()
    order = []
    for il,l in enumerate(el_order):
        order.append(labels.index(l))
    ord_handles = [handles[idx] for idx in order]
    ord_labels = [labels[idx] for idx in order]
    ord_labels[-1] = 'hard wall'
    ax[0].legend(ord_handles,ord_labels,loc = 7, handlelength = 1.5, labelspacing=0.11)
        
    plt.savefig("../figures/radial_density_N2.pdf")

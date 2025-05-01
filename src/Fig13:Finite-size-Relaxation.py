import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.integrate as intgr
plt.style.use('aps.mplstyle')
mpl.rcParams["figure.figsize"] = [3.4039, 2.10373]

rm = 2.9673;
def normalize_psi_PIMC(psi, x):
   int_psi_square = 2*np.pi*25*intgr.simpson(y = x*psi, x = x)
   print ("Norm = " + str(int_psi_square))
   return int_psi_square

fname = "../data/Radial-wavefunction-CsR6.npz"
with np.load(fname, 'rb') as f:
    rvalCs6 = f['arr_0']
    radialCs6 = f['arr_1']

fname = "../data/Radial-wavefunction-Csz16.npz"
with np.load(fname, 'rb') as f:
    rvalCsz16 = f['arr_0']
    radialCsz16 = f['arr_1']

fname = "../data/Radial-wavefunction-Cs96.npz"
with np.load(fname, 'rb') as f:
    rvalCs96 = f['arr_0']
    radialCs96 = f['arr_1']

with plt.style.context('aps'):
   figsize = plt.rcParams['figure.figsize']
   fig,(ax1,ax2) = plt.subplots(2,1,figsize=(figsize[0],2*figsize[1]), sharey = True, constrained_layout=True)
   ax1.plot(rvalCs6,radialCs6,marker='o',label='N = 72',color='#5E4FA2',linestyle='dashed',alpha=0.8)
   ax1.plot(rvalCs96,radialCs96,marker='^',label = 'N = 96',color='#5E4FA2',linestyle='dashed',alpha=0.5)
   ax2.plot(rvalCs6,radialCs6,marker='o',label='Nz = 12',color='#5E4FA2',linestyle='dashed',alpha=0.8)
   ax2.plot(rvalCsz16,radialCsz16,marker='^',label = 'Nz = 16',color='#5E4FA2',linestyle='dashed',alpha=0.5)
   ax1.set_ylabel(r'$|\psi|^2$')
   #ax2.set_ylabel(r'$|\psi|^2$')
   ax1.set_xlabel(r'$r$ [Å]')
   ax2.set_xlabel(r'$r$ [Å]')
   ax1.set_xlim(0,5.999)
   ax2.set_xlim(0,5.999)
   ax1.legend(handlelength=1)
   ax2.legend(handlelength=1)
   plt.savefig('Finite-size-relaxation.pdf')
   plt.show()


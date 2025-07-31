import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.integrate as intgr
plt.style.use('../include/aps.mplstyle')
mpl.rcParams["figure.figsize"] = [3.4039, 2.10373]

rm = 2.9673;
def normalize_psi_PIMC(psi, x):
   int_psi_square = 2*np.pi*25*intgr.simpson(y = x*psi, x = x)
   print ("Norm = " + str(int_psi_square))
   return int_psi_square



fname = "../data/Relaxation/Radial-wavefunction-CsR6.npz"
with np.load(fname) as f:
   rval = f['arr_0']
   radial = f['arr_1']

norm = normalize_psi_PIMC(radial,rval)


f2 = open('../data/pimc/tauscale/radial-N-reduce-1a32c4e3-7ed0-463e-93dc-530e81f82cc6.dat','r')
lines = f2.readlines()
x = np.array([])
y = np.array([])
z = np.array([])
for line in lines[3:]:
    p = line.split()
    x = np.append(x,float(p[0]))
    y = np.append(y,float(p[1]))
    z = np.append(z,float(p[2]))
f2.close()

def normalize_psi_PIMC(psi, x):
   int_psi_square = 2*np.pi*25*intgr.simpson(y = x*psi, x = x)
   print(int_psi_square)
   return int_psi_square

norm = normalize_psi_PIMC(y,x)
with plt.style.context('aps'):
   figsize = plt.rcParams['figure.figsize']
   fig,ax = plt.subplots(figsize=(figsize[0],figsize[1]), constrained_layout=True)
   ax.plot(rval,radial, 'o--', label="Relaxation",color='#5E4FA2',alpha=1.0)
   ax.errorbar(x,2*y/norm,yerr=2*z/norm, linestyle="None",marker = '+', capsize=1,label='Re-normalised PIMC',color='#5E4FA2',alpha=0.7)
   ax.errorbar(x,y,yerr=z, linestyle="None",marker = '+',capsize=1,label='PIMC',color='#5E4FA2',alpha=0.5)
   ax.set_ylabel(r'$|\psi(r)|^2$')
   ax.set_xlabel(r'$r$ [Å]' )
   ax.set_xlim(0,5.9)
   ax.legend()
   plt.savefig('../figures/Pigs_comparison.pdf')
   plt.show()    



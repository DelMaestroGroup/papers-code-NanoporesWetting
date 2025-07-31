import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.integrate as intgr
plt.style.use('../include/aps.mplstyle')
mpl.rcParams["figure.figsize"] = [2*3.4039, 3*2.10373]
color_cycle = ['C2E69F','5E4Fa2', 'D7414E','F57949','3C93B8', '79C9A4', 'FDBB6C','FEEC9F', '9E0142']

rm = 2.9673;
def normalize_psi_PIMC(psi, x):
   int_psi_square = 2*np.pi*25*intgr.simpson(y = x*psi, x = x)
   print ("Norm = " + str(int_psi_square))
   return int_psi_square


def radial_ave(ψ,X,Y):

    x = X[0,:]
    y = Y[:,0]
    N = X.shape[0]
    dx = X[1,0]-X[0,0]
    dR = np.sqrt(2*dx**2)
    dϕ = 2*π/N
    r_vals = np.arange(0, R, dR)
    ϕ_vals = np.arange(0, 2*π, dϕ)

    if len(r_vals)*len(ϕ_vals) > N**2:
        print("Warning: Oversampling")

    # Initialize data on polar grid with fill values
    fill_value = -9999.0
    data_polar = fill_value*np.ones((len(r_vals), len(ϕ_vals)))

    # Define radius of influence. A nearest neighbour outside this radius will not
    # be taken into account.
    radius_of_influence = np.sqrt(0.1**2 + 0.1**2)

    # For each cell in the polar grid, find the nearest neighbour in the cartesian
    # grid. If it lies within the radius of influence, transfer the corresponding
    # data.
    for r, row_polar in zip(r_vals, range(len(r_vals))):
        for ϕ, col_polar in zip(ϕ_vals, range(len(ϕ_vals))):
            # Transform polar to cartesian
            _x = r*np.cos(ϕ)
            _y = r*np.sin(ϕ)

            # Find nearest neighbour in cartesian grid
            d = np.sqrt((_x-X)**2 + (_y-Y)**2)
            nn_row_cart, nn_col_cart = np.unravel_index(np.argmin(d), d.shape)
            dmin = d[nn_row_cart, nn_col_cart]

            # Transfer data
            if dmin <= radius_of_influence:
                data_polar[row_polar, col_polar] = ψ[nn_row_cart, nn_col_cart]

    # Mask remaining fill values
    data_polar = np.ma.masked_equal(data_polar, fill_value)

    return r_vals, np.average(data_polar,axis=1)

#Compute Recoil Energy

hbar = 11606  * 6.5821 * 10**(-16) #Kelvin second
hbar2by2 = hbar ** 2/2 #Kelvin^2 second^2
mass = 4* 9.3827 * 11606 * 10**8 #Kelvin/c^2
c = 2.998 * 10**(18) #Angstrom per second
E_rec = hbar2by2/(mass*rm**2)*c**2
print (E_rec)

figsize = plt.rcParams['figure.figsize']
fig,((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2,figsize=(figsize[0],figsize[1]), constrained_layout=True)
ax1.set_prop_cycle('color',color_cycle)
ax2.set_prop_cycle('color',color_cycle)
ax3.set_prop_cycle('color',color_cycle)
ax4.set_prop_cycle('color',color_cycle)
ax5.set_prop_cycle('color',color_cycle)
ax6.set_prop_cycle('color',color_cycle)

fname = "../data/Relaxation/Radial-wavefunction-KR4.npz"
with np.load(fname, 'rb') as f:
    rval = f['arr_0']
    radial = f['arr_1']
norm = normalize_psi_PIMC(radial,rval)
ax1.plot(rval,radial, 'o-', label="K",alpha=0.5)

fname = "../data/Relaxation/Radial-wavefunction-CsR4.npz"
with np.load(fname, 'rb') as f:
    rval = f['arr_0']
    radial = f['arr_1']
norm = normalize_psi_PIMC(radial,rval)
ax1.plot(rval,radial, 'o-', label="Cs",alpha=0.5)

fname = "../data/Relaxation/Radial-wavefunction-ArR4.npz"
with np.load(fname, 'rb') as f:
    rval = f['arr_0']
    radial = f['arr_1']
norm = normalize_psi_PIMC(radial,rval)
ax1.plot(rval,radial, 'o-', label="Ar",alpha=0.5)

fname = "../data/Relaxation/Radial-wavefunction-AuR4.npz"
with np.load(fname, 'rb') as f:
    rval = f['arr_0']
    radial = f['arr_1']
norm = normalize_psi_PIMC(radial,rval)
ax1.plot(rval,radial, 'o-', label="Au",alpha=0.5)

fname = "../data/Relaxation/Radial-wavefunction-RbR4.npz"
with np.load(fname, 'rb') as f:
    rval = f['arr_0']
    radial = f['arr_1']
norm = normalize_psi_PIMC(radial,rval)
ax1.plot(rval,radial, 'o-', label="Rb",alpha=0.5)

fname = "../data/Relaxation/Radial-wavefunction-MgR4.npz"
with np.load(fname, 'rb') as f:
    rval = f['arr_0']
    radial = f['arr_1']
norm = normalize_psi_PIMC(radial,rval)
ax1.plot(rval,radial, 'o-', label="Mg",alpha=0.5)

fname = "../data/Relaxation/Radial-wavefunction-NeR4.npz"
with np.load(fname, 'rb') as f:
    rval = f['arr_0']
    radial = f['arr_1']
norm = normalize_psi_PIMC(radial,rval)
ax1.plot(rval,radial, 'o-', label="Ne",alpha=0.5)

fname = "../data/Relaxation/Radial-wavefunction-hwallR4.npz"
with np.load(fname, 'rb') as f:
    rval = f['arr_0']
    radial = f['arr_1']
norm = normalize_psi_PIMC(radial,rval)
ax1.plot(rval,radial, 'ko-', label="Hardwall",alpha=0.2)
ax1.text(0.95,0.95,'R = 4 Å',ha="right",va="top",transform = ax1.transAxes)
ax1.set_ylabel(r'$|\psi(r)|^2$',fontsize=10)
ax1.set_xlabel('$r$ [Å]',fontsize=10)
ax1.set_xlim(0,3.9)
#ax1.legend(handlelength = 1.5)

fname = "../data/Relaxation/Radial-wavefunction-KR5.npz"
with np.load(fname, 'rb') as f:
    rval = f['arr_0']
    radial = f['arr_1']
norm = normalize_psi_PIMC(radial,rval)
ax2.plot(rval,radial, 'o-', label="K",alpha=0.5)

fname = "../data/Relaxation/Radial-wavefunction-CsR5.npz"
with np.load(fname, 'rb') as f:
    rval = f['arr_0']
    radial = f['arr_1']
norm = normalize_psi_PIMC(radial,rval)
ax2.plot(rval,radial, 'o-', label="Cs",alpha=0.5)

fname = "../data/Relaxation/Radial-wavefunction-ArR5.npz"
with np.load(fname, 'rb') as f:
    rval = f['arr_0']
    radial = f['arr_1']
norm = normalize_psi_PIMC(radial,rval)
ax2.plot(rval,radial, 'o-', label="Ar",alpha=0.5)

fname = "../data/Relaxation/Radial-wavefunction-AuR5.npz"
with np.load(fname, 'rb') as f:
    rval = f['arr_0']
    radial = f['arr_1']
norm = normalize_psi_PIMC(radial,rval)
ax2.plot(rval,radial, 'o-', label="Au",alpha=0.5)

fname = "../data/Relaxation/Radial-wavefunction-RbR5.npz"
with np.load(fname, 'rb') as f:
    rval = f['arr_0']
    radial = f['arr_1']
norm = normalize_psi_PIMC(radial,rval)
ax2.plot(rval,radial, 'o-', label="Rb",alpha=0.5)

fname = "../data/Relaxation/Radial-wavefunction-MgR5.npz"
with np.load(fname, 'rb') as f:
    rval = f['arr_0']
    radial = f['arr_1']
norm = normalize_psi_PIMC(radial,rval)
ax2.plot(rval,radial, 'o-', label="Mg",alpha=0.5)

fname = "../data/Relaxation/Radial-wavefunction-NeR5.npz"
with np.load(fname, 'rb') as f:
    rval = f['arr_0']
    radial = f['arr_1']
norm = normalize_psi_PIMC(radial,rval)
ax2.plot(rval,radial, 'o-', label="Ne",alpha=0.5)

fname = "../data/Relaxation/Radial-wavefunction-hwallR5.npz"
with np.load(fname, 'rb') as f:
    rval = f['arr_0']
    radial = f['arr_1']
norm = normalize_psi_PIMC(radial,rval)
ax2.plot(rval,radial, 'ko-', label="Hardwall",alpha=0.5)

ax2.text(0.95,0.95,'R = 5 Å',ha="right",va="top",transform = ax2.transAxes)
ax2.set_ylabel(r'$|\psi(r)|^2$',fontsize=10)
ax2.set_xlabel('$r$ [Å]',fontsize=10)
handles, labels = ax2.get_legend_handles_labels()
order = [1,4,0,5,6,3,2,7]
ax2.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc = 7, handlelength = 1.5)
ax2.set_xlim(0,4.9)

fname = "../data/Relaxation/Radial-wavefunction-KR6.npz"
with np.load(fname, 'rb') as f:
    rval = f['arr_0']
    radial = f['arr_1']
norm = normalize_psi_PIMC(radial,rval)
ax3.plot(rval,radial, 'o-', label="K",alpha=0.5)

fname = "../data/Relaxation/Radial-wavefunction-CsR6.npz"
with np.load(fname, 'rb') as f:
    rval = f['arr_0']
    radial = f['arr_1']
norm = normalize_psi_PIMC(radial,rval)
ax3.plot(rval,radial, 'o-', label="Cs",alpha=0.5)

fname = "../data/Relaxation/Radial-wavefunction-ArR6.npz"
with np.load(fname, 'rb') as f:
    rval = f['arr_0']
    radial = f['arr_1']
norm = normalize_psi_PIMC(radial,rval)
ax3.plot(rval,radial, 'o-', label="Ar",alpha=0.5)

fname = "../data/Relaxation/Radial-wavefunction-AuR6.npz"
with np.load(fname, 'rb') as f:
    rval = f['arr_0']
    radial = f['arr_1']
norm = normalize_psi_PIMC(radial,rval)
ax3.plot(rval,radial, 'o-', label="Au",alpha=0.5)

fname = "../data/Relaxation/Radial-wavefunction-RbR6.npz"
with np.load(fname, 'rb') as f:
    rval = f['arr_0']
    radial = f['arr_1']
norm = normalize_psi_PIMC(radial,rval)
ax3.plot(rval,radial, 'o-', label="Rb",alpha=0.5)

fname = "../data/Relaxation/Radial-wavefunction-MgR6.npz"
with np.load(fname, 'rb') as f:
    rval = f['arr_0']
    radial = f['arr_1']
norm = normalize_psi_PIMC(radial,rval)
ax3.plot(rval,radial, 'o-', label="Mg",alpha=0.5)

fname = "../data/Relaxation/Radial-wavefunction-NeR6.npz"
with np.load(fname, 'rb') as f:
    rval = f['arr_0']
    radial = f['arr_1']
norm = normalize_psi_PIMC(radial,rval)
ax3.plot(rval,radial, 'o-', label="Ne",alpha=0.5)

fname = "../data/Relaxation/Radial-wavefunction-hwallR6.npz"
with np.load(fname, 'rb') as f:
    rval = f['arr_0']
    radial = f['arr_1']
norm = normalize_psi_PIMC(radial,rval)
ax3.plot(rval,radial, 'ko-', label="Hardwall",alpha=0.5)

ax3.text(0.95,0.95,'R = 6 Å',ha="right",va="top",transform = ax3.transAxes)
ax3.set_ylabel(r'$|\psi(r)|^2$',fontsize=10)
ax3.set_xlabel('r (Å)',fontsize=10)
ax3.set_xlim(0,5.9)
#ax3.legend(handlelength = 1.5)

fname = "../data/Relaxation/Radial-wavefunction-KR8.npz"
with np.load(fname, 'rb') as f:
    rval = f['arr_0']
    radial = f['arr_1']
norm = normalize_psi_PIMC(radial,rval)
ax4.plot(rval,radial, 'o-', label="K",alpha=0.5)

fname = "../data/Relaxation/Radial-wavefunction-CsR8.npz"
with np.load(fname, 'rb') as f:
    rval = f['arr_0']
    radial = f['arr_1']
norm = normalize_psi_PIMC(radial,rval)
ax4.plot(rval,radial, 'o-', label="Cs",alpha=0.5)

fname = "../data/Relaxation/Radial-wavefunction-ArR8.npz"
with np.load(fname, 'rb') as f:
    rval = f['arr_0']
    radial = f['arr_1']
norm = normalize_psi_PIMC(radial,rval)
ax4.plot(rval,radial, 'o-', label="Ar",alpha=0.5)

fname = "../data/Relaxation/Radial-wavefunction-AuR8.npz"
with np.load(fname, 'rb') as f:
    rval = f['arr_0']
    radial = f['arr_1']
norm = normalize_psi_PIMC(radial,rval)
ax4.plot(rval,radial, 'o-', label="Au",alpha=0.5)

fname = "../data/Relaxation/Radial-wavefunction-RbR8.npz"
with np.load(fname, 'rb') as f:
    rval = f['arr_0']
    radial = f['arr_1']
norm = normalize_psi_PIMC(radial,rval)
ax4.plot(rval,radial, 'o-', label="Rb",alpha=0.5)

fname = "../data/Relaxation/Radial-wavefunction-MgR8.npz"
with np.load(fname, 'rb') as f:
    rval = f['arr_0']
    radial = f['arr_1']
norm = normalize_psi_PIMC(radial,rval)
ax4.plot(rval,radial, 'o-', label="Mg",alpha=0.5)

fname = "../data/Relaxation/Radial-wavefunction-NeR8.npz"
with np.load(fname, 'rb') as f:
    rval = f['arr_0']
    radial = f['arr_1']
norm = normalize_psi_PIMC(radial,rval)
ax4.plot(rval,radial, 'o-', label="Ne",alpha=0.5)

fname = "../data/Relaxation/Radial-wavefunction-hwallR8.npz"
with np.load(fname, 'rb') as f:
    rval = f['arr_0']
    radial = f['arr_1']
norm = normalize_psi_PIMC(radial,rval)
ax4.plot(rval,radial, 'ko-', label="Hardwall",alpha=0.5)

ax4.text(0.95,0.95,'R = 8 Å',ha="right",va="top",transform = ax4.transAxes)
ax4.set_ylabel(r'$|\psi(r)|^2$',fontsize=10)
ax4.set_xlabel('$r$ [Å]',fontsize=10)
ax4.set_xlim(0,7.9)
#ax4.legend(handlelength = 1.5)

fname = "../data/Relaxation/Radial-wavefunction-KR10.npz"
with np.load(fname, 'rb') as f:
    rval = f['arr_0']
    radial = f['arr_1']
norm = normalize_psi_PIMC(radial,rval)
ax5.plot(rval,radial, 'o-', label="K",alpha=0.5)

fname = "../data/Relaxation/Radial-wavefunction-CsR10.npz"
with np.load(fname, 'rb') as f:
    rval = f['arr_0']
    radial = f['arr_1']
norm = normalize_psi_PIMC(radial,rval)
ax5.plot(rval,radial, 'o-', label="Cs",alpha=0.5)

fname = "../data/Relaxation/Radial-wavefunction-ArR10.npz"
with np.load(fname, 'rb') as f:
    rval = f['arr_0']
    radial = f['arr_1']
norm = normalize_psi_PIMC(radial,rval)
ax5.plot(rval,radial, 'o-', label="Ar",alpha=0.5)

fname = "../data/Relaxation/Radial-wavefunction-AuR10.npz"
with np.load(fname, 'rb') as f:
    rval = f['arr_0']
    radial = f['arr_1']
norm = normalize_psi_PIMC(radial,rval)
ax5.plot(rval,radial, 'o-', label="Au",alpha=0.5)

fname = "../data/Relaxation/Radial-wavefunction-RbR10.npz"
with np.load(fname, 'rb') as f:
    rval = f['arr_0']
    radial = f['arr_1']
norm = normalize_psi_PIMC(radial,rval)
ax5.plot(rval,radial, 'o-', label="Rb",alpha=0.5)

fname = "../data/Relaxation/Radial-wavefunction-MgR10.npz"
with np.load(fname, 'rb') as f:
    rval = f['arr_0']
    radial = f['arr_1']
norm = normalize_psi_PIMC(radial,rval)
ax5.plot(rval,radial, 'o-', label="Mg",alpha=0.5)

fname = "../data/Relaxation/Radial-wavefunction-NeR10.npz"
with np.load(fname, 'rb') as f:
    rval = f['arr_0']
    radial = f['arr_1']
norm = normalize_psi_PIMC(radial,rval)
ax5.plot(rval,radial, 'o-', label="Ne",alpha=0.5)

fname = "../data/Relaxation/Radial-wavefunction-hwallR10.npz"
with np.load(fname, 'rb') as f:
    rval = f['arr_0']
    radial = f['arr_1']
norm = normalize_psi_PIMC(radial,rval)
ax5.plot(rval,radial, 'ko-', label="Hardwall",alpha=0.5)

ax5.text(0.95,0.95,'R = 10 Å',ha="right",va="top",transform = ax5.transAxes)
ax5.set_ylabel(r'$|\psi(r)|^2$',fontsize=10)
ax5.set_xlabel('$r$ [Å]',fontsize=10)
ax5.set_xlim(0,9.9)
#ax5.legend(handlelength = 1.5)

fname = "../data/Relaxation/Radial-wavefunction-KR12.npz"
with np.load(fname, 'rb') as f:
    rval = f['arr_0']
    radial = f['arr_1']
norm = normalize_psi_PIMC(radial,rval)
ax6.plot(rval,radial, 'o-', label="K",alpha=0.5)

fname = "../data/Relaxation/Radial-wavefunction-CsR12.npz"
with np.load(fname, 'rb') as f:
    rval = f['arr_0']
    radial = f['arr_1']
norm = normalize_psi_PIMC(radial,rval)
ax6.plot(rval,radial, 'o-', label="Cs",alpha=0.5)

fname = "../data/Relaxation/Radial-wavefunction-ArR12.npz"
with np.load(fname, 'rb') as f:
    rval = f['arr_0']
    radial = f['arr_1']
norm = normalize_psi_PIMC(radial,rval)
ax6.plot(rval,radial, 'o-', label="Ar",alpha=0.5)

fname = "../data/Relaxation/Radial-wavefunction-AuR12.npz"
with np.load(fname, 'rb') as f:
    rval = f['arr_0']
    radial = f['arr_1']
norm = normalize_psi_PIMC(radial,rval)
ax6.plot(rval,radial, 'o-', label="Au",alpha=0.5)

fname = "../data/Relaxation/Radial-wavefunction-RbR12.npz"
with np.load(fname, 'rb') as f:
    rval = f['arr_0']
    radial = f['arr_1']
norm = normalize_psi_PIMC(radial,rval)
ax6.plot(rval,radial, 'o-', label="Rb",alpha=0.5)

fname = "../data/Relaxation/Radial-wavefunction-MgR12.npz"
with np.load(fname, 'rb') as f:
    rval = f['arr_0']
    radial = f['arr_1']
norm = normalize_psi_PIMC(radial,rval)
ax6.plot(rval,radial, 'o-', label="Mg",alpha=0.5)

fname = "../data/Relaxation/Radial-wavefunction-NeR12.npz"
with np.load(fname, 'rb') as f:
    rval = f['arr_0']
    radial = f['arr_1']
norm = normalize_psi_PIMC(radial,rval)
ax6.plot(rval,radial, 'o-', label="Ne",alpha=0.5)

fname = "../data/Relaxation/Radial-wavefunction-hwallR12.npz"
with np.load(fname, 'rb') as f:
    rval = f['arr_0']
    radial = f['arr_1']
norm = normalize_psi_PIMC(radial,rval)
ax6.plot(rval,radial, 'ko-', label="Hardwall",alpha=0.5)

ax6.text(0.95,0.95,'R = 12 Å',ha="right",va="top",transform = ax6.transAxes)
ax6.set_ylabel(r'$|\psi(r)|^2$',fontsize=10)
ax6.set_xlabel('$r$ [Å]',fontsize=10)
ax6.set_xlim(0,11.9)
#ax6.legend(handlelength = 1.5)
plt.savefig("../figures/Density_plot.pdf")
plt.show()




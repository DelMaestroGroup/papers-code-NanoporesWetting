import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.integrate as intgr
plt.style.use('aps.mplstyle')
mpl.rcParams["figure.figsize"] = [2*3.4039, 3*2.10373]
color_cycle = ['C2E69F','5E4Fa2', 'D7414E','F57949','3C93B8', '79C9A4', 'FDBB6C','FEEC9F', '9E0142']

rm = 2.9673;
def normalize_psi_PIMC(psi, x):
   int_psi_square = 2*np.pi*25*intgr.simps(x*psi, x)
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

R = 4
Lx = 2*R; Ly = 2*R; Lz = 25
Lxnd = Lx/rm;
Lynd = Ly/rm;
Lznd = Lz/rm;
nx = 72
ny = 72
nz = 12
dx = Lxnd/nx
dy = Lynd/ny
dz = Lznd/nz

xmin = -Lxnd/2; xmax = Lxnd/2; ymin = -Lynd/2; ymax = Lynd/2; zmin = -Lznd/2; zmax = Lznd/2;

x = dx * np.concatenate((np.arange(-nx/2,0,1),np.arange(0,nx/2,1)))
y = dy * np.arange(-ny/2,ny/2,1)
z = dz * np.arange(-nz/2, nz/2,1)

x1, x2, y1, y2, zr = np.meshgrid(x,x,y,y,z, indexing='ij')
xrad,yrad = np.meshgrid(x,y, indexing='ij')

fname = "R4/Final-wavefunction-KR4.npy"
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
rho_rad = []
π = np.pi
#R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
ind = rval < R
rval = rval[ind]
radial = radial[ind]
norm = normalize_psi_PIMC(radial,rval)
ax1.plot(rval,radial, '*-', label="K")

fname = "R4/Final-wavefunction-CsR4.npy"
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
rho_rad = []
π = np.pi
#R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
ind = rval < R
rval = rval[ind]
radial = radial[ind]
norm = normalize_psi_PIMC(radial,rval)
ax1.plot(rval,radial, '*-', label="Cs")

fname = "R4/Final-wavefunction-ArR4.npy"
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
rho_rad = []
π = np.pi
#R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
ind = rval < R
rval = rval[ind]
radial = radial[ind]
norm = normalize_psi_PIMC(radial,rval)
ax1.plot(rval,radial, '*-', label="Ar")

fname = "R4/Final-wavefunction-AuR4.npy"
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
rho_rad = []
π = np.pi
#R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
ind = rval < R
rval = rval[ind]
radial = radial[ind]
norm = normalize_psi_PIMC(radial,rval)
ax1.plot(rval,radial, '*-', label="Au")

fname = "R4/Final-wavefunction-RbR4.npy"
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
rho_rad = []
π = np.pi
#R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
ind = rval < R
rval = rval[ind]
radial = radial[ind]
norm = normalize_psi_PIMC(radial,rval)
ax1.plot(rval,radial, '*-', label="Rb")

fname = "R4/Final-wavefunction-MgR4.npy"
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
rho_rad = []
π = np.pi
#R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
ind = rval < R
rval = rval[ind]
radial = radial[ind]
norm = normalize_psi_PIMC(radial,rval)
ax1.plot(rval,radial, '*-', label="Mg")

fname = "R4/Final-wavefunction-NeR4.npy"
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
rho_rad = []
π = np.pi
#R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
ind = rval < R
rval = rval[ind]
radial = radial[ind]
norm = normalize_psi_PIMC(radial,rval)
ax1.plot(rval,radial, '*-', label="Ne")

fname = "R4/Final-wavefunction-hwallR4.npy"
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
rho_rad = []
π = np.pi
#R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
ind = rval < R
rval = rval[ind]
radial = radial[ind]
norm = normalize_psi_PIMC(radial,rval)
ax1.plot(rval,radial, 'k-', label="Hardwall")

ax1.text(0.95,0.95,'R = 4 Å',ha="right",va="top",transform = ax1.transAxes)
ax1.set_ylabel(r'$|\psi(r)|^2$')
ax1.set_xlabel('r (Å)')
#ax1.legend(handlelength = 1.5)


R = 5
Lx = 2*R; Ly = 2*R; Lz = 25
Lxnd = Lx/rm;
Lynd = Ly/rm;
Lznd = Lz/rm;
nx = 72
ny = 72
nz = 12
dx = Lxnd/nx
dy = Lynd/ny
dz = Lznd/nz

xmin = -Lxnd/2; xmax = Lxnd/2; ymin = -Lynd/2; ymax = Lynd/2; zmin = -Lznd/2; zmax = Lznd/2;

x = dx * np.concatenate((np.arange(-nx/2,0,1),np.arange(0,nx/2,1)))
y = dy * np.arange(-ny/2,ny/2,1)
z = dz * np.arange(-nz/2, nz/2,1)

x1, x2, y1, y2, zr = np.meshgrid(x,x,y,y,z, indexing='ij')
xrad,yrad = np.meshgrid(x,y, indexing='ij')

fname = "R5/Final-wavefunction-KR5.npy"
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
rho_rad = []
π = np.pi
#R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
ind = rval < R
rval = rval[ind]
radial = radial[ind]
norm = normalize_psi_PIMC(radial,rval)
ax2.plot(rval,radial, '*-', label="K")

fname = "R5/Final-wavefunction-CsR5.npy"
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
rho_rad = []
π = np.pi
#R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
ind = rval < R
rval = rval[ind]
radial = radial[ind]
norm = normalize_psi_PIMC(radial,rval)
ax2.plot(rval,radial, '*-', label="Cs")

fname = "R5/Final-wavefunction-ArR5.npy"
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
rho_rad = []
π = np.pi
#R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
ind = rval < R
rval = rval[ind]
radial = radial[ind]
norm = normalize_psi_PIMC(radial,rval)
ax2.plot(rval,radial, '*-', label="Ar")

fname = "R5/Final-wavefunction-AuR5.npy"
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
rho_rad = []
π = np.pi
#R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
ind = rval < R
rval = rval[ind]
radial = radial[ind]
norm = normalize_psi_PIMC(radial,rval)
ax2.plot(rval,radial, '*-', label="Au")

fname = "R5/Final-wavefunction-RbR5.npy"
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
rho_rad = []
π = np.pi
#R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
ind = rval < R
rval = rval[ind]
radial = radial[ind]
norm = normalize_psi_PIMC(radial,rval)
ax2.plot(rval,radial, '*-', label="Rb")

fname = "R5/Final-wavefunction-MgR5.npy"
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
rho_rad = []
π = np.pi
#R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
ind = rval < R
rval = rval[ind]
radial = radial[ind]
norm = normalize_psi_PIMC(radial,rval)
ax2.plot(rval,radial, '*-', label="Mg")

fname = "R5/Final-wavefunction-NeR5.npy"
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
rho_rad = []
π = np.pi
#R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
ind = rval < R
rval = rval[ind]
radial = radial[ind]
norm = normalize_psi_PIMC(radial,rval)
ax2.plot(rval,radial, '*-', label="Ne")

fname = "R5/Final-wavefunction-hwallR5.npy"
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
rho_rad = []
π = np.pi
#R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
ind = rval < R
rval = rval[ind]
radial = radial[ind]
norm = normalize_psi_PIMC(radial,rval)
ax2.plot(rval,radial, 'k-', label="Hardwall")

ax2.text(0.95,0.95,'R = 5 Å',ha="right",va="top",transform = ax2.transAxes)
ax2.set_ylabel(r'$|\psi(r)|^2$')
ax2.set_xlabel('r (Å)')
handles, labels = ax2.get_legend_handles_labels()
order = [1,4,0,5,6,3,2,7]
ax2.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc = 7, handlelength = 1.5)


R = 6
Lx = 2*R; Ly = 2*R; Lz = 25
Lxnd = Lx/rm;
Lynd = Ly/rm;
Lznd = Lz/rm;
nx = 72
ny = 72
nz = 12
dx = Lxnd/nx
dy = Lynd/ny
dz = Lznd/nz

xmin = -Lxnd/2; xmax = Lxnd/2; ymin = -Lynd/2; ymax = Lynd/2; zmin = -Lznd/2; zmax = Lznd/2;

x = dx * np.concatenate((np.arange(-nx/2,0,1),np.arange(0,nx/2,1)))
y = dy * np.arange(-ny/2,ny/2,1)
z = dz * np.arange(-nz/2, nz/2,1)

x1, x2, y1, y2, zr = np.meshgrid(x,x,y,y,z, indexing='ij')
xrad,yrad = np.meshgrid(x,y, indexing='ij')

fname = "R6/Final-wavefunction-KR6.npy"
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
rho_rad = []
π = np.pi
#R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
ind = rval < R
rval = rval[ind]
radial = radial[ind]
norm = normalize_psi_PIMC(radial,rval)
ax3.plot(rval,radial, '*-', label="K")


fname = "R6/Final-wavefunction-CsR6.npy"
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
rho_rad = []
π = np.pi
#R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
ind = rval < R
rval = rval[ind]
radial = radial[ind]
norm = normalize_psi_PIMC(radial,rval)
ax3.plot(rval,radial, '*-', label="Cs")

fname = "R6/Final-wavefunction-ArR6.npy"
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
rho_rad = []
π = np.pi
#R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
ind = rval < R
rval = rval[ind]
radial = radial[ind]
norm = normalize_psi_PIMC(radial,rval)
ax3.plot(rval,radial, '*-', label="Ar")

fname = "R6/Final-wavefunction-AuR6.npy"
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
rho_rad = []
π = np.pi
#R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
ind = rval < R
rval = rval[ind]
radial = radial[ind]
norm = normalize_psi_PIMC(radial,rval)
ax3.plot(rval,radial, '*-', label="Au")

fname = "R6/Final-wavefunction-RbR6.npy"
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
rho_rad = []
π = np.pi
#R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
ind = rval < R
rval = rval[ind]
radial = radial[ind]
norm = normalize_psi_PIMC(radial,rval)
ax3.plot(rval,radial, '*-', label="Rb")

fname = "R6/Final-wavefunction-MgR6.npy"
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
rho_rad = []
π = np.pi
#R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
ind = rval < R
rval = rval[ind]
radial = radial[ind]
norm = normalize_psi_PIMC(radial,rval)
ax3.plot(rval,radial, '*-', label="Mg")

fname = "R6/Final-wavefunction-NeR6.npy"
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
rho_rad = []
π = np.pi
#R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
ind = rval < R
rval = rval[ind]
radial = radial[ind]
norm = normalize_psi_PIMC(radial,rval)
ax3.plot(rval,radial, '*-', label="Ne")

fname = "R6/Final-wavefunction-hwallR6.npy"
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
rho_rad = []
π = np.pi
#R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
ind = rval < R
rval = rval[ind]
radial = radial[ind]
norm = normalize_psi_PIMC(radial,rval)
ax3.plot(rval,radial, 'k-', label="Hardwall")

ax3.text(0.95,0.95,'R = 6 Å',ha="right",va="top",transform = ax3.transAxes)
ax3.set_ylabel(r'$|\psi(r)|^2$')
ax3.set_xlabel('r (Å)')
#ax3.legend(handlelength = 1.5)



R = 8
Lx = 2*R; Ly = 2*R; Lz = 25
Lxnd = Lx/rm;
Lynd = Ly/rm;
Lznd = Lz/rm;
nx = 72
ny = 72
nz = 12
dx = Lxnd/nx
dy = Lynd/ny
dz = Lznd/nz

xmin = -Lxnd/2; xmax = Lxnd/2; ymin = -Lynd/2; ymax = Lynd/2; zmin = -Lznd/2; zmax = Lznd/2;

x = dx * np.concatenate((np.arange(-nx/2,0,1),np.arange(0,nx/2,1)))
y = dy * np.arange(-ny/2,ny/2,1)
z = dz * np.arange(-nz/2, nz/2,1)

x1, x2, y1, y2, zr = np.meshgrid(x,x,y,y,z, indexing='ij')
xrad,yrad = np.meshgrid(x,y, indexing='ij')

fname = "R8/Final-wavefunction-KR8.npy"
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
rho_rad = []
π = np.pi
#R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
ind = rval < R
rval = rval[ind]
radial = radial[ind]
norm = normalize_psi_PIMC(radial,rval)
ax4.plot(rval,radial, '*-', label="K")


fname = "R8/Final-wavefunction-CsR8.npy"
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
rho_rad = []
π = np.pi
#R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
ind = rval < R
rval = rval[ind]
radial = radial[ind]
norm = normalize_psi_PIMC(radial,rval)
ax4.plot(rval,radial, '*-', label="Cs")

fname = "R8/Final-wavefunction-ArR8.npy"
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
rho_rad = []
π = np.pi
#R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
ind = rval < R
rval = rval[ind]
radial = radial[ind]
norm = normalize_psi_PIMC(radial,rval)
ax4.plot(rval,radial, '*-', label="Ar")

#fname = "R8/Final-wavefunction-AuR8.npy"
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
rho_rad = []
π = np.pi
R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
ind = rval < R
rval = rval[ind]
radial = radial[ind]
norm = normalize_psi_PIMC(radial,rval)
ax4.plot(rval,radial, '*-', label="Au")

fname = "R8/Final-wavefunction-RbR8.npy"
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
rho_rad = []
π = np.pi
#R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
ind = rval < R
rval = rval[ind]
radial = radial[ind]
norm = normalize_psi_PIMC(radial,rval)
ax4.plot(rval,radial, '*-', label="Rb")

fname = "R8/Final-wavefunction-MgR8.npy"
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
rho_rad = []
π = np.pi
#R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
ind = rval < R
rval = rval[ind]
radial = radial[ind]
norm = normalize_psi_PIMC(radial,rval)
ax4.plot(rval,radial, '*-', label="Mg")

fname = "R8/Final-wavefunction-NeR8.npy"
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
rho_rad = []
π = np.pi
#R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
ind = rval < R
rval = rval[ind]
radial = radial[ind]
norm = normalize_psi_PIMC(radial,rval)
ax4.plot(rval,radial, '*-', label="Ne")

fname = "R8/Final-wavefunction-hwallR8.npy"
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
rho_rad = []
π = np.pi
#R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
ind = rval < R
rval = rval[ind]
radial = radial[ind]
norm = normalize_psi_PIMC(radial,rval)
ax4.plot(rval,radial, 'k-', label="Hardwall")

ax4.text(0.95,0.95,'R = 8 Å',ha="right",va="top",transform = ax4.transAxes)
ax4.set_ylabel(r'$|\psi(r)|^2$')
ax4.set_xlabel('r (Å)')
#ax4.legend(handlelength = 1.5)

R = 10
Lx = 2*R; Ly = 2*R; Lz = 25
Lxnd = Lx/rm;
Lynd = Ly/rm;
Lznd = Lz/rm;
nx = 72
ny = 72
nz = 12
dx = Lxnd/nx
dy = Lynd/ny
dz = Lznd/nz

xmin = -Lxnd/2; xmax = Lxnd/2; ymin = -Lynd/2; ymax = Lynd/2; zmin = -Lznd/2; zmax = Lznd/2;

x = dx * np.concatenate((np.arange(-nx/2,0,1),np.arange(0,nx/2,1)))
y = dy * np.arange(-ny/2,ny/2,1)
z = dz * np.arange(-nz/2, nz/2,1)

x1, x2, y1, y2, zr = np.meshgrid(x,x,y,y,z, indexing='ij')
xrad,yrad = np.meshgrid(x,y, indexing='ij')

fname = "R10/Final-wavefunction-KR10.npy"
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
rho_rad = []
π = np.pi
#R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
ind = rval < R
rval = rval[ind]
radial = radial[ind]
norm = normalize_psi_PIMC(radial,rval)
ax5.plot(rval,radial, '*-', label="K")


fname = "R10/Final-wavefunction-CsR10.npy"
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
rho_rad = []
π = np.pi
#R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
ind = rval < R
rval = rval[ind]
radial = radial[ind]
norm = normalize_psi_PIMC(radial,rval)
ax5.plot(rval,radial, '*-', label="Cs")

fname = "R10/Final-wavefunction-ArR10.npy"
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
rho_rad = []
π = np.pi
#R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
ind = rval < R
rval = rval[ind]
radial = radial[ind]
norm = normalize_psi_PIMC(radial,rval)
ax5.plot(rval,radial, '*-', label="Ar")

fname = "R10/Final-wavefunction-AuR10.npy"
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
rho_rad = []
π = np.pi
#R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
ind = rval < R
rval = rval[ind]
radial = radial[ind]
norm = normalize_psi_PIMC(radial,rval)
ax5.plot(rval,radial, '*-', label="Au")

fname = "R10/Final-wavefunction-RbR10.npy"
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
rho_rad = []
π = np.pi
#R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
ind = rval < R
rval = rval[ind]
radial = radial[ind]
norm = normalize_psi_PIMC(radial,rval)
ax5.plot(rval,radial, '*-', label="Rb")

fname = "R10/Final-wavefunction-MgR10.npy"
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
rho_rad = []
π = np.pi
#R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
ind = rval < R
rval = rval[ind]
radial = radial[ind]
norm = normalize_psi_PIMC(radial,rval)
ax5.plot(rval,radial, '*-', label="Mg")

fname = "R10/Final-wavefunction-NeR10.npy"
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
rho_rad = []
π = np.pi
#R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
ind = rval < R
rval = rval[ind]
radial = radial[ind]
norm = normalize_psi_PIMC(radial,rval)
ax5.plot(rval,radial, '*-', label="Ne")

fname = "R10/Final-wavefunction-hwallR10.npy"
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
rho_rad = []
π = np.pi
#R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
ind = rval < R
rval = rval[ind]
radial = radial[ind]
norm = normalize_psi_PIMC(radial,rval)
ax5.plot(rval,radial, 'k-', label="Hardwall")

ax5.text(0.95,0.95,'R = 10 Å',ha="right",va="top",transform = ax5.transAxes)
ax5.set_ylabel(r'$|\psi(r)|^2$')
ax5.set_xlabel('r (Å)')
#ax5.legend(handlelength = 1.5)

R = 12
Lx = 2*R; Ly = 2*R; Lz = 25
Lxnd = Lx/rm;
Lynd = Ly/rm;
Lznd = Lz/rm;
nx = 72
ny = 72
nz = 12
dx = Lxnd/nx
dy = Lynd/ny
dz = Lznd/nz

xmin = -Lxnd/2; xmax = Lxnd/2; ymin = -Lynd/2; ymax = Lynd/2; zmin = -Lznd/2; zmax = Lznd/2;

x = dx * np.concatenate((np.arange(-nx/2,0,1),np.arange(0,nx/2,1)))
y = dy * np.arange(-ny/2,ny/2,1)
z = dz * np.arange(-nz/2, nz/2,1)

x1, x2, y1, y2, zr = np.meshgrid(x,x,y,y,z, indexing='ij')
xrad,yrad = np.meshgrid(x,y, indexing='ij')

fname = "R12/Final-wavefunction-KR12.npy"
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
rho_rad = []
π = np.pi
#R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
ind = rval < R
rval = rval[ind]
radial = radial[ind]
norm = normalize_psi_PIMC(radial,rval)
ax6.plot(rval,radial, '*-', label="K")

fname = "R12/Final-wavefunction-CsR12.npy"
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
rho_rad = []
π = np.pi
#R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
ind = rval < R
rval = rval[ind]
radial = radial[ind]
norm = normalize_psi_PIMC(radial,rval)
ax6.plot(rval,radial, '*-', label="Cs")

fname = "R12/Final-wavefunction-ArR12.npy"
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
rho_rad = []
π = np.pi
#R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
ind = rval < R
rval = rval[ind]
radial = radial[ind]
norm = normalize_psi_PIMC(radial,rval)
ax6.plot(rval,radial, '*-', label="Ar")

fname = "R12/Final-wavefunction-AuR12.npy"
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
rho_rad = []
π = np.pi
#R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
ind = rval < R
rval = rval[ind]
radial = radial[ind]
norm = normalize_psi_PIMC(radial,rval)
ax6.plot(rval,radial, '*-', label="Au")

fname = "R12/Final-wavefunction-RbR12.npy"
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
rho_rad = []
π = np.pi
#R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
ind = rval < R
rval = rval[ind]
radial = radial[ind]
norm = normalize_psi_PIMC(radial,rval)
ax6.plot(rval,radial, '*-', label="Rb")

fname = "R12/Final-wavefunction-MgR12.npy"
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
rho_rad = []
π = np.pi
#R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
ind = rval < R
rval = rval[ind]
radial = radial[ind]
norm = normalize_psi_PIMC(radial,rval)
ax6.plot(rval,radial, '*-', label="Mg")

fname = "R12/Final-wavefunction-NeR12.npy"
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
rho_rad = []
π = np.pi
#R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
ind = rval < R
rval = rval[ind]
radial = radial[ind]
norm = normalize_psi_PIMC(radial,rval)
ax6.plot(rval,radial, '*-', label="Ne")

fname = "R12/Final-wavefunction-hwallR12.npy"
with open(fname, 'rb') as f:
    psif = np.load(f)/np.sqrt((rm**5))
#psif = np.load(f)/(dx*dy*np.sqrt(dz*rm)*rm*rm)
print(psif.shape)
#print(np.sum(np.abs(psif)**2)*(dx*dx*dy*dy*dz)*(rm**5))
p = np.sum(np.abs(np.multiply(np.conj(psif),psif)),axis=(1,3))*dx*dy*rm*rm
rho_rad = []
π = np.pi
#R = np.sqrt(np.max(xrad)**2 +np.max(yrad)**2)*rm
for iz in range(nz):
   rval,rho = radial_ave(p[:,:,iz], xrad*rm, yrad*rm)
   rho_rad.append(rho)
rho_rad = np.asarray(rho_rad)
radial = np.mean(rho_rad, axis = 0)
rval = rval
ind = rval < R
rval = rval[ind]
radial = radial[ind]
norm = normalize_psi_PIMC(radial,rval)
ax6.plot(rval,radial, 'k-', label="Hardwall")

ax6.text(0.95,0.95,'R = 12 Å',ha="right",va="top",transform = ax6.transAxes)
ax6.set_ylabel(r'$|\psi(r)|^2$')
ax6.set_xlabel('r (Å)')
#ax6.legend(handlelength = 1.5)
plt.savefig("Density_plot.pdf")
plt.show()




# pre-plating nanopores analysis helper utilities

import os
import numpy as np
π = np.pi
import scipy.integrate as intgr
import scipy.special as scp


elements = ['Ar', 'Mg', 'Cs', 'K', 'Au', 'Rb', 'Mg', 'Ne', 'hwall']
element_colors = {'Ar':'#D7414E', 'K':'#79C9A4', 'Cs':"#5E4FA2", 'Mg':'#C2E69F', 'Au':'#F57949', 
                  'Rb':'#3C93B8', 'Ne':'#FDBB6C', 'hwall':"#858585"}

σ = {'Cs': 5.44, 'Ar':  3.02265, 'Au':  3.305, 'K': 5.14, 'Mg': 3.885, 'Rb': 5.414, 'Ne': 2.695}
ɛ = {'Cs': 1.359, 'Ar':  36.136, 'Au':  19.59, 'K': 1.512, 'Mg': 5.661, 'Rb': 1.251, 'Ne': 19.75}
n_density = {'Cs': 0.0091 ,'Ar':  0.0265, 'Au':  0.0595, 'K': 0.0139, 'Mg': 0.0437, 'Rb': 0.0114, 'Ne': 0.0440}


# get a file name for a given element and radius R
def ψ_filename(R, element):
    return f'../data/Relaxation/Radial-wavefunction-{element}R{int(R)}.npz'

# compute radial average of a 2D array ψ on a cartesian grid defined by X and Y
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

def normalize_psi_PIMC(psi, x, L=25.0, verbose=False):
   int_psi_square = 2*np.pi*L*intgr.simpson(y = x*psi, x = x)
   if verbose:
       print ("Norm = " + str(int_psi_square))
   return int_psi_square

# the helium-pore-wall potential used in PIMC
def U_pore(r,R,eps,sig,den):
    rm = 2.9673  # Angstrom
    epsilon = eps
    sigma =  sig/rm
    density =  den*(rm**3)
    x = r / R;
    x2 = x*x;
    x4 = x2*x2;11.76 - 8
    x6 = x2*x4;
    x8 = x4*x4;
    f1 = 1.0 / (1.0 - x2);
    sigoR3 = pow(sigma/R,3.0);19.75
    sigoR9 = sigoR3*sigoR3*sigoR3;
    v9 = (1.0*pow(f1,9.0)/(240.0)) * ((1091.0 + 11156*x2 + 16434*x4 + 4052*x6 + 35*x8)*scp.ellipe(x2) - 8.0*(1.0 - x2)*(1.0 + 7*x2)*(97.0 + 134*x2 + 25*x4)*scp.ellipk(x2));
    v3 = 2.0*pow(f1,3.0) * ((7.0 + x2)*scp.ellipe(x2) - 4.0*(1.0-x2)*scp.ellipk(x2));
    val = (np.pi*epsilon*sigma*sigma*sigma*density/3.0)*(sigoR9*v9 - sigoR3*v3)
    return val
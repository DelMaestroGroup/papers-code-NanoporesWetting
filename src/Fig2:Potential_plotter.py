import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.special as scp
import scipy.integrate as intgr

plt.style.use('../include/aps.mplstyle')
mpl.rcParams["figure.figsize"] = [2*3.4039, 2*2.10373]

rm = 2.9673;
#Compute Recoil Energy
hbar = 11606  * 6.5821 * 10**(-16) #Kelvin second
hbar2by2 = hbar ** 2/2 #Kelvin^2 second^2
mass = 4* 9.3827 * 11606 * 10**8 #Kelvin/c^2
c = 2.998 * 10**(18) #Angstrom per second
E_rec = hbar2by2/(mass*rm**2)*c**2
print (E_rec)


def PIMC_potential(r,R,eps,sig,den):
    epsilon = eps
    sigma =  sig
    density =  den
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


def PIMC_potential_MCM(r,R):
    epsilon = 1.59
    sigma = 3.443
    density = 1
    x = r / R;
    x2 = x*x;
    x4 = x2*x2;
    x6 = x2*x4;
    x8 = x4*x4;
    f1 = 1.0 / (1.0 - x2);
    sigoR3 = pow(sigma/R,3.0);
    sigoR9 = sigoR3*sigoR3*sigoR3;
    v9 = (1.0*pow(f1,9.0)/(240.0)) * ((1091.0 + 11156*x2 + 16434*x4 + 4052*x6 + 35*x8)*scp.ellipe(x2) - 8.0*(1.0 - x2)*(1.0 + 7*x2)*(97.0 + 134*x2 + 25*x4)*scp.ellipk(x2));
    v3 = 2.0*pow(f1,3.0) * ((7.0 + x2)*scp.ellipe(x2) - 4.0*(1.0-x2)*scp.ellipk(x2));
    val = (np.pi*epsilon*sigma*sigma*sigma*density/3.0)*(sigoR9*v9 - sigoR3*v3)
    return val


def V_2shell_Cesium(x,R):
    #6.86 A is the diameter of a Cesium atom
    R_out2 =  R
    R_in2 = R - 6.86
    R_out1 = R - 2*6.86
    R_in1 = R - 3*6.86
    return PIMC_potential(x,R_in1, 1.359, 5.44,0.0091)  - PIMC_potential(x,R_out1, 1.359, 5.44,0.0091) + PIMC_potential(x,R_in2, 1.359, 5.44,0.0091)  - PIMC_potential(x,R_out2, 1.359, 5.44,0.0091) + PIMC_potential_MCM(x,R_out2)


def V_shell_Cesium(x,R):
    R_out1 =  R
    R_in1 = R - 6.86
    return PIMC_potential(x,R_in1, 1.359, 5.44,0.0091)  - PIMC_potential(x,R_out1, 1.359, 5.44,0.0091) + PIMC_potential_MCM(x,R_out1)

r = np.linspace(0,7.9,100)
R = 28.58
R1 = 8 + 6.86
R2 = 28.58 - (3*6.86)

print("Max deviation: ", np.max(np.abs(PIMC_potential(r,R2,1.359, 5.44,0.0091) - V_2shell_Cesium(r,R))))
print("Deviation range: ", np.max(np.abs(PIMC_potential(r,R2,1.359, 5.44,0.0091) - V_2shell_Cesium(r,R))), " to ", np.min(np.abs(PIMC_potential(r,R2,1.359, 5.44,0.0091) - V_2shell_Cesium(r,R))))

with plt.style.context('../include/aps.mplstyle'):
    figsize = plt.rcParams['figure.figsize']
    fig,ax = plt.subplots(figsize=(figsize[0],figsize[1]), constrained_layout=True)
    mpl.rcParams['axes.linewidth'] = 2.0
    R = 28.58
    R1 = 8 + 6.86
    R2 = 28.58 - (3*6.86)
    r = np.linspace(0,7.9,100)
    r2 = np.linspace(0,12.0,100)
    #ax.plot(r,V_shell_Cesium(r2,R1), linestyle='solid', color='green',label='MCM-41/Cs (1 layer)')
    ax.plot(r,PIMC_potential(r,R2,1.359, 5.44,0.0091), linestyle='--', color='#5E4Fa2',label='Infinite Cs')
    ax.plot(r,V_2shell_Cesium(r,R), linestyle='solid', color='#5E4Fa2',alpha=0.7,label='MCM-41/Cs/Cs (2 layers)')
    ax.legend(handlelength = 1)
    ax.set_ylabel(r'$U_{\rm pore}\; [{\rm K}]$')
    ax.set_xlabel(r'$r\; [{\rm \AA}]$')
    ax.set_ylim(-19,19)
    ax.set_xlim(0,4.7)
    
    #ax.tick_params(size=6)
    plt.savefig('../figures/Potential-comparison.pdf')
    #plt.show()





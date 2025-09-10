import numpy as np
import scipy.special as scp
import scipy.integrate as intgr
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import interp1d


plt.style.use('../include/aps.mplstyle')
rm = 2.9673;

def PIMC_potential(r,R):
    epsilon = 1.359
    sigma = 5.442152204855649/rm
    density =  0.0091*(rm**3)
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

def V_infinite(x):
    #Shell interface
    R_out = 1
    R_in = 8
    #return Ucyl(x,R_in)-Ucyl(x,R_out)
    return PIMC_potential(x,R_in)

def V_shell(x):
    #Shell interface
    R_out = 15.51
    R_in = 8.65
    #R_out = 12.86
    #R_in = 6.00
    #return Ucyl(x,R_in)-Ucyl(x,R_out)
    return PIMC_potential(x,R_in)  - PIMC_potential(x,R_out)

def V_2shell(x):
    #Shell interface
    R_out1 = 12.86
    R_in1 = 6.00
    R_in2 = 19.86
    R_out2 = 26.72
    #return Ucyl(x,R_in)-Ucyl(x,R_out)
    return PIMC_potential(x,R_in1)  - PIMC_potential(x,R_out1) + PIMC_potential(x,R_in2)  - PIMC_potential(x,R_out2)

def Ucyl_MCM(x,R):
    rho = 1
    eps = 1.59
    sigma = 3.44 
    coefffirst = 63*np.pi*rho*eps*(sigma**3)*(sigma**9)/64
    coeffsecond = 3*np.pi*rho*eps*(sigma**3)*(sigma**3)/2
    return coefffirst*I(10,x,R) - coeffsecond*I(4,x,R)

def PIMC_potential_MCM(r,R):
    epsilon = 1.59
    sigma = 3.44
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

lb = 0.001
rb = 10
p = 500
xval = np.linspace(lb,rb,p)
xval2 = np.linspace(0.001,14.5,500)
rHe = 1.4

def ave_r(r,ρ,R, L=25.0, N=2):
       return 2*np.pi*L*intgr.simpson(y = r*r*ρ, x = r)/N

with np.load("../data/Relaxation/Radial-wavefunction-CsR8.npz") as f:
    Csrval = f['arr_0']
    Csradial = f['arr_1']

with np.load("../data/Relaxation/Radial-wavefunction-ArR8.npz") as f:
    Arrval = f['arr_0']
    Arradial = f['arr_1']

oAve = ave_r(Arrval, Arradial,8.0)
print('oAve = ',oAve)

_r = np.linspace(0,7.85,1000)
with plt.style.context('../include/aps.mplstyle'):
    figsize = plt.rcParams['figure.figsize']
    fig,ax = plt.subplots(figsize=(figsize[0],figsize[1]), constrained_layout=True)

    ax.set_ylim(-10,10.5)
    ax.set_xlim(0,7.8)
    #plt.title('Radial Helium - Cesium interaction potential')
    #ax.set_ylabel(r'$U_{\rm pore}$ [K]')
    #ax.set_xlabel(r'$r$ [$\rm \AA$]')
    ax.plot(xval,V_infinite(xval), color='k')
    Csradial = 2000*Csradial
    Arradial = 2000*Arradial

    ρ_cubic = interp1d(Csrval, Csradial, kind='cubic')
    ax.plot(_r,ρ_cubic(_r), color='#5E4Fa2', label = 'Cs')
    aver_Cs = ave_r(_r, ρ_cubic(_r), 8.0)/2000
    ax.axvline(aver_Cs, color='#5E4Fa2', linestyle='--', linewidth=0.5)

    ρ_cubic = interp1d(Arrval, Arradial, kind='cubic')
    ax.plot(_r,ρ_cubic(_r), color='#D7414E', label = 'Ar')
    aver_Ar = ave_r(_r, ρ_cubic(_r), 8.0)/2000
    ax.axvline(aver_Ar, color='#D7414E', linestyle='--', linewidth=0.5)
    
    r0x = 6.424
    r0y = -5.194
    ax.axvline(r0x, color='#000000', linestyle='--', linewidth=0.5)

    #ax.plot(r0x,r0y,'ro')
    #ax.text(6.30, -6.5, r'$r_0$', fontsize=8)
    yval = -7.5
    #ax.fill_betweenx([r0x+2*rHe,r0x-2*rHe],[yval,yval],color='k', linestyle='-', linewidth=1)
    ax.axvspan(r0x-2*rHe, 8.0, alpha=0.1, edgecolor='none', facecolor='#858585', zorder=-100)
    #print(r0x+2*rHe)
    #ax.vlines(r0x-2*rHe,-10,10.5,color='k',linestyle='--')
    #ax.vlines(7.8,-10,10.5,color='k',linestyle='--')
    #ax.arrow(r0x,yval,2*rHe,0,length_includes_head=True,head_width=0.3, head_length=0.1)
    #ax.arrow(rHe/2,yval,rHe/2,0,length_includes_head=True,head_width=0.3, head_length=0.1)
    #ax.plot([0,rHe],[yval,yval],color='k', linestyle='-', linewidth=1)
    ax.text(0.05,Csradial[0]+0.2 , r'$\varrho(r)\rvert_{\rm Cs}$', color='#5E4FA2', va='bottom')
    ax.text(0.05,Arradial[0]+0.2 , r'$\varrho(r)\rvert_{\rm Ar}$', color='#D7414E', va='bottom')
    ax.text(r0x+0.1,r0y , r'$U_{\rm pore}$', va='top')


    #ax.text(rHe/2-0.4, -8.5, r'$\rho_{{\rm vdW}}$', fontsize=8)
    #plt.text(6.30-0.2, -8.5, r'$\rho_{{\rm well}}$', fontsize=8)
    #plt.plot(xval,V_2shell(xval)+PIMC_potential_MCM(xval,26.72),label=r'$U_{Cs/MCM}(r)$ (2 layer)')
    #plt.plot(xval,V_shell(xval)+PIMC_potential_MCM(xval,12.86),label=r'$U_{Cs/MCM}(r)$ (1 layer)')
    #plt.plot(xval2,PIMC_potential_MCM(xval2,15.51),label=r'$U(r)}$')
    #plt.plot(xval,Ucyl_MCM(xval,15.51), label=r'$U_{cyl}$')
    #ax.legend(handlelength = 1.5)

    ax.set_xticks([0, aver_Cs, aver_Ar, r0x-2*rHe, r0x,7.8])
    ax.set_xticklabels(['0', r'$\langle r\rangle_{\rm Cs}$', r'$\langle r\rangle_{\rm Ar}$', 
                        r'$r_0-2r_{\rm He}$', r'$r_0$', r'$R$'], rotation=0)

    ax.set_yticks([])
    ax.set_yticks([], minor=True)
    plt.savefig('../figures/Wetting_parameter.pdf')

    plt.show()

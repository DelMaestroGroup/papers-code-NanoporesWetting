import numpy as np
import scipy.special as scp
import scipy.integrate as intgr
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use('../aps.mplstyle')
mpl.rcParams["figure.figsize"] = [3.4039, 2.10373]
rm = 2.9673;

def PIMC_potential_Cs(r,R):
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

def PIMC_potential_Ar(r,R):
    epsilon = 36.136
    sigma = 3.0225/rm
    density =  0.0265*(rm**3)
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
    print("Here")
    R_out = 1
    R_in = 8.65
    #return Ucyl(x,R_in)-Ucyl(x,R_out)
    return PIMC_potential_Cs(x,R_in)

def V_shell(x):
    #Shell interface
    R_out = 15.51
    R_in = 8.65
    #R_out = 12.86
    #R_in = 6.00
    #return Ucyl(x,R_in)-Ucyl(x,R_out)
    return PIMC_potential_Cs(x,R_in)  - PIMC_potential_Cs(x,R_out)

def V_shell_Ar(x):
    #Shell interface
    R_out = 15.51
    R_in = 11.75
    #R_out = 12.86
    #R_in = 6.00
    #return Ucyl(x,R_in)-Ucyl(x,R_out)
    return PIMC_potential_Ar(x,R_in)  - PIMC_potential_Ar(x,R_out)

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
    sigma = 3.44/rm
    density = 1*(rm**3)
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
xval2 = np.linspace(0.001,16.5,500)
with plt.style.context('aps'):
    figsize = plt.rcParams['figure.figsize']
    fig,ax = plt.subplots(figsize=(figsize[0],figsize[1]), constrained_layout=True)

    plt.ylim(-200,100)
    #plt.title('Radial Helium - Cesium interaction potential')
    plt.ylabel(r'V[K]')
    plt.xlabel(r'r[Å]')
    plt.plot(xval2,V_shell_Ar(xval2)+PIMC_potential_MCM(xval2,15.51),label=r'$U_{Ar/MCM}(r)$',color='#D7414E')
    plt.plot(xval,V_shell(xval)+PIMC_potential_MCM(xval,15.51),label=r'$U_{Cs/MCM}(r)$',color='#5E4Fa2')
    #plt.plot(xval2,PIMC_potential_MCM(xval2,15.51),label=r'$U(r)}$')
    plt.plot(xval2,PIMC_potential_MCM(xval2,15.51), label=r'$U_{MCM}$',color='k')
    plt.legend(handlelength=1)
    plt.savefig('Potential_comp.pdf')
    plt.show()

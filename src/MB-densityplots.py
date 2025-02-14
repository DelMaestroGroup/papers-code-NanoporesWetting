#!/usr/bin/env python
# coding: utf-8

# In[44]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.integrate as intgr


# In[45]:


def normalize_psi_PIMC(psi, x):
   int_psi_square = 2*np.pi*25*intgr.simpson(y = x*psi, x = x)
   print(int_psi_square)
   return int_psi_square


# In[46]:


plt.style.use('aps')
mpl.rcParams["figure.figsize"] = [2*3.4039, 2*2.10373]


# In[47]:


f2 = open('ArR6/radial-N-reduce-8c91642a-74a9-43ea-ba51-09a0aa756691.dat','r')
lines = f2.readlines()
x1 = np.array([])
y1 = np.array([])
z1 = np.array([])
for line in lines[3:]:
    p = line.split()
    x1 = np.append(x1,float(p[0]))
    y1 = np.append(y1,float(p[1]))
    z1 = np.append(z1,float(p[2]))
f2.close()
f2 = open('ArR6/radial-N-reduce-bef69b2e-0c94-4f54-8901-3cd6f6974ed3.dat','r')
lines = f2.readlines()
x4 = np.array([])
y4 = np.array([])
z4 = np.array([])
for line in lines[3:]:
    p = line.split()
    x4 = np.append(x4,float(p[0]))
    y4 = np.append(y4,float(p[1]))
    z4 = np.append(z4,float(p[2]))
f2.close() 
f2 = open('ArR6/radial-N-reduce-f91dbe28-280e-4fc8-8333-ba8b13a9521b.dat','r')
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
f2 = open('ArR6/radial-N-reduce-06923e57-b262-41e3-9094-0e82f382c23b.dat','r')
lines = f2.readlines()
x6 = np.array([])
y6 = np.array([])
z6 = np.array([])
for line in lines[3:]:
    p = line.split()
    x6 = np.append(x6,float(p[0]))
    y6 = np.append(y6,float(p[1]))
    z6 = np.append(z6,float(p[2]))
f2.close()


# In[48]:


norm = normalize_psi_PIMC(y,x)
norm = normalize_psi_PIMC(y1,x1)
norm = normalize_psi_PIMC(y4,x4)
norm = normalize_psi_PIMC(y6,x6)
figsize = plt.rcParams['figure.figsize']
fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(figsize[0],figsize[1]), constrained_layout=True)
ax3.errorbar(x,y,yerr=z, linestyle="None",marker = '+', label=r'$\mu = -58.0$ K', color='#D7414E', alpha = 0.2)
ax3.errorbar(x4,y4,yerr=z4, linestyle="None",marker = '+',label=r'$\mu = -40.0$ K', color='#D7414E', alpha = 0.4)
ax3.errorbar(x6,y6,yerr=z6, linestyle="None",marker = '+',label=r'$\mu = -18.0$ K', color='#D7414E', alpha = 0.7)
ax3.errorbar(x1,y1,yerr=z1, linestyle="None",marker = '+', label=r'$\mu = -5.0$ K',  color='#D7414E', alpha = 1.0)

ax3.axhline(0, color='black', linewidth=.5)
ax3.axvline(0, color='black', linewidth=.5)
ax3.set_ylabel(r'$|\psi(r)|^2$',fontsize=10)
ax3.set_xlabel('r',fontsize=10)
leg = ax3.legend(title = ' (c) R = 6 Å')
leg._legend_box.align = "left"


# In[49]:


f2 = open('ArR8/radial-N-reduce-4fd8254f-12fa-49c6-b175-625245c7b90f.dat','r')
lines = f2.readlines()
x1 = np.array([])
y1 = np.array([])
z1 = np.array([])
for line in lines[3:]:
    p = line.split()
    x1 = np.append(x1,float(p[0]))
    y1 = np.append(y1,float(p[1]))
    z1 = np.append(z1,float(p[2]))
f2.close()
f2 = open('ArR8/radial-N-reduce-2b83d13d-2662-4599-8814-8008f0791188.dat','r')
lines = f2.readlines()
x4 = np.array([])
y4 = np.array([])
z4 = np.array([])
for line in lines[3:]:
    p = line.split()
    x4 = np.append(x4,float(p[0]))
    y4 = np.append(y4,float(p[1]))
    z4 = np.append(z4,float(p[2]))
f2.close()
f2 = open('ArR8/radial-N-reduce-581f432a-5144-4449-99a9-b3ac8f1cf6df.dat','r')
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
f2 = open('ArR8/radial-N-reduce-72495c80-c5fc-4cc5-9e5b-477b56f7a098.dat','r')
lines = f2.readlines()
x6 = np.array([])
y6 = np.array([])
z6 = np.array([])
for line in lines[3:]:
    p = line.split()
    x6 = np.append(x6,float(p[0]))
    y6 = np.append(y6,float(p[1]))
    z6 = np.append(z6,float(p[2]))
f2.close()


# In[50]:


norm = normalize_psi_PIMC(y,x)
norm = normalize_psi_PIMC(y1,x1)
norm = normalize_psi_PIMC(y4,x4)
norm = normalize_psi_PIMC(y6,x6)
ax4.errorbar(x,y,yerr=z, linestyle="None",marker = '+', label=r'$\mu = -58.0$ K', color='#D7414E', alpha = 0.2)
ax4.errorbar(x4,y4,yerr=z4, linestyle="None",marker = '+',label=r'$\mu = -40.0$ K', color='#D7414E', alpha = 0.4)
ax4.errorbar(x6,y6,yerr=z6, linestyle="None",marker = '+',label=r'$\mu = -18.0$ K', color='#D7414E', alpha = 0.7)
ax4.errorbar(x1,y1,yerr=z1, linestyle="None",marker = '+', label=r'$\mu = -5.0$ K',  color='#D7414E', alpha = 1.0)

ax4.axhline(0, color='black', linewidth=.5)
ax4.axvline(0, color='black', linewidth=.5)
ax4.set_ylabel(r'$|\psi(r)|^2$',fontsize=10)
ax4.set_xlabel('r',fontsize=10)
leg = ax4.legend(title = ' (d) R = 8 Å')
leg._legend_box.align = "left"


# In[51]:


f2 = open('CsR6/radial-N-reduce-54c1cbf6-cbfb-4368-ba38-5b70c0e6cf8b.dat','r')
lines = f2.readlines()
x1 = np.array([])
y1 = np.array([])
z1 = np.array([])
for line in lines[3:]:
    p = line.split()
    x1 = np.append(x1,float(p[0]))
    y1 = np.append(y1,float(p[1]))
    z1 = np.append(z1,float(p[2]))
f2.close()
f2 = open('CsR6/radial-N-reduce-1e851cc1-1e59-4d44-8786-2896477c9f6b.dat','r')
lines = f2.readlines()
x6 = np.array([])
y6 = np.array([])
z6 = np.array([])
for line in lines[3:]:
    p = line.split()
    x6 = np.append(x6,float(p[0]))
    y6 = np.append(y6,float(p[1]))
    z6 = np.append(z6,float(p[2]))
f2.close()
f2 = open('CsR6/radial-N-reduce-d0a4e45e-9d74-4615-8378-f75907da4b6f.dat','r')
lines = f2.readlines()
x4 = np.array([])
y4 = np.array([])
z4 = np.array([])
for line in lines[3:]:
    p = line.split()
    x4 = np.append(x4,float(p[0]))
    y4 = np.append(y4,float(p[1]))
    z4 = np.append(z4,float(p[2]))
f2.close()
f2 = open('CsR6/radial-N-reduce-cb54392c-d1c2-4dea-97b8-86f981a9111f.dat','r')
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


# In[52]:


norm = normalize_psi_PIMC(y,x)
norm = normalize_psi_PIMC(y1,x1)
norm = normalize_psi_PIMC(y4,x4)
norm = normalize_psi_PIMC(y6,x6)
ax1.errorbar(x,y,yerr=z, linestyle="None",marker = '+', label=r'$\mu = -58.0$ K', color='#5E4FA2', alpha = 0.2)
ax1.errorbar(x4,y4,yerr=z4, linestyle="None",marker = '+',label=r'$\mu = -40.0$ K', color='#5E4FA2', alpha = 0.4)
ax1.errorbar(x6,y6,yerr=z6, linestyle="None",marker = '+',label=r'$\mu = -18.0$ K', color='#5E4FA2', alpha = 0.7)
ax1.errorbar(x1,y1,yerr=z1, linestyle="None",marker = '+', label=r'$\mu = -5.0$ K',  color='#5E4FA2', alpha = 1.0)

ax1.axhline(0, color='black', linewidth=.5)
ax1.axvline(0, color='black', linewidth=.5)
ax1.set_ylabel(r'$|\psi(r)|^2$',fontsize=10)
ax1.set_xlabel('r',fontsize=10)
leg = ax1.legend(title = ' (a) R = 6 Å')
leg._legend_box.align = "left"


# In[53]:


f2 = open('CsR8/radial-N-reduce-0f3f952d-638d-410c-b99b-bd8c3bdb2934.dat','r')
lines = f2.readlines()
x1 = np.array([])
y1 = np.array([])
z1 = np.array([])
for line in lines[3:]:
    p = line.split()
    x1 = np.append(x1,float(p[0]))
    y1 = np.append(y1,float(p[1]))
    z1 = np.append(z1,float(p[2]))
f2.close()
f2 = open('CsR8/radial-N-reduce-2148c31c-a07e-4aa2-b530-6de11f7edd86.dat','r')
lines = f2.readlines()
x6 = np.array([])
y6 = np.array([])
z6 = np.array([])
for line in lines[3:]:
    p = line.split()
    x6 = np.append(x6,float(p[0]))
    y6 = np.append(y6,float(p[1]))
    z6 = np.append(z6,float(p[2]))
f2.close()
f2 = open('CsR8/radial-N-reduce-bae75c59-fb78-4f51-a930-d656e4387976.dat','r')
lines = f2.readlines()
x4 = np.array([])
y4 = np.array([])
z4 = np.array([])
for line in lines[3:]:
    p = line.split()
    x4 = np.append(x4,float(p[0]))
    y4 = np.append(y4,float(p[1]))
    z4 = np.append(z4,float(p[2]))
f2.close()
f2 = open('CsR8/radial-N-reduce-5c592e8d-7d73-4f61-b25b-e2e1099120ea.dat','r')
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


# In[54]:


norm = normalize_psi_PIMC(y,x)
norm = normalize_psi_PIMC(y1,x1)
norm = normalize_psi_PIMC(y4,x4)
norm = normalize_psi_PIMC(y6,x6)
ax2.errorbar(x,y,yerr=z, linestyle="None",marker = '+', label=r'$\mu = -58.0$ K', color='#5E4FA2', alpha = 0.2)
ax2.errorbar(x4,y4,yerr=z4, linestyle="None",marker = '+',label=r'$\mu = -40.0$ K', color='#5E4FA2', alpha = 0.4)
ax2.errorbar(x6,y6,yerr=z6, linestyle="None",marker = '+',label=r'$\mu = -18.0$ K', color='#5E4FA2', alpha = 0.7)
ax2.errorbar(x1,y1,yerr=z1, linestyle="None",marker = '+', label=r'$\mu = -5.0$ K',  color='#5E4FA2', alpha = 1.0)

ax2.axhline(0, color='black', linewidth=.5)
ax2.axvline(0, color='black', linewidth=.5)
ax2.set_ylabel(r'$|\psi(r)|^2$',fontsize=10)
ax2.set_xlabel('r',fontsize=10)
leg = ax2.legend(title = ' (b) R = 8 Å')
leg._legend_box.align = "left"
plt.savefig('Mb-density.pdf')


# In[ ]:





# In[ ]:





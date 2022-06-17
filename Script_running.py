# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 09:30:39 2021

@author: Ožbej
"""

import numpy as np
import scipy.signal as scisi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import scipy.optimize as sciop
import CV_Explicit_method_IR_drop as cve
import CV_IR_Nonlinear_capacitance as cvi
import Fitting_file_opening_functions as ff
import sys
import time
import random

def rectangular(f,w0,w):
    return np.where(abs(f-w0)<=w,1,0)

#constants

R=8.314
Temp=273+25
F=96485
n=1
f=n*F/(R*Temp)

#input parameters
#podatki za DC signal
v=0.1
Ei=0.5
Ef=-0.5
E0=0.

#Prostorske konstante
D=10**-9 #[m^2/s]
nx=20

#time constants
nt=100
nt=int(2*nt*abs((f*Ei-f*Ef)))
#bulk koncentracija
cbulk=1

#ključna kinetična parametra
alfa=0.5
k0=10 #[m/s]

#glavni konstanti pri ac-cv, amplituda in frekvenca sinusa
frequency=9
amplitude=0.

#parametri za kapacitivnost
Cdl=0*10**-5#[F]
Ru=000#[Ohm]

A=15*10**-6 #[m^2]

base_angle=1.4992525990476047

tau=1/(f*v)

E_sim1,t_sim,dt=cvi.potencial(Ei, Ef, v, amplitude, frequency, nt, 0)
E_sim=E_sim1+amplitude*np.sin(2*np.pi*frequency*t_sim)

t=t_sim/tau
dt=t[-1]/nt

sp=np.fft.fft(E_sim)
freq=np.fft.fftfreq(t_sim.shape[-1],d=dt*tau)

# plt.plot(freq,np.log10(sp))

# sys.exit()

numerical_params=[dt,tau,f,n,R,Temp,F,cbulk,v,Ei,Ef,D,nt,t]
dim_params=[n,F,cbulk,D,v,R,Temp,frequency,t_sim]

pnom=f*E_sim

x0=[alfa,k0,E0,Ru,A,Cdl]

dx=0.1
tau=1/(f*v)
dt=t[-1]/len(t+1)
xmax=6*np.sqrt(2*abs(f*Ei-f*Ef))
gama=cvi.find_gama(dx, xmax, nx)
N=np.arange(nx+3)
x=dx*(gama**N-1)/(gama-1)

numerical_params=[dx,dt,tau,f,n,R,Temp,F,cbulk,v,Ei,Ef,D,nx,len(t),x,t]

dim_params=[n,F,cbulk,D,v,R,Temp,frequency,t]

I_sim=cvi.CV_simulation(pnom, x0, numerical_params)[1]*n*F*A*cbulk*np.sqrt(n*F*D*v/R/Temp)

pnom=pnom/f

plt.figure("Voltamogram")
plt.plot(pnom,I_sim)
plt.xlabel("E [V]")
plt.ylabel("I [A]")

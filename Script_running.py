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

I_sim=cvi.CV_simulation(pnom, x0, numerical_params)[1]#*n*F*A*cbulk*np.sqrt(n*F*D*v/R/Temp)
# I_sim=amplitude*np.sin(2*np.pi*frequency*t_sim)
pnom=pnom/f
hp=int(len(I_sim)/2)

# pnom1=E_sim1*f
# x0=[0,1,k0,E0,Ru,A,Cdl]
# I_sim1=cvi.CV_simulation(pnom, x0, numerical_params)[1]

# Inew=I_sim-I_sim1
# anal_signal=np.abs(scisi.hilbert(Inew))
# plt.plot(E_sim1,I_sim,label='3')
# # plt.plot(E_sim1,np.abs(anal_signal),label="ovojnica obdelanega signala")
# ax = plt.gca()
# ax.set_yticklabels([])
# ax.set_xticklabels([])
# ax.set_xlabel('E')
# ax.set_ylabel('i')
# plt.grid()

# sp=np.fft.fft(I_sim)

# # plt.plot(freq,np.log10(np.abs(sp)))

# w=1.3*np.ones(6)
# I_harmonics=[]
# an=[]
# an1=[]
# for i in range(6):
#     filter_sp=sp.copy()
#     window=rectangular(freq,i*frequency,w[i])
#     filter_sp=window*filter_sp
#     Inew=np.real(np.fft.ifft(filter_sp))
#     Inew=np.fft.ifft(filter_sp).real
#     I_harmonics.append(Inew)
#     anal_signal=np.abs(scisi.hilbert(Inew))
#     an.append(anal_signal)

# fig, axs = plt.subplots(2, 3, sharex="all")
# axs[0, 0].plot(E_sim1, I_harmonics[0])
# axs[0,0].vlines(0,min(I_harmonics[0]),max(I_harmonics[0]),linestyle="dashed",color="black")
# axs[0, 0].set_title("0-ti harmonik")
# axs[0,0].set(xticks=[], yticks=[],xlabel="E",ylabel="i")

# axs[0, 1].plot(E_sim1,an[1])
# axs[0,1].vlines(0,0,max(I_harmonics[1]),linestyle="dashed",color="black")
# # axs[0,1].hlines(I_harmonics[1][0],E_sim1[0],E_sim1[hp],linestyle="dashed",color="red")
# # axs[0,1].fill_between(0,I_harmonics[1][0])
# axs[0, 1].set_title("1. harmonik")
# axs[0,1].set(yticks=[],xlabel="E",ylabel="i")

# axs[0, 2].plot(E_sim1,an[2])
# axs[0,2].vlines(0,0,max(I_harmonics[2]),linestyle="dashed",color="black")
# axs[0, 2].set_title("2. harmonik")
# axs[0,2].set(yticks=[],xlabel="E",ylabel="i")

# axs[1, 0].plot(E_sim1,an[3])
# axs[1,0].vlines(0,0,max(I_harmonics[3]),linestyle="dashed",color="black")
# axs[1, 0].set_title("3. harmonik")
# axs[1,0].set(yticks=[],xlabel="E",ylabel="i")

# axs[1, 1].plot(E_sim1,an[4])
# axs[1,1].vlines(0,0,max(I_harmonics[4]),linestyle="dashed",color="black")
# axs[1, 1].set_title("4. harmonik")
# axs[1,1].set(yticks=[],xlabel="E",ylabel="i")

# axs[1, 2].plot(E_sim1,an[5])
# axs[1,2].vlines(0,0,max(I_harmonics[5]),linestyle="dashed",color="black")
# axs[1, 2].set_title("5. harmonik")
# axs[1,2].set(yticks=[],xlabel="E",ylabel="i")

plt.figure(2)
plt.plot(E_sim1,I_sim,label="signal z izmenično napetostjo")
# plt.plot(E_sim1,I_sim1,label="Signal z enosmerno napetostjo")
# plt.plot(pnom[hp:],I_sim[hp:])
# plt.plot(pnom,I_sim,label="None")
# x1=np.where(I_sim==max(I_sim))[0][0]
# x2=np.where(I_sim==min(I_sim))[0][0]
# plt.plot([pnom[x2],pnom[x1]],[I_sim[x2],I_sim[x1]],color="black",linestyle="--")
# # plt.plot([0],[0],marker="o",color="red")
# # t1,t2=[pnom[x1],pnom[x1]],[0,max(I_sim)]
# # t3,t4=[pnom[x2],pnom[x2]],[0,min(I_sim)]
# # plt.plot(t1,t2,color="black",linestyle="dashed")
# # plt.plot(t3,t4,color="black",linestyle="dashed")
# # plt.plot([pnom[x1]],[max(I_sim)],color="Black",marker="o")
# # plt.plot([pnom[x2]],[min(I_sim)],color="Black",marker="o")
# # plt.plot([pnom[hp]],I_sim[hp],color="Black",marker="o")
# # plt.plot([pnom[-1]],I_sim[-1],color="Black",marker="o")
# # plt.text(pnom[x1]*2,I_sim[x1]*1.01,"$i_{a}$")
# # plt.text(pnom[x2]*3,I_sim[x2]*1.01,"$i_{c}$")
# # plt.text(pnom[hp],I_sim[hp]*1.3,"$i_{lc}$")
# # plt.text(pnom[-1],I_sim[-1]*1.3,"$i_{la}$")
# ax = plt.gca()
# ax.set_yticklabels([])
# ax.set_xticklabels([])
# ax.set_xlabel('E')
# ax.set_ylabel('i')
# plt.grid()
# angle=np.arctan((I_sim[x1]-I_sim[x2])/(pnom[x1]-pnom[x2]))
# print(angle)
# theta=base_angle-angle
# rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
#                        [np.sin(theta), np.cos(theta)]])

# data=np.array([pnom,I_sim]).T
# data_rot = (rot_matrix @ data.T).T

# plt.plot(data_rot[:,0],data_rot[:,1])
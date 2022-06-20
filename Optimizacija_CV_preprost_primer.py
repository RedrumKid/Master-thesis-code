# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 09:36:28 2021

@author: Ožbej
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import scipy.optimize as sciop
import CV_IR_Fitting_functions as cv


def loss_func(X,x,y_data,f,extra_params):
    return np.sum((y_data-f((x),X,extra_params)[1])**2)

#constants

R=8.314
Temp=293
F=96485
n=1
f=n*F/(R*Temp)

#input parameters
#podatki za DC signal
v=0.1
Ei=0.5
Ef=-0.5
E0=0.1

#Prostorske konstante
D=10**-8
nx=20

#time constants
nt=100
#bulk koncentracija
cbulk=1

#ključna kinetična parametra
alfa=0.5
k0=10**-5

#glavni konstanti pri ac-cv, amplituda in frekvenca sinusa
frequency=9
amplitude=0.0

#parametri za kapacitivnost
Cdl=0.0001
Ru=0.005

dx=0.2
params=[alfa,k0,E0,Cdl,Ru]

t,dt,nt,x,time,xmax,tau,pnom,pdc,psin=cv.Non_ranges(Ei,Ef,v,amplitude,frequency,E0,f,D,dx,nt,nx,alfa)
numerical_params=[dx,dt,tau,f,n,R,Temp,F,cbulk,v,Ei,Ef,D,nx,nt,x,t]

Es,I_data,t=cv.CV_simulation(pnom,params,numerical_params) 

rng=np.random.default_rng()
I_data=I_data+0.00*rng.normal(size=pnom.size)

plt.figure("CV")
plt.plot(pnom,I_data)


x0=np.array([0.6,10**-4,0.07,0,0])

print(loss_func(x0, pnom, I_data, cv.CV_simulation, numerical_params))

res=sciop.minimize(loss_func,x0,method="Nelder-Mead",tol=10**-6,args=(pnom,I_data,cv.CV_simulation,numerical_params))

plt.plot(pnom,cv.CV_simulation(pnom, res.x, numerical_params)[1])
print(loss_func(res.x, pnom, I_data, cv.CV_simulation, numerical_params))
print("alfa="+str(round(res.x[0],3)))
print("k0="+str(round(res.x[1],3)))
print("E0="+str(round(res.x[2],3)))
print("Cdl="+str(round(res.x[3],6)))
print("Ru="+str(round(res.x[4],6)))


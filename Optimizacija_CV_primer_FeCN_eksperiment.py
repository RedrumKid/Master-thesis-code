# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 10:34:21 2021

@author: Ožbej
"""

import numpy as np
import scipy.signal as scisi
import scipy.stats as scist
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import scipy.optimize as sciop
# import CV_IR_Fitting_functions as cv
import CV_IR_Nonlinear_capacitance as cv
import Fitting_file_opening_functions as ff
import sys
import time

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def loss_func(Fit_params,x,y_data,f,extra_params,dim_params):
    n,F,cbulk,D,v,R,Temp,frequency,t=dim_params
    y_sim=f(x,Fit_params,extra_params)[1]*n*F*Fit_params[4]*cbulk*np.sqrt(n*F*D*v/R/Temp)
    return (np.sum((y_data[10:]-y_sim[10:])**2)/np.sum(y_data[10:]**2))

def loss_func_ACV(Fit_params,x,y_data,f,extra_params,dim_params,N_harmonics,w):
    n,F,cbulk,D,v,R,Temp,frequency,t=dim_params
    y_sim=f(x,Fit_params,extra_params)[1]*n*F*Fit_params[4]*cbulk*np.sqrt(n*F*D*v/R/Temp)
    
    # l=0.
    # window=scisi.windows.tukey(int(len(E_data)),l)
    # window=scisi.windows.tukey(int(len(E_data)/2),l)
    # window=np.append(window,scisi.windows.tukey(int(len(E_data)/2),l))
    
    a=[y_data,t]
    b=[extra_params[1],frequency]
    I_har_data=cv.FFT_analysis_loss(a, b, N_harmonics, w)
    
    # a=[y_sim*window,t]
    a=[y_sim,t]
    I_har_sim=cv.FFT_analysis_loss(a, b, N_harmonics, w)
    
    Psi=0
    
    for i in range(len(I_har_sim)):
        Psi+=np.sqrt(np.sum((I_har_data[i]-I_har_sim[i])**2)/np.sum(I_har_data[i]**2))
    
    return Psi/(N_harmonics+1)

def cot_f(x,A):
    return A/np.sqrt(x)

#constants

R=8.314
Temp=297
F=96485
n=1
f=n*F/(R*Temp)

#input parameters
#podatki za DC signal
v=101.725*10**-3
Ei=0.5
Ef=-0.5

#glavni konstanti pri ac-cv, amplituda in frekvenca sinusa
frequency=9
amplitude=0.1

#Prostorske konstante
D=7.2*10**-10 #[m^2/s] FeCN
# D=2.4*10**-9 #Fc
nx=20

#time constants
nt=100
#bulk koncentracija
cbulk=1

alfa=0.6
k0=0.0002
E0=0.1
Ru=30
A=2*10**-5
Cdl0=5e-06
Cdl1=0

dx=0.1

smothf=0
# a,b=ff.open_single_file()
# ff_param=abs(len(a[:,0])-2**14)
# ff_param=1
# E_data=a[:,0]#318088
# # E_data=smooth(E_data,smothf)
# I_data=a[:,1]*10**-3
# # I_data=scisi.savgol_filter(I_data, 51, 4)
# # I_data=smooth(I_data,smothf)
# t_data=a[:,2]-a[0,2]
# dt=t_data[2]-t_data[1]

# sp=np.fft.fft(E_data)
# freq=np.fft.fftfreq(t_data.shape[-1],d=dt)
# E_sim=np.fft.ifft(cv.rectangular(freq,0,0.01)*sp).real

# plt.plot(freq,np.log10(sp))
# plt.plot(t_data,E_data)

# sys.exit()

E_sim,t_sim,dt=cv.potencial(Ei, Ef, v, amplitude, frequency, int(2*nt*(abs(f*Ei-f*Ef))), 0)
# # E_sim,t_sim,dt=cv.potencial(Ei, Ef, v, amplitude, frequency, len(E_data), 0)
E_sim1=E_sim+amplitude*np.sin(2*np.pi*frequency*t_sim)
# # frequency=b["fs"]
# # amplitude=1.1*b["A (mV)"]*10**-3

E_data=E_sim1
t_data=t_sim

tau=1/(f*v)
t=t_data/tau
dt=t[-1]/len(t+1)
xmax=6*np.sqrt(abs(f*Ei-f*Ef))
gama=cv.find_gama(dx, xmax, nx)
N=np.arange(nx+3)
x=dx*(gama**N-1)/(gama-1)

pnom=E_data*f

numerical_params=[dx,dt,tau,f,n,R,Temp,F,cbulk,v,Ei,Ef,D,nx,len(t),x,t]
dim_params=[n,F,cbulk,D,v,R,Temp,frequency,t]

# x0=[alfa,k0,E0,Cdl,Ru,A]
x0=[alfa,k0,E0,Ru,A,Cdl0]

I_data=cv.CV_simulation(pnom, x0, numerical_params)[1]*n*F*A*cbulk*np.sqrt(n*F*D*v/R/Temp)
# I_data=cv.CV_simulation(pnom, x0, numerical_params)[1]
# print(min(I))

# l=0.1
# # window=scisi.windows.tukey(int(len(E_data)),l)
# window=scisi.windows.tukey(int(len(E_data)/2),l)
# window=np.append(window,scisi.windows.tukey(int(len(E_data)/2)+1,l))

# I_data=window*I_data

# sys.exit()

# I_data=scisi.decimate(I_data,12)
# E_data=scisi.decimate(E_data,12)
# t_data=scisi.decimate(t_data,12)

# I_data=scisi.wiener(I_data)
# I_data=scisi.savgol_filter(I_data,33,5)

# target_noise_db=60
# target_noise_watts = 10 ** (target_noise_db / 10)
# mean_noise=0
# noise=0.0001*max(I_data)*np.random.normal(mean_noise, np.sqrt(target_noise_watts), len(I_data))
# sp=np.fft.fft(noise)

plt.figure("CV")
plt.plot(t_data,I_data,label="Osnovni signal")

# sys.exit()

a=[E_data,I_data,t_data,0]
b=[amplitude,frequency]
N_harmonics=5
w=[1.5,1.7,1,1.2,1.3,1.,1.3]
# w=np.ones(8)*0.015
# freq,spectrum=scisi.periodogram(I_data,fs=1/dt)
spectrum,freq,I_har=cv.FFT_analysis(a, b, N_harmonics, w)
# spectrum=scisi.savgol_filter(spectrum,33,2)
plt.figure("FT_Spectrum")
plt.plot(freq,np.log10(np.abs(spectrum)),label="Osnovni signal")
# plt.plot(freq,(spectrum))
# plt.plot(freq,np.log10(sp))

# sys.exit()

for i in range(len(I_har)):
    plt.figure(str(i)+"_harmonic")
    plt.plot(t_data,I_har[i],label="Osnovni signal")

# sys.exit()

x0=[0.8*alfa,0.99*k0,0.95*E0,2*Ru,1*A,0.99*Cdl0]

start_time = time.process_time()
# dim_params=[n,F,cbulk,D,v,R,Temp]
# res=sciop.minimize(loss_func,x0,method="Nelder-Mead",tol=10**-3,args=(pnom,I_data,cv.CV_simulation,numerical_params,dim_params))
res=sciop.minimize(loss_func_ACV,x0,method="Nelder-Mead",tol=10**-8,args=(pnom,I_data,cv.CV_simulation,numerical_params,dim_params,N_harmonics,w))

print("Fitting time")
print("--- %s seconds ---" % (time.process_time() - start_time))

I_sim=cv.CV_simulation(pnom, res.x, numerical_params)[1]*n*F*res.x[4]*cbulk*np.sqrt(n*F*D*v/R/Temp)
# I_sim=cv.CV_simulation(pnom, x0, numerical_params)[1]*n*F*A*cbulk*np.sqrt(n*F*D*v/R/Temp)
# I_sim=window*I_sim

plt.figure("CV")
plt.title("Primerjava osnovni signal in najden; meritev v času")
plt.plot(t_data,I_sim,label="simulacija")
plt.xlabel("t [s]")
plt.ylabel("i [A]")
plt.legend()
a=[E_data,I_sim,t_data,0]

spectrum,freq,I_har=cv.FFT_analysis(a, b, N_harmonics, w)

plt.figure("FT_Spectrum")
plt.title("Fourierjev spekter eksperiment-simulacija")
plt.plot(freq,np.log10(np.abs(spectrum)),label="Najden signal")
plt.xlabel("f [Hz]")
plt.ylabel("A [dB]")
plt.legend()
for i in range(len(I_har)):
    plt.figure(str(i)+"_harmonic")
    plt.title("primerjava "+str(i)+" harmonik")
    plt.plot(t_data,I_har[i],label="najden")
    plt.xlabel("t [s]")
    plt.ylabel("i [A]")
    plt.legend()

# Phi0=loss_func_ACV(x0,pnom,I_data,cv.CV_simulation,numerical_params,dim_params,N_harmonics,w)

# print(Phi0)

# df=(len(I_data)-len(x0))
# sigma=np.sum((I_data-I_sim)**2)/(len(I_data)-len(x0))
# sigma=np.sqrt(sigma)
# chi=np.sum((I_data-I_sim)**2)/sigma
# print(scist.chi.sf(chi,df))

# sys.exit()



print()
print("Message: "+res.message)
print()
print("alfa="+str(round(res.x[0],3)))
print("k0="+str(round(res.x[1],8)))
print("E0="+str(round(res.x[2],3)))
print("Ru="+str(round(res.x[3],6)))
print("A="+str(round(res.x[4],6)))
print("c0="+str(round(res.x[5],10)))
print("c1="+str(round(res.x[6],10)))
    
    
print()
print("Nit")
print(res.nit)
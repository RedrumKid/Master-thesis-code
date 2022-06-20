# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 08:58:20 2021

@author: Ožbej
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 08:40:19 2021

@author: Ožbej
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sciop
import scipy.signal as scisi
import sys

def Fornberg_weights(z,x,n,m):
# From Bengt Fornbergs (1998) SIAM Review paper.
#  	Input Parameters
#	z location where approximations are to be accurate,
#	x(0:nd) grid point locations, found in x(0:n)
#	n one less than total number of grid points; n must
#	not exceed the parameter nd below,
#	nd dimension of x- and c-arrays in calling program
#	x(0:nd) and c(0:nd,0:m), respectively,
#	m highest derivative for which weights are sought,
#	Output Parameter
#	c(0:nd,0:m) weights at grid locations x(0:n) for derivatives
#	of order 0:m, found in c(0:n,0:m)
#      	dimension x(0:nd),c(0:nd,0:m)
    
    c=np.zeros((n+1,m+1))
    c1=1
    c4=x[0]-z
    
    for k in range(0,m+1):
        for j in range(0,n+1):
            c[j,k]=0
    
    c[0,0]=1
    
    for i in range(1,n):
        mn=min([i,m])
        c2=1
        c5=c4
        c4=x[i]-z
        for j in range(0,i):
            c3=x[i]-x[j]
            c2=c3*c2
            
            if j==i-1:
                for k in range(mn,0,-1):
                    c[i,k]=c1*(k*c[i-1,k-1]-c5*c[i-1,k])/c2
                c[i,0]=-c1*c5*c[i-1,0]/c2
            
            for k in range(mn,0,-1):
                c[j,k]=(c4*c[j,k]-k*c[j,k-1])/c3
            
            c[j,0]=c4*c[j,0]/c3
        
        c1=c2
    
    return c

def potencial(Ei,Ef,v,amplitude,frequency,nt,E0):
    # calculate the potentail input signal
    En=(Ei+Ef)/2
    tp=2*(Ei-Ef)/v
    a=Ei-En
    ts=abs(2*(-Ei+Ef)/v)
    t=np.linspace(0,ts,nt)
    Edc=a-2*a/np.pi*np.arccos(np.cos(2*np.pi/tp*t))+En
    return Edc,t,ts/nt

def find_gama(dx,xmax,nx):
    # bisection method for finding gama
    a=1
    b=2
    for it in range(0,50):
        gama=(a+b)/2
        f=dx*(gama**nx-1)/(gama-1)-xmax
        if f<=0:
            a=gama
        else:
            b=gama
        
        if abs(b-a)<=10**-8:
            break
    gama=(a+b)/2
    if gama>2:
        print("bad gama value")
        sys.exit()
    return gama

def Non_ranges(Ei,Ef,v,amplitude,frequency,E0,f,D,dx,nt,nx,alfa):
    # nondimensionalise all variables to suit simulation parameters,
    # final simulation is nondimensional
    amplitude=f*amplitude
    omega=2*np.pi*frequency
    omega=omega/f/v
    nt=int(2*nt*abs((f*Ei-f*Ef)))
    Edc,time,dt=potencial(Ei,Ef,v,amplitude,frequency,nt,E0)
    tau=1/(f*v)
    xmax=6*np.sqrt(abs(f*Ei-f*Ef))
    gama=find_gama(dx, xmax, nx)
    # print("gama value is: ",gama)
    t=time/tau
    Esin=amplitude*np.sin(omega*t)
    N=np.arange(nx+3)
    x=dx*(gama**N-1)/(gama-1)
    p=f*Edc+Esin
    p1=f*Edc
    p2=Esin
    return t,t[-1]/nt,nt,x,time,xmax,tau,p,p1,p2

def calc_K(alfa,k0,E0,p):
    # calculate EC reaction constants
    return k0*np.exp(-alfa*(p-E0)),k0*np.exp((1-alfa)*(p-E0))

def bound_function(X,alfa,k0,E0,weights,co,cr,gamac,rhou,dt,Gap,Gcp,pnom,delta):
    # boundary condition 6 equations to solve
    co0,cr0,Ga,Gb,Gc,pc=X
    Kf,Kb=calc_K(alfa, k0, E0, pc)
    # Neernst condition 
    # f1=co0-np.exp(pc)*cr0
    # BV condition
    f1=-Ga+Kf*co0-Kb*cr0
    f2=-Ga+np.dot(weights[1:3,1],co[1:3])+weights[0,1]*co0
    f3=-Gb+np.dot(weights[1:3,1],cr[1:3])+weights[0,1]*cr0
    f4=Ga+Gb 
    f5=rhou*gamac*Ga/dt+(1+rhou*gamac/dt)*Gc+delta*gamac-rhou*gamac*(Gap+Gcp)/dt
    f6=-rhou*Ga-rhou*Gc+pc-pnom
    return np.array([f1,f2,f3,f4,f5,f6])

def CV_simulation_explicit(pnom,params,numerical_params):
    alfa,k0,E0,Ru,A,Cdl=params
    dt,tau,f,n,R,Temp,F,cbulk,v,Ei,Ef,D,nt,t=numerical_params
    gamac=Cdl/(n*F*D**0.5*cbulk*A)*np.sqrt(R*Temp*v/n/F)
    rhou=Ru*f*(n*F*D**0.5*cbulk)*A*np.sqrt(f*v)
    k0=k0*np.sqrt(tau/D)
    E0=f*E0
    
    # precalc
    
    dx=np.sqrt(dt/0.45)
    xmax=6*np.sqrt(abs(f*Ei-f*Ef))
    nx=int(xmax/dx)
    # print(nx)
    # print(gamac)
    x=np.linspace(0,xmax,nx+2)
    
    weights=Fornberg_weights(x[0],x[0:7],3,1)
    Ga=[0]
    Gc=[0]
    p_corrected=[pnom[0]]
    
    co=np.ones(nx+2)
    cr=np.zeros(nx+2)
    delta=np.diff(pnom)/np.diff(t)
    for tt in range(1,nt):
        # calc boundary
        # co[1:-1]=co[1:-1]+0.45*(co[0:-2]-2*co[1:-1]+co[2:])
        # cr[1:-1]=cr[1:-1]+0.45*(cr[0:-2]-2*cr[1:-1]+cr[2:])
        
        guess=[co[0],cr[0],Ga[tt-1],-Ga[tt-1],Gc[tt-1],pnom[tt]]
        co[0],cr[0],ga,gb,gc,pc=sciop.root(bound_function,guess,args=(alfa,k0,E0,weights,co,cr,gamac,rhou,dt,Ga[tt-1],Gc[tt-1],pnom[tt],delta[tt-1])).x
        
        
        co_old=co.copy()
        cr_old=cr.copy()
        for i in range(1,nx+1):
            co[i]=co_old[i]+0.45*(co_old[i-1]-2*co_old[i]+co_old[i+1])
            cr[i]=cr_old[i]+0.45*(cr_old[i-1]-2*cr_old[i]+cr_old[i+1])
            
        Ga.append(ga)
        Gc.append(gc)
        # p_corrected.append(pc)
    G=-np.array(Ga)-np.array(Gc)
    return G

def rectangular(f,w0,w):
    return np.where(abs(f-w0)<=w,1,0)

def FFT_analysis(a,b,N,w):
    #poberem različne podatke iz knjižnjice
    V=b[0]
    f=b[1]
    maxI=[]
    #izračun dt, je pomemben za generacijo frekvenc
    dt=np.average(np.diff(a[2]))
    #generacija DC signala E, praviloma plotamo proti temu EDC, ne pa EAC, so lepši grafi.
    EDC=a[3]
    #matrika frekvenc, je narejena za kompleksno FFT
    freq=np.fft.fftfreq(a[2].shape[-1],d=dt)
    #FFT na podatke in plot podatkov
    sp=np.fft.fft(a[1])
    #za N harmonikov, od 0 do N, se generira okenska funkcija, ki odreže signal
    for i in range(N+1):
    #     #kopiram FFT
        if i==0:
            filter_sp=sp.copy()
            # window=rectangular(freq,i*f,5)
            #okenska funkcija za ustrezen harmonik
            window=rectangular(freq,i*f,w[i])
            #filtriranje in potem IFFT za podatke harmonika
            filter_sp=window*filter_sp
            Inew=np.real(np.fft.ifft(filter_sp))
            Inew=np.fft.ifft(filter_sp).real
            maxI.append(Inew)
        else:
            filter_sp=sp.copy()
            #okenska funkcija za ustrezen harmonik
            window=rectangular(freq,i*f,w[i])
            #filtriranje in potem IFFT za podatke harmonika
            filter_sp=window*filter_sp
            Inew=np.real(np.fft.ifft(filter_sp))
            Inew=np.fft.ifft(filter_sp).real
            anal_signal=np.abs(scisi.hilbert(Inew))
            maxI.append(anal_signal)
    return sp,freq,maxI

def FFT_analysis_loss(a,b,N,w):
    #poberem različne podatke iz knjižnjice
    dt=b[0]
    f=b[1]
    maxI=[]
    #matrika frekvenc, je narejena za kompleksno FFT
    freq=np.fft.fftfreq(a[1].shape[-1],d=dt)
    #FFT na podatke in plot podatkov
    sp=np.fft.fft(a[0])
    #za N harmonikov, od 0 do N, se generira okenska funkcija, ki odreže signal
    for i in range(N+1):
    #     #kopiram FFT
        if i==0:
            filter_sp=sp.copy()
            # window=rectangular(freq,i*f,5)
            #okenska funkcija za ustrezen harmonik
            window=rectangular(freq,i*f,w[i])
            #filtriranje in potem IFFT za podatke harmonika
            filter_sp=window*filter_sp
            Inew=np.real(np.fft.ifft(filter_sp))
            Inew=np.fft.ifft(filter_sp).real
            maxI.append(Inew)
        else:
            filter_sp=sp.copy()
            #okenska funkcija za ustrezen harmonik
            window=rectangular(freq,i*f,w[i])
            #filtriranje in potem IFFT za podatke harmonika
            filter_sp=window*filter_sp
            Inew=np.real(np.fft.ifft(filter_sp))
            Inew=np.fft.ifft(filter_sp).real
            anal_signal=np.abs(scisi.hilbert(Inew))
            maxI.append(anal_signal)
    return maxI
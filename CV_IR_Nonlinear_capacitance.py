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

def calc_main_coef(x,dt,nx):
    # calculate alfas and a's used in simulation
    a1=[]
    a2=[]
    a3=[]
    a4=[]
    
    for i in range(1,nx+1):
        
        weights=Fornberg_weights(x[i],x[i-1:i+4],4,2)
        
        alfa1=weights[0,2]
        alfa2=weights[1,2]
        alfa3=weights[2,2]
        alfa4=weights[3,2]
        
        a1.append((alfa2-1/dt)/alfa1)
        a2.append(alfa3/alfa1)
        a3.append(alfa4/alfa1)
        a4.append(-1/(dt*alfa1))
    
    return np.array(a1),np.array(a2),np.array(a3),np.array(a4)

def calc_K(alfa,k0,E0,p):
    # calculate EC reaction constants
    return k0*np.exp(-alfa*(p-E0)),k0*np.exp((1-alfa)*(p-E0))

def bound_function(X,alfa,k0,E0,Va,Ua,Vb,Ub,gamac,rhou,dt,Gap,Gcp,pnom,delta):
    # boundary condition 6 equations to solve
    co0,cr0,Ga,Gb,Gc,pc=X
    Kf,Kb=calc_K(alfa, k0, E0, pc)
    # Neernst condition 
    # f1=co0-np.exp(pc)*cr0
    # BV condition
    f1=-Ga+Kf*co0-Kb*cr0
    f2=-Va*co0+Ga-Ua
    f3=-Vb*cr0+Gb-Ub
    f4=Ga+Gb 
    f5=rhou*gamac(pc)*Ga/dt+(1+rhou*gamac(pc)/dt)*Gc-gamac(pc)*delta-rhou*gamac(pc)*(Gap+Gcp)/dt
    f6=rhou*Ga+rhou*Gc-pc+pnom
    return np.array([f1,f2,f3,f4,f5,f6])

def calc_boundary(nx,co,cr,ao,bo,ar,br,args):
    # function to calculate boundary values
    alfa,k0,E0,gamac,rhou,dt,Gap,Gcp,pnom,pc,delta,weights=args
    # calculate U-V's for both species
    uo=np.zeros(6)
    vo=np.ones(6)
    ur=np.zeros(6)
    vr=np.ones(6)
    
    for i in range(1,3):
        uo[i]=(bo[i-1]-uo[i-1])/ao[i-1]
        vo[i]=-vo[i-1]/ao[i-1]
        ur[i]=(br[i-1]-ur[i-1])/ar[i-1]
        vr[i]=-vr[i-1]/ar[i-1]
    # do the deriative using Fornberg
    Uo=np.matmul(weights[:4,1],uo[:4])
    Vo=np.matmul(weights[:4,1],vo[:4])
    Ur=np.matmul(weights[:4,1],ur[:4])
    Vr=np.matmul(weights[:4,1],vr[:4])
    
    # Initial estimate
    x=np.array([co[0],cr[0],Gap,-Gap,Gcp,pnom])
    # useing root function from scipy to find zeros of function
    sol=sciop.root(bound_function,x,args=(alfa,k0,E0,Vo,Uo,Vr,Ur,gamac,rhou,dt,Gap,Gcp,pnom,delta))
    x=sol.x
    
    return x

def time_step(nx,co,cr,ao1,ao2,ao3,ao4,ar1,ar2,ar3,ar4,args):
    # calculate a time step using Thomas algorithm 
    ao=np.zeros(nx)
    bo=np.zeros(nx)
    ar=np.zeros(nx)
    br=np.zeros(nx)
    
    ao[-1]=ao1[-1]
    bi=ao4[-1]*co[nx]
    bo[-1]=bi-ao2[-1]*co[nx+1]-ao3[-1]*co[nx+2]
    
    ar[-1]=ar1[-1]
    bi=ar4[-1]*cr[nx]
    br[-1]=bi-ar2[-1]*cr[nx+1]-ar3[-1]*cr[nx+2]
    
    ao[-2]=ao1[-2]-ao2[-2]/ao[-1]
    bi=ao4[-2]*co[nx-1]
    bo[-2]=bi-ao2[-1]*bo[-1]/ao[-1]-ao3[-1]*co[nx+1]
    
    ar[-2]=ar1[-2]-ar2[-2]/ar[-1]
    bi=ar4[-2]*cr[nx-1]
    br[-2]=bi-ar2[-1]*br[-1]/ar[-1]-ar3[-1]*cr[nx+1]
    
    for i in range(nx-3,-1,-1):
        ao[i]=ao1[i]-(ao2[i]-ao3[i]/ao[i+2])/ao[i+1]
        bi=ao4[i]*co[i+1]
        bo[i]=bi-ao2[i]*bo[i+1]/ao[i+1]-ao3[i]/ao[i+2]*(bo[i+2]-bo[i+1]/ao[i+1])
        
        ar[i]=ar1[i]-(ar2[i]-ar3[i]/ar[i+2])/ar[i+1]
        bi=ar4[i]*cr[i+1]
        br[i]=bi-ar2[i]*br[i+1]/ar[i+1]-ar3[i]/ar[i+2]*(br[i+2]-br[i+1]/ar[i+1])
    
    co[0],cr[0],ga,gb,gc,p=calc_boundary(nx,co,cr,ao,bo,ar,br,args)

    for i in range(1,nx+1):
        co[i]=(bo[i-1]-co[i-1])/ao[i-1]
        cr[i]=(br[i-1]-cr[i-1])/ar[i-1]
    

    return co,cr,ga,gb,gc,p

def CV_simulation(pnom,params,numerical_params):
    alfa,k0,E0,Ru,A=params[:5]
    C=np.array(params[5:])
    dx,dt,tau,f,n,R,Temp,F,cbulk,v,Ei,Ef,D,nx,nt,x,t=numerical_params
    
    # gamac=Cdl/(n*F*D**0.5*cbulk*A)*np.sqrt(R*Temp*v/n*F)
    rhou=Ru*f*(n*F*D**0.5*cbulk)*A*np.sqrt(f*v)
    gamac=[]
    for i in range(len(C)):
        gamac.append(C[i]/(n*F*D**0.5*cbulk*A)*np.sqrt(v/f)/(f**i))
    
    gamac=np.poly1d(np.flip(gamac))
    
    k0=k0*np.sqrt(tau/D)
    E0=f*E0
    
    # precalc
    # 
    weights=Fornberg_weights(x[0],x[0:7],3,1)
    # print(weights)
    Ga=[0]
    Gc=[0]
    p_corrected=[pnom[0]]
    # calc a's
    ao1,ao2,ao3,ao4=calc_main_coef(x, dt,nx)
    bo1,bo2,bo3,bo4=calc_main_coef(x, dt/2,nx)
    ar1,ar2,ar3,ar4=calc_main_coef(x, dt,nx)
    br1,br2,br3,br4=calc_main_coef(x, dt/2,nx)
    
    co=1*np.ones(nx+3)
    cr=0*np.ones(nx+3)
    # dummy_o=np.ones(nx+3)
    # dummy_r=np.zeros(nx+3)
    delta=np.diff(pnom)/np.diff(t)
    # time steps
    for tt in range(1,int(nt)):
        co,cr,ga,gb,gc,p=time_step(nx,co,cr,ao1,ao2,ao3,ao4,ar1,ar2,ar3,ar4,[alfa,k0,E0,gamac,rhou,dt,Ga[tt-1],Gc[tt-1],pnom[tt],
                                                                             p_corrected[tt-1],delta[tt-1],weights])
        # dummy_o,dummy_r,ga_du,gb_du,gc_du,p_du=time_step(nx,dummy_o,dummy_r,bo1,bo2,bo3,bo4,br1,br2,br3,br4,[alfa,k0,gamac,rhou,dt,Ga[tt-1],Gc[tt-1],
        #                                                                      pnom[tt-1]+dt/2,
        #                                                                      p_corrected[tt-1],pnom[tt]-pnom[tt-1],weights])
        # dummy_o,dummy_r,ga_du,gb_du,gc_du,p_du=time_step(nx,dummy_o,dummy_r,bo1,bo2,bo3,bo4,br1,br2,br3,br4,[alfa,k0,gamac,rhou,dt,ga_du,gc_du,pnom[tt],
        #                                                                      p_du,pnom[tt]-pnom[tt-1],weights])
        # # using extrapolation 2nd order
        # co=2*dummy_o-co
        # cr=2*dummy_r-cr
        # ga=2*ga_du-ga
        # gb=2*gb_du-gb
        # gc=2*gc_du-gc
        # p=2*p_du-p
        Ga.append(ga)
        # Ga.append(np.matmul(weights[:3,1],co[:3]))
        Gc.append(gc)
        p_corrected.append(p)
        # if tt%1000==0:
        #     plt.figure("conc profile")
        #     plt.plot(x,co,linestyle="--",marker="o")
        
    Ga=-np.array(Ga)
    Gc=np.array(Gc)
    G=Ga+Gc
    p_corrected=np.array(p_corrected)
        
    return pnom,G,t

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
            # Inew=np.real(np.fft.ifft(filter_sp))
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
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 11:34:58 2021

@author: Ožbej
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sciop
import scipy.signal as scisi
import sys
import time

#constants

R=8.314
Temp=298
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
D=7.2*10**-9
nx=50

#time constants
nt=100
#bulk koncentracija
cbulk=1

#ključna kinetična parametra
alfa=0.5
k0=10**3

# homogena kintika
# first order kinetics
ko1=0
kr1=0
# second order kinetics
ko2=0
kr2=0

#glavni konstanti pri ac-cv, amplituda in frekvenca sinusa
frequency=9
amplitude=0.1

#parametri za kapacitivnost
Cdl=0.000
Ru=0

A=10**-4
#parametri za konvekcijo
omega=100
ni=10**-5

#parametri za adsorbcijo
bo=0
br=000
max_coverage=10**-10

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
    Edc=a-2*a/np.pi*np.arccos(np.cos(2*np.pi/tp*t))+En-E0
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

def Non_ranges(Ei,Ef,v,amplitude,frequency,E0,f,D,dx,nt,nx,alfa,k0):
    # nondimensionalise all variables to suit simulation parameters,
    # final simulation is nondimensional
    amplitude=f*amplitude
    omega=2*np.pi*frequency
    omega=omega/f/v
    tau=1/(f*v)
    nt=int(2*nt*abs((f*(Ei)-f*(Ef))))
    Edc,time,dt1=potencial(Ei,Ef,v,amplitude,frequency,nt,E0)
    xmax=6*np.sqrt(abs(f*(Ei)-f*(Ef)))
    gama=find_gama(dx, xmax, nx)
    t=time/tau
    Esin=amplitude*np.sin(omega*t)
    N=np.arange(nx+3)
    x=dx*(gama**N-1)/(gama-1)
    p=f*Edc+Esin
    p1=f*Edc
    p2=Esin
    k0=k0*np.sqrt(tau/D)
    return t,t[-1]/nt,nt,x,time,np.sqrt(D*tau),tau,k0,p,p1,p2

def calc_main_coef(x,dt,nx,K1,K2,B):
    # calculate alfas and a's used in simulation
    a1=[]
    a2=[]
    a3=[]
    a4=[]
    a5=[]
    ak=[]
    
    for i in range(1,nx+1):
        
        weights=Fornberg_weights(x[i],x[i-1:i+4],4,2)
        
        alfa1=weights[0,2]-(B*x[i]**2)*weights[0,1]
        alfa2=weights[1,2]-(B*x[i]**2)*weights[1,1]
        alfa3=weights[2,2]-(B*x[i]**2)*weights[2,1]
        alfa4=weights[3,2]-(B*x[i]**2)*weights[3,1]
        
        a1.append((alfa2-1/dt+K1)/alfa1)
        a2.append(alfa3/alfa1)
        a3.append(alfa4/alfa1)
        a4.append(-1/(dt*alfa1))
        a5.append(2*K2/alfa1)
        ak.append(4*K2/alfa1)
    
    return np.array(a1),np.array(a2),np.array(a3),np.array(a4),np.array(a5),np.array(ak)

def Isotherm(thetaa,thetab,params):
    return thetaa/(1-thetaa-thetab)

def calc_K(alfa,k0,p):
    # calculate EC reaction constants
    return k0*np.exp(-alfa*p),k0*np.exp((1-alfa)*p)

def bound_function(X,alfa,k0,Va,Ua,Vb,Ub,gamac,rhou,dt,Gap,Gcp,pnom,delta,Bo,Br,Kad,Ktheta,thetaop,thetarp):
    # boundary condition 6 equations to solve
    co0,cr0,thetao,thetar,Gc,pc=X
    Kf,Kb=calc_K(alfa, k0, pc)
    Ga=Va*co0+Ua+(thetao-thetaop)/dt
    
    
    f1=Kad*(Ua+Va*co0)+Kad*(Ub+Vb*cr0)-(thetao-thetaop)/dt-(thetar-thetarp)/dt
    
    f2=Bo*co0-Isotherm(thetao,thetar,0)
    
    f3=+Br*cr0-Isotherm(thetar,thetao,0)
    
    f4=-Kad*(Ua+Va*co0)+(Kad*Kf*co0-Kad*Kb*cr0)+(Kf*Ktheta*thetao-Kb*Ktheta*thetar)+(thetao-thetaop)/dt
    
    
    
    if delta >= 0: 
        f5=rhou*gamac*Ga/dt+(1+rhou*gamac/dt)*Gc-gamac-rhou*gamac*(Gap+Gcp)/dt
    else:
        f5=rhou*gamac*Ga/dt+(1+rhou*gamac/dt)*Gc+gamac-rhou*gamac*(Gap+Gcp)/dt
    f6=-rhou*Ga-rhou*Gc+pc-pnom
    return np.array([f1,f2,f3,f4,f5,f6])

def calc_boundary(nx,co,cr,ao,bo,ar,br,args):
    # function to calculate boundary values
    alfa,k0,gamac,rhou,dt,Gap,Gcp,pnom,pc,delta,weights,Ko1,Kr1,Ko2,Kr2,Kad,Bo,Br,Ktheta,thetaop,thetarp=args
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
    Uo=np.matmul(weights[:3,1],uo[:3])
    Vo=np.matmul(weights[:3,1],vo[:3])
    Ur=np.matmul(weights[:3,1],ur[:3])
    Vr=np.matmul(weights[:3,1],vr[:3])
    
    # Initial estimate
    x=np.array([co[0],cr[0],thetaop,thetarp,Gcp,pnom])
    # useing root function from scipy to find zeros of function
    sol=sciop.root(bound_function,x,args=(alfa,k0,Vo,Uo,Vr,Ur,gamac,rhou,dt,Gap,Gcp,pnom,delta,Bo,Br,Kad,Ktheta,thetaop,thetarp))
    x=sol.x
    
    # BV case without IR
    # Kf,Kb=calc_K(alfa, k0, pnom)
    # A=np.array([[-Vo,-Vr],
    #             [Kf-Vo,-Kb]])
    # B=np.array([Uo+Ur,
    #             Uo])
    # C=np.linalg.solve(A,B)
    # x=np.zeros(6)
    # x[0]=C[0]
    # x[1]=C[1]
    
    # Neernst case without IR
    # x[0]=(-Uo-Ur)/(Vo+np.exp(-pnom)*Vr)
    # x[1]=np.exp(-pnom)*x[0]
    
    return x

def time_step(nx,co,cr,ao1,ao2,ao3,ao4,ao5,aok,ar1,ar2,ar3,ar4,ar5,ark,args):
    # correct bound values to fit homogenious reaction
    alfa,k0,gamac,rhou,dt,Gap,Gcp,pnom,pc,delta,weights,Ko1,Kr1,Ko2,Kr2,Kad,Bo,Br,Ktheta,thetaop,thetarp=args
    co[nx+2]=(1/dt*co[nx+2]-2*Ko2*co[nx+2]**2)/(1/dt-4*Ko2*co[nx+2]-Ko1)
    co[nx+1]=(1/dt*co[nx+1]-2*Ko2*co[nx+1]**2)/(1/dt-4*Ko2*co[nx+1]-Ko1)
    cr[nx+2]=(1/dt*cr[nx+2]-2*Kr2*cr[nx+2]**2)/(1/dt-4*Kr2*cr[nx+2]-Kr1)
    cr[nx+1]=(1/dt*cr[nx+1]-2*Kr2*cr[nx+1]**2)/(1/dt-4*Kr2*cr[nx+1]-Kr1)
    # calculate a time step using Thomas algorithm 
    ao=np.zeros(nx)
    bo=np.zeros(nx)
    ar=np.zeros(nx)
    br=np.zeros(nx)
    
    ao[-1]=ao1[-1]+aok[-1]*co[nx]
    bi=ao4[-1]*co[nx]+ao5[-1]*co[nx]**2
    bo[-1]=bi-ao2[-1]*co[nx+1]-ao3[-1]*co[nx+2]
    
    ar[-1]=ar1[-1]+ark[-1]*cr[nx]
    bi=ar4[-1]*cr[nx]+ar5[-1]*cr[nx]**2
    br[-1]=bi-ar2[-1]*cr[nx+1]-ar3[-1]*cr[nx+2]
    
    ao[-2]=ao1[-2]-ao2[-2]/ao[-1]+aok[-2]*co[nx-1]
    bi=ao4[-2]*co[nx-1]+ao5[-2]*co[nx-1]**2
    bo[-2]=bi-ao2[-1]*bo[-1]/ao[-1]-ao3[-1]*co[nx+1]
    
    ar[-2]=ar1[-2]-ar2[-2]/ar[-1]+ark[-2]*cr[nx-1]
    bi=ar4[-2]*cr[nx-1]+ar5[-2]*cr[nx-1]**2
    br[-2]=bi-ar2[-1]*br[-1]/ar[-1]-ar3[-1]*cr[nx+1]
    
    for i in range(nx-3,-1,-1):
        ao[i]=ao1[i]+aok[i]*co[i+1]-(ao2[i]-ao3[i]/ao[i+2])/ao[i+1]
        bi=ao4[i]*co[i+1]+ao5[i]*co[i+1]**2
        bo[i]=bi-ao2[i]*bo[i+1]/ao[i+1]-ao3[i]/ao[i+2]*(bo[i+2]-bo[i+1]/ao[i+1])
        
        ar[i]=ar1[i]+ark[i]*cr[i+1]-(ar2[i]-ar3[i]/ar[i+2])/ar[i+1]
        bi=ar4[i]*cr[i+1]+ar5[i]*cr[i+1]**2
        br[i]=bi-ar2[i]*br[i+1]/ar[i+1]-ar3[i]/ar[i+2]*(br[i+2]-br[i+1]/ar[i+1])
    
    
    co[0],cr[0],thetao,thetar,gc,p=calc_boundary(nx,co,cr,ao,bo,ar,br,args)

    for i in range(1,nx+1):
        co[i]=(bo[i-1]-co[i-1])/ao[i-1]
        cr[i]=(br[i-1]-cr[i-1])/ar[i-1]

    return co,cr,thetao,thetar,gc,p

def CV_simulation(f,n,R,Temp,F,cbulk,v,Ei,Ef,E0,D,nx,nt,alfa,k0,frequency,amplitude,Cdl,Ru,ko1,kr1,ko2,kr2,omega,ni,bo,br,max_coverage):
    dx=10**-3
    
    gamac=Cdl/(n*F*D**0.5*cbulk)*np.sqrt(R*Temp*v/n*F)
    rhou=Ru*f*n*F*D**0.5*cbulk*np.sqrt(f*v)
    
    # precalc
    t,dt,nt,x,time,delta,tau,k0,pnom,pdc,psin=Non_ranges(Ei,Ef,v,amplitude,frequency,E0,f,D,dx,nt,nx,alfa,k0)
    Ko1=ko1*tau
    Kr1=kr1*tau
    Ko2=ko2*cbulk*tau
    Kr2=kr2*cbulk*tau
    Kad=cbulk*np.sqrt(D*tau)/max_coverage
    Ktheta=np.sqrt(D*tau)
    Bo=bo*cbulk
    Br=br*cbulk
    B=-0.51*omega**1.5*ni**-0.5*delta*tau
    weights=Fornberg_weights(x[0],x[0:7],3,1)
    Ga=[0]
    Gc=[0]
    Gt=[0]
    p_corrected=[pnom[0]]
    # calc a's
    ao1,ao2,ao3,ao4,ao5,aok=calc_main_coef(x, dt,nx,Ko1,Ko2,B)
    # bo1,bo2,bo3,bo4,bo5,bok=calc_main_coef(x, dt/2,nx,Ko1,Ko2,B)
    ar1,ar2,ar3,ar4,ar5,ark=calc_main_coef(x, dt,nx,Kr1,Kr2,B)
    # br1,br2,br3,br4,br5,brk=calc_main_coef(x, dt/2,nx,Kr1,Kr2,B)
    co=np.ones(nx+3)
    cr=np.zeros(nx+3)
    thetao=0.9999
    thetar=0
    # dummy_o=np.ones(nx+3)
    # dummy_r=np.zeros(nx+3)
    theta=[[thetao,thetar]]
    # time steps
    for tt in range(1,int(nt)):
        args=[alfa,k0,gamac,rhou,dt,Ga[tt-1],Gc[tt-1],pnom[tt],
             p_corrected[tt-1],pnom[tt]-pnom[tt-1],weights,Ko1,Kr1,Ko2,Kr2,Kad,Bo,Br,Ktheta,thetao,thetar]
        co,cr,thetao,thetar,gc,p=time_step(nx,co,cr,ao1,ao2,ao3,ao4,ao5,aok,ar1,ar2,ar3,ar4,ar5,ark,args)
        # dummy_o,dummy_r,ga_du,gb_du,gc_du,p_du=time_step(nx,dummy_o,dummy_r,bo1,bo2,bo3,bo4,bo5,bok,br1,br2,br3,br4,br5,brk,[alfa,k0,gamac,rhou,dt,Ga[tt-1],Gc[tt-1],
        #                                                                      pnom[tt-1]+dt/2,
        #                                                                      p_corrected[tt-1],pnom[tt]-pnom[tt-1],weights,Ko1,Kr1,Ko2,Kr2])
        # dummy_o,dummy_r,ga_du,gb_du,gc_du,p_du=time_step(nx,dummy_o,dummy_r,bo1,bo2,bo3,bo4,bo5,bok,br1,br2,br3,br4,br5,brk,[alfa,k0,gamac,rhou,dt,ga_du,gc_du,pnom[tt],
        #                                                                      p_du,pnom[tt]-pnom[tt-1],weights,Ko1,Kr1,Ko2,Kr2])
        # using extrapolation 2nd order
        # co=2*dummy_o-co
        # cr=2*dummy_r-cr
        # ga=2*ga_du-ga
        # gb=2*gb_du-gb
        # gc=2*gc_du-gc
        # p=2*p_du-p
        theta.append([thetao,thetar])
        Ga.append(np.matmul(weights[:3,1],co[:3]))
        Gt.append((theta[tt][0]-theta[tt-1][0])/dt)
        # Ga.append(np.matmul(weights[:3,1],co[:3]))
        Gc.append(gc)
        p_corrected.append(p)
        # if tt%1000==0:
        #     plt.figure("conc profile")
        #     plt.plot(x,co,linestyle="--",marker="o",color="blue")
        #     plt.plot(x,cr,linestyle="--",marker="o",color="orange")
        
    Ga=-np.array(Ga)
    Gc=np.array(Gc)
    G=Ga+Gc+Gt
    p_corrected=np.array(p_corrected)
        
    return pnom,G,t,p_corrected,Ga,Gc,Gt,pdc

def rectangular(f,w0,w):
    return np.where(abs(f-w0)<=w,1,0)

def FFT_analysis(a,b,N):
    #poberem različne podatke iz knjižnjice
    V=b[0]
    f=b[1]
    maxI=[[123,f,V]]
    # print(np.average((np.diff(a[:,0])/np.diff(a[:,2]))))
    #izračun dt, je pomemben za generacijo frekvenc
    dt=np.average(np.diff(a[2]))
    #generacija DC signala E, praviloma plotamo proti temu EDC, ne pa EAC, so lepši grafi.
    EDC=a[3]
    # plt.plot(a[:,0])
    # plt.plot(EDC)
    # sys.exit()
    j=0
    #Plot za eksperimentalne podatke
    # fig,(ax1,ax2)=plt.subplots(2)
    # plt.suptitle('Experimental data')
    # ax1.plot(EDC[:],a[1])
    # ax2.plot(a[2],a[1])
    # plt.savefig('Experimental_Data_freq_'+'.png')
    # plt.close(fig)
    #matrika frekvenc, je narejena za kompleksno FFT
    freq=np.fft.fftfreq(a[2].shape[-1],d=dt)
    #FFT na podatke in plot podatkov
    sp=np.fft.fft(a[1])
    plt.figure("FT")
    plt.plot(freq,np.log10(sp))
    plt.xlabel("freq [/]")
    plt.ylabel("log(Amplitude)")
    # sys.exit()
    #za N harmonikov, od 0 do N, se generira okenska funkcija, ki odreže signal
    for i in range(N+1):
    #     #kopiram FFT
        if i==0:
            filter_sp=sp.copy()
            # window=rectangular(freq,i*f,5)
            #okenska funkcija za ustrezen harmonik
            w=0.4
            window=rectangular(freq,i*f,w)
            #filtriranje in potem IFFT za podatke harmonika
            filter_sp=window*filter_sp
            Inew=np.real(np.fft.ifft(filter_sp))
            Inew=np.fft.ifft(filter_sp).real
            # high,low=hl_envelopes_idx(Inew)
            #plotanje harmonika
            plt.figure('hamonik '+str(i))
            plt.title('hamonik '+str(i))
            # plt.plot(a[j:,2]-a[j,2],Inew)
            # plt.plot(a[:,2],Inew)
            plt.plot(EDC[:],Inew)
            plt.xlabel("p [/]")
            plt.ylabel("G [/]")
            # plt.hlines(max(Inew[int(0.2*len(Inew)):int(0.8*len(Inew))]),0,a[-1,2],linestyle='dashed')
            # plt.text(0.2*a[-1,2],round(0.9*max(Inew),5),str(round(max(Inew),5))+'mA')
            # plt.savefig('hamonik '+'_'+str(i)+'.png')
            plt.close(fig='hamonik '+str(i)+str(V))
            maxI.append([i,i*f,max(Inew)])
        else:
            filter_sp=sp.copy()
            # window=rectangular(freq,i*f,5)
            #okenska funkcija za ustrezen harmonik
            if i==1:
                w=1
            elif i==2 or i==3 or i==4:
                w=1
            elif i>=5:
                w=1
            window=rectangular(freq,i*f,w)
            #filtriranje in potem IFFT za podatke harmonika
            filter_sp=window*filter_sp
            Inew=np.real(np.fft.ifft(filter_sp))
            Inew=np.fft.ifft(filter_sp).real
            anal_signal=np.abs(scisi.hilbert(Inew))
            # high,low=hl_envelopes_idx(Inew)
            #plotanje harmonika
            # plt.figure('hamonik '+str(i)+" forward")
            # plt.title('hamonik '+str(i)+" forward")
            # # plt.plot(a[j:,2]-a[j,2],Inew)
            # # plt.plot(a[j:,2]-a[j,2],anal_signal,'r')
            # # plt.plot(a[:,2],Inew)
            # # plt.plot(a[:,2],anal_signal[:],'r')
            # # plt.plot(EDC[:int(len(Inew)/2)],Inew[:int(len(Inew)/2)],label=label)
            # plt.plot(EDC[:int(len(Inew)/2)],anal_signal[:int(len(Inew)/2)],label=label,color="b")
            
            # plt.figure('hamonik '+str(i)+" backward")
            # plt.title('hamonik '+str(i)+" backward")
            # # plt.plot(a[j:,2]-a[j,2],Inew)
            # # plt.plot(a[j:,2]-a[j,2],anal_signal,'r')
            # # plt.plot(a[:,2],Inew)
            # # plt.plot(a[:,2],anal_signal[:],'r')
            # # plt.plot(EDC[int(len(Inew)/2):],Inew[int(len(Inew)/2):],label=label)
            # plt.plot(EDC[int(len(Inew)/2):],anal_signal[int(len(Inew)/2):],label=label,color="b")
            
            plt.figure(str(i)+"_harmonic")
            plt.title('hamonik '+str(i))
            # plt.plot(a[j:,2]-a[j,2],Inew)
            # plt.plot(a[j:,2]-a[j,2],anal_signal,'r')
            # plt.plot(a[:,2],Inew)
            # plt.plot(a[:,2],anal_signal[:],'r')
            # plt.plot(EDC,Inew)
            plt.plot(EDC,anal_signal)
            plt.xlabel("p [/]")
            plt.ylabel("G [/]")
            # plt.hlines(max(Inew[int(0.2*len(Inew)):int(0.8*len(Inew))]),0,a[-1,2],linestyle='dashed')
            # plt.text(0.2*a[-1,2],round(0.9*max(Inew),5),str(round(max(Inew),5))+'mA')
            # plt.savefig('hamonik '+'_'+str(i)+'_'+' osnovna frekvenca '+'_'+str(round(f,2))+'Hz'+' amplituda '+'_'+str(round(V,2))+'mA'+'.png')
            # plt.close(fig='hamonik '+str(i)+' osnovna frekvenca '+'_'+str(round(f,2))+'Hz'+' amplituda '+'_'+str(round(V,2))+'mA')
            maxI.append([i,i*f,max(Inew)])
    # plt.figure('Fourier transform'+str(f)+'_'+str(V))
    # plt.plot(freq,sp)
    return np.array(maxI)

Es,I,t,Ec,If,Ic,Iad,Edc=CV_simulation(f,n,R,Temp,F,cbulk,v,Ei,Ef,0,D,nx,nt,alfa,k0,frequency,amplitude,Cdl,Ru,ko1,kr1,ko2,kr2,omega,ni,bo,br,max_coverage)  
Es=Es/f
Edc=Edc/f
# Edc=Edc[10:]
# I=I[10:]*n*F*A*cbulk*np.sqrt(n*F*D*v/R/Temp)
tau=1/(f*v)
delta=np.sqrt(D*tau)
# plt.figure("CV")
# plt.plot(Edc,I)
# plt.hlines(-0.62*n*F*A*cbulk*D**(2/3)*omega**0.5*ni**(-1/6),Ei,Ef,color="black",linestyle="--")
# print(min(I))
# print(-0.62*n*F*A*cbulk*D**(2/3)*omega**0.5*ni**(-1/6))
# plt.plot(Edc,If)
# plt.plot(Edc,If)
# plt.plot(Edc,Iad)
# plt.plot(Edc,If+Iad)


a=[Es[10:],If[10:],t[10:],Edc[10:]]
amplitude=f*amplitude
omega=frequency/f/v

b=[amplitude,omega]
N=5
anal=FFT_analysis(a,b,N)
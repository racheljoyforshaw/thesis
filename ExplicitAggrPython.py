#preamble
#cd 'C:\Thesis'!!!!!!!!!!!!
import numpy as np
from numpy import linalg
from scipy.interpolate import griddata
from scipy.interpolate import interp2d

#defining the transition matrix
DurationUnempG = 1.5
DurationUnempB = 2.5
corr = .25

UnempG = .04
UnempB = .1
Unemp = np.array([[UnempG], [UnempB]])

DurationZG = 8.0
DurationZB = 8.0

pZG = 1.0-1.0/DurationZG
pZB = 1.0-1.0/DurationZB

PZ = np.array([[pZG, 1-pZG],[1-pZB, pZB]]) 

p22 = 1.0-1.0/DurationUnempG
p21 = 1.0-p22
p11 = ((1.0-UnempG)-UnempG*p21)/(1.0-UnempG)

P11 = np.array([[p11,1-p11],[p21,p22]])

p22 = 1.0-1.0/DurationUnempB
p21 = 1.0-p22
p11 = ((1.0-UnempB)-UnempB*p21)/(1.0-UnempB)

P00 = np.array([[p11,1-p11],[p21,p22]])

p22 = (1+corr)*p22
p21 = 1-p22
p11 = ((1-UnempB)-UnempG*p21)/(1-UnempG)

P10 = np.array([[p11,1-p11],[p21,p22]])

p22 = (1-corr)*(1-1/DurationUnempG)
p21 = 1-p22
p11 = ((1-UnempG)-UnempB*p21)/(1-UnempB)

P01 = np.array([[p11,1-p11],[p21,p22]])

P=np.vstack([np.hstack([PZ[0,0]*P11,PZ[0,1]*P10]),np.hstack([PZ[1,0]*P01,PZ[1,1]*P00])]) 

# Model Parameters

alpha  = 0.36           
beta   = 0.99
delta  = .025
sigma  = 1.0
phi    = 0
Nk     = 250
NK     = 12
UI     = 0.15
h      = 1/(1-UnempB)
tau    = np.array([UI*UnempG/(h*(1-UnempG)),UI*UnempB/(h*(1-UnempB))])
BiasCorrection = 1

if BiasCorrection==1:
    de = 0.01257504725554
    du = 0.03680683961167
else:
    de=0
    du=0

deltaZ = 0.01

# Grid

kmin   = phi
kmax   = 200

zg = 1+deltaZ
zb = 1-deltaZ
Z = np.array([[zg],[zb]])
ZZ = Z 

Kumin = 33
Kemin = 35
Kumax = 42.5
Kemax = 43.5

kptemp = np.transpose(np.linspace(0,np.log(kmax+1-kmin),num=Nk)) #row vector of Nk points linearly spaced between 0 and log(kmax+1-kmin)
kp = np.exp(kptemp)-1+kmin

Ke = np.transpose(np.linspace(Kemin,Kemax,num=NK))
Ku = np.transpose(np.linspace(Kumin,Kumax,num=NK))


#Initial policy functions

# Individual policy function
kpp1=np.empty(shape=(Nk,NK,NK,2,2))
kpp2=np.empty(shape=(Nk,NK,NK,2,2))
for ii in range(0,2):
    for i in range(0,2):
        for j in range(0,NK):
            for l in range(0,NK):
                kpp1[:,j,l,i,ii] = (1.0-delta)*kp
                kpp2[:,j,l,i,ii] = .3*(1-delta)*kp
                
                
def ndmesh(*args):
    args=map(np.asarray,args)
    return np.broadcast_arrays(*[x[(slice(None),)+(None,)*i] for i, x in enumerate(args)])
    

Kemat,kmat,Kumat = ndmesh(Ke,kp,Ku)   # nb. returns 12x250x12 instead of 250x12x12 !!!!!!!!!!!!!!!

#Initial aggregate policy function (guess: unit root.)

A=[linalg.pinv(np.array([[P11[0,0], P11[1,0]*UnempG/(1-UnempG)], [P11[0,1]*(1-UnempG)/UnempG, \
P11[1,1]]])),linalg.pinv(np.array([[P00[0,0], P00[1,0]*UnempB/(1-UnempB)], [P00[0,1]*(1-UnempB)/UnempB, \
P00[1,1]]]))]


Kpe=np.empty(shape=(12,12,2))
Kpu=np.empty(shape=(12,12,2))
for ii in range(0,2):
    for i in range(0,NK):
        for j in range(0,NK):
            Kpeu = np.dot(A[:][:][ii],np.array([[Ke[i]],[Ku[j]]]))
            Kpe[i,j,ii] = Kpeu[0]
            Kpu[i,j,ii] = Kpeu[1]


KpeN=np.empty(shape=(12,12,2,2))
KpeN[:,:,0,0] = np.divide((P11[0,0]*(1-UnempG)*Kpe[:,:,0]+P11[1,0]*UnempG*Kpu[:,:,0]),(1-UnempG))
KpeN[:,:,0,1] = np.divide((P01[0,0]*(1-UnempB)*Kpe[:,:,1]+P01[1,0]*UnempB*Kpu[:,:,1]),(1-UnempG))
KpeN[:,:,1,0] = np.divide(P10[0,0]*(1-UnempG)*Kpe[:,:,0]+P10[1,0]*UnempG*Kpu[:,:,0],(1-UnempB))
KpeN[:,:,1,1] = np.divide(P00[0,0]*(1-UnempB)*Kpe[:,:,1]+P00[1,0]*UnempB*Kpu[:,:,1],(1-UnempB))

KpuN=np.empty(shape=(12,12,2,2))
KpuN[:,:,0,0] = np.divide(P11[0,1]*(1-UnempG)*Kpe[:,:,0]+P11[1,1]*UnempG*Kpu[:,:,0],(UnempG))
KpuN[:,:,0,1] = np.divide(P01[0,1]*(1-UnempB)*Kpe[:,:,1]+P01[1,1]*UnempB*Kpu[:,:,1],(UnempG))
KpuN[:,:,1,0] = np.divide(P10[0,1]*(1-UnempG)*Kpe[:,:,0]+P10[1,1]*UnempG*Kpu[:,:,0],(UnempB))
KpuN[:,:,1,1] = np.divide(P00[0,1]*(1-UnempB)*Kpe[:,:,1]+P00[1,1]*UnempB*Kpu[:,:,1],(UnempB))


KpeN = np.minimum(KpeN,np.amax(Ke))
KpeN = np.maximum(KpeN,np.amin(Ke))
KpuN = np.minimum(KpuN,np.amax(Ku))
KpuN = np.maximum(KpuN,np.amin(Ku))

# Solving the individual's Euler equation with endogenous gridpoints.

ConvCrit = 1.0
s = 0
k=np.empty(shape=(250,12,12,2,2))

print 'Solving the individual and aggregate problem until ConvCrit<1e-6'

while ConvCrit>1e-6:
    s = s+1
    for i in range(0,2):
        for j in range(0,NK):
            for l in range(0,NK):
                Kp = ((1-Unemp[i])*Kpe[j,l,i]+Unemp[i]*Kpu[j,l,i])
                Rp = np.transpose(np.array([(1+alpha*zg*(Kp/(h*(1-UnempG)))**(alpha-1)-delta),(1+alpha*zg*(Kp/(h*(1-UnempG)))**(alpha-1)-delta),(1+alpha*zb*(Kp/(h*(1-UnempB)))**(alpha-1)-delta),(1+alpha*zb*(Kp/(h*(1-UnempB)))**(alpha-1)-delta)]))
                Wp = np.transpose(np.array([h*(1-alpha)*zg*(Kp/(h*(1-UnempG)))**(alpha)*(1-tau[0]),(1-alpha)*zg*(Kp/(h*(1-UnempG)))**(alpha),h*(1-alpha)*zb*(Kp/(h*(1-UnempB)))**(alpha)*(1-tau[1]),(1-alpha)*zb*(Kp/(h*(1-UnempB)))**(alpha)]))
                RHS = np.transpose(np.array([(beta*Rp[0,0]*(Rp[0,0]*kp+Wp[0,0]-kpp1[:,j,l,0,i]))**(-sigma),(beta*Rp[0,1]*(Rp[0,1]*kp+UI*Wp[0,1]-kpp2[:,j,l,0,i]))**(-sigma),(beta*Rp[0,2]*(Rp[0,2]*kp+Wp[0,2]-kpp1[:,j,l,1,i]))**(-sigma),(beta*Rp[0,3]*(Rp[0,3]*kp+UI*Wp[0,3]-kpp2[:,j,l,1,i]))**(-sigma)]))
                C1 = np.dot(RHS,np.transpose(P[i*2,:]))**(-1/sigma) 
                C2 = np.dot(RHS,np.transpose(P[i*2+1,:]))**(-1/sigma)
                K = (1-Unemp[i])*Ke[j]+Unemp[i]*Ku[l]
                k[:,j,l,0,i] = (C1-h*(1-alpha)*Z[i]*(K/(h*(1-Unemp[i])))**(alpha)*(1-tau[i])+kp)/((1+alpha*Z[i]*(K/(h*(1-Unemp[i])))**(alpha-1)-delta))
                k[:,j,l,1,i] = (C2-UI*(1-alpha)*Z[i]*(K/(h*(1-Unemp[i])))**(alpha)+kp)/((1+alpha*Z[i]*(K/(h*(1-Unemp[i])))**(alpha-1)-delta))
                
    kpmat=np.empty(shape=(250,12,12,2,2))
                              
    for i in range(0,2):
        for j in range(0,NK):
            for l in range(0,NK):
                if np.amin(k[:,j,l,0,i])>0:
                    #kpmat(:,j,l,1,i) = interp1([0;k(:,j,l,1,i)],[0;kp],kp,[],'extrap')
                    #interp1(x,Y,xi) function Y at points xi
                    #checked -  very close to Matab output
                    kpmat[:,j,l,0,i] = np.interp(kp,np.hstack([[0],k[:,j,l,0,i]]),np.hstack([[0],kp])) 
                else:
                    kpmat[:,j,l,0,i] = np.interp(kp,k[:,j,l,0,i],kp)
            
                if np.amin(k[:,j,l,1,i])>0:
                    kpmat[:,j,l,1,i] = np.interp(kp,np.hstack([[0],k[:,j,l,1,i]]),np.hstack([[0],kp]))
                else:
                    kpmat[:,j,l,1,i] = np.interp(kp,k[:,j,l,1,i],kp)

#SHIT CHECKING ENDS HERE##################
# Update the individual policy functions for t+1:
    for ii in range(0,2):
        for i in range(0,2):
            for j in range (0,NK):
                for l in range(0,NK):
                    #interp3(X,Y,Z,V,XI,YI,ZI) values of underlying 3D function V at points in arrays XI, YI and ZI
                    #X=Kemat, Y = kmat, Z=Kumat, V=kpmat(:,:,:,1,i) XI= KpeN(j,l,i,ii), YI = kp, ZI = KpuN
                    #griddata(points - data coordinates, values - data values, xi - points at which to interpolate data)
                    ################################################## does't work
                    kpp1[:,j,l,i,ii] = griddata((KpeN[j,l,i,ii],kp,KpuN[j,l,i,ii]),kpmat[:,:,:,0,i],(Kemat,kmat,Kumat))
                    kpp1[:,j,l,i,ii] = griddata((KpeN[j,l,i,ii],kp,KpuN[j,l,i,ii]),kpmat[:,:,:,1,i],(Kemat,kmat,Kumat))
                    kpp2New[:,j,l,i,ii] = griddata((KpeN[j,l,i,ii],kp,KpuN[j,l,i,ii]),kpmat[:,:,:,2,i],(Kemat,kmat,Kumat))
                    ################################################## doesn't work


    # Update the convergence measure:
    ConvCrit = np.amax(np.absolute(kpp2New[:,0,0,0,0]-kpp2[:,0,0,0,0])/(1+np.absolute(kpp2[:,0,0,0,0))))

    kpp2 = kpp2New

    # Update the aggregate policy function using explicit aggregation:
    for i in range(0,NK):
        for j in range(0,NK):
            for l in range(0,2):
                KpeNew(i,j,l) = np.interp(Ke[i],kp,kpmat(:,i,j,1,l))+de
                KpuNew(i,j,l) = interp1(kp,kpmat(:,i,j,2,l),Ku(j))+du


    if s>200:   
        rho = 0
        Kpe = rho*Kpe+(1-rho)*KpeNew
        Kpu = rho*Kpu+(1-rho)*KpuNew

    KpeN[:,:,0,0] = (P11[0,0]*(1-UnempG)*Kpe[:,:,0]+P11[1,0]*UnempG*Kpu[:,:,0])/(1-UnempG)
    KpeN[:,:,0,1] = (P01[0,0]*(1-UnempB)*Kpe[:,:,1]+P01[1,0]*UnempB*Kpu[:,:,1])/(1-UnempG)
    KpeN[:,:,1,0] = (P10[0,0]*(1-UnempG)*Kpe[:,:,0]+P10[1,0]*UnempG*Kpu[:,:,0])/(1-UnempB)
    KpeN[:,:,1,1] = (P00[0,0]*(1-UnempB)*Kpe[:,:,1]+P00[1,0]*UnempB*Kpu[:,:,1])/(1-UnempB)

    KpuN[:,:,0,0] = (P11[0,1]*(1-UnempG)*Kpe[:,:,0]+P11[1,1]*UnempG*Kpu[:,:,0])/(UnempG)
    KpuN[:,:,0,1] = (P01[0,1]*(1-UnempB)*Kpe[:,:,1]+P01[1,1]*UnempB*Kpu[:,:,1])/(UnempG)
    KpuN[:,:,1,0] = (P10[0,1]*(1-UnempG)*Kpe[:,:,0]+P10[1,1]*UnempG*Kpu[:,:,0])/(UnempB)
    KpuN[:,:,1,1] = (P00[0,1]*(1-UnempB)*Kpe[:,:,1]+P00[1,1]*UnempB*Kpu[:,:,1])/(UnempB)
    
    KpeN = np.minimum(KpeN,np.amax(Ke))
    KpeN = np.maximum(KpeN,np.amin(Ke))
    KpuN = np.minimum(KpuN,np.amax(Ku))
    KpuN = np.maximum(KpuN,np.amin(Ku))
    


# And repeat this procedure until convergence.

print 'The individual and aggregate problem has converged. Simulation will proceed until s=10000'

# Initial Distribution

with open('pdistyoung.txt') as f:
    temp = []
    for line in f:
        line = line.split() 
        if line:            
            line = [float(i) for i in line]
            temp.append(line)


InDist= np.asarray(temp)
InDist = InDist[:,1:3]

NDist = InDist.shape[0]

kk = np.transpose(np.linspace(0,kmax,NDist))

Pe = InDist[:,1]
Pu = InDist[:,0]

# Exogenous shocks


with open('Z.txt') as f:
    temp = []
    for line in f:
        line = line.split() 
        if line:            
            line = [int(i) for i in line]
            temp.append(line)
            
ZSim = np.asarray(temp)*-1+3

KeSim = np.zeros(shape=(ZSim.shape[0]))
KuSim = KeSim

KeSim[0] = np.dot(np.transpose(kk),Pe)
KuSim[0] = np.dot(np.transpose(kk),Pu)

KeImp = KeSim
KuImp = KuSim

KeFit = KeSim
KuFit = KuSim

KuMat2,KeMat2 = ndmesh(Ku,Ke)



with open('ind_switch.txt') as f:
    temp = []
    for line in f:
        line = line.split() 
        if line:            
            line = [int(i) for i in line]
            temp.append(line)
            
ind_switch = np.asarray(temp)*-1+3
Kind = np.zeros(shape=(ZSim.shape[0])) 
Cind = np.zeros(shape=(ZSim.shape[0]-1,1))
Kind[0] = 43
percentile = np.zeros(shape=(ZSim.shape[0],6))
moments = np.zeros(shape=(ZSim.shape[0],5))
momentse = np.zeros(shape=(ZSim.shape[0],5))
momentsu = np.zeros(shape=(ZSim.shape[0],5))
Rvec = np.zeros(shape=(ZSim.shape[0],1))
Wvec = np.zeros(shape=(ZSim.shape[0],1))

# Let's rock'n'roll

s = 0
SimLength = 10000-1
for i in range(0,SimLength):
    s = s+1
    #interp2(X,Y,Z,XI,YI) returns matrix containing elements corresponding to the elements of XI and YI
    #determined by interpolation within 2d function specified by X,Y,Z
    #interp2d(x,y,z) x,y,x arrays of values used to approximate z=f(x,y)
    #KpeSim = interp2(KuMat2,KeMat2,Kpe(:,:,ZSim(i)),KuSim(i),KeSim(i))
    KpeSim = griddata(np.hstack([[KuMat2],[KeMat2]]) ,Kpe[:,:,int(ZSim[i])-1],)
    KpeSim = griddata((KuMat2,KeMat2),Kpe[:,:,int(ZSim[i])-1],np.hstack([[KuSim[0]],[KeSim[0]]]))
    KpeSim = griddata(KuMat2,Kpe[:,:,int(ZSim[i])-1],KeMat2)
    KpuSim = interp2(KuMat2,KeMat2,Kpu(:,:,ZSim(i)),KuSim(i),KeSim(i))
    
    KpeFit = interp2(KuMat2,KeMat2,Kpe(:,:,ZSim(i)),KuImp(i),KeImp(i))
    KpuFit = interp2(KuMat2,KeMat2,Kpu(:,:,ZSim(i)),KuImp(i),KeImp(i))
    
    kprimeE = interp3(Kemat,kmat,Kumat,kpmat(:,:,:,1,ZSim(i)),np.dot(np.transpose(kk),Pe),kk,np.dot(np.transpose(kk),Pu))
    kprimeU = interp3(Kemat,kmat,Kumat,kpmat(:,:,:,2,ZSim(i)),np.dot(np.transpose(kk),Pe),kk,np.dot(np.transpose(kk),Pu))

    Kind(i+1) = interp3(Kemat,kmat,Kumat,kpmat(:,:,:,ind_switch(i),ZSim(i)),np.dot(np.transpose(kk),Pe),Kind(i),np.dot(np.transpose(kk)*Pu))
    K = (1-Unemp[int(ZSim[i])])*KeImp[i]+Unemp[int(ZSim[i])]*KuImp[i]
    r = 1+alpha*ZZ[int(ZSim[i])]*(K/(h*(1-Unemp[int(ZSim[i])])))**(alpha-1)-delta
    w = (1-alpha)*ZZ[int(ZSim[i])]*(K/(h*(1-Unemp[int(ZSim[i]]))))**(alpha)
    Cind(i,1) = r*Kind[i]+(2-ind_switch[i])*h*w*(1-tau[int(ZSim[i])])+(ind_switch[i]-1)*w*UI-Kind[i+1]
    Rvec[i] = r
    Wvec[i] = w

    kprimeE = np.minimum(kprimeE,np.amax(kk))
    kprimeE = np.maximum(kprimeE,np.amin(kk))

    kprimeU = np.minimum(kprimeU,np.amax(kk))
    kprimeU = np.maximum(kprimeU,np.amin(kk))

    P = (1-Unemp[int(ZSim[i])])*Pe+Unemp[int(ZSim[i])]*Pu
    P=[0.01,0.02,0.02,0.05,0.06]
    CP = np.cumsum(P)
    IP5 = np.where(CP<0.05)[0]
    IP5 = IP5[-1:]
    percentile[i,0]=(0.05-CP[IP5+1])/(CP[IP5]-CP[IP5+1])*kk[IP5]+(1-(0.05-CP[IP5+1])/(CP[IP5]-C[IP5+1]))*kk[IP5+1]
    IP10 = np.where(CP<0.10)[0]
    IP10 = IP10[-1:]
    percentile[i,1]=(0.1-CP[IP10+1])/(CP[IP10]-CP[IP10+1])*kk[IP10]+(1-(0.1-CP[IP10+1])/(CP[IP10]-CP[IP10+1]))*kk[IP10+1]

    CPe = np.cumsum(Pe)
    IPe5 = np.where(CPe<0.05)[0]
    IPe5 = IPe5[-1:]
    percentile[i,2]=(0.05-CPe[IPe5+1])/(CPe[IPe5]-CPe[IPe5+1])*kk[IPe5]+(1-(0.05-CPe([IPe5+1]))/(CPe[IPe5]-CPe[IPe5+1]))*kk[IPe5+1]
    IPe10 = np.where(CPe<0.10)[0]
    IPe10 = IPe10[-1:]
    percentile[i,3]=(0.1-CPe[IPe10+1])/(CPe[IPe10]-CPe[IPe10+1])*kk[IPe10]+(1-(0.1-CPe[IPe10+1])/(CPe[IPe10]-CPe[IPe10+1]))*kk[IPe10+1]

    CPu = np.cumsum(Pu)
    IPu5 = np.where(CPu<0.05)[0]
    IPu5 = IPu5[-1:]
    percentile[i,4]=(0.05-CPu[IPu5+1])/(CPu[IPu5]-CPu[IPu5+1])*kk[IPu5]+(1-(0.05-CPu[IPu5+1])/(CPu[IPu5]-CPu[IPu5+1]))*kk[IPu5+1]
    IPu10 = np.where(CPu<0.10)[0]
    IPu10 = IPu10[-1:]
    percentile[i,5]=(0.1-CPu[IPu10+1])/(CPu[IPu10]-CPu[IPu10+1])*kk[IPu10]+(1-(0.1-CPu[IPu10+1])/(CPu[IPu10]-CPu[IPu10+1]))*kk[IPu10+1]

    moments[i,0] = n.dot(np.transpose(kk),P)
    momentse[i,0] = np.dot(np.transpose(kk),Pe)
    momentsu[i,0] = np.dot(np.transpose(kk),Pu)

    for j in range (1,5):
        moments[i,j] = (np.dot(np.transpose(kk**j),P))**(1/j)/moments[i,0]
        momentse[i,j] = (np.dot(np.transpose(kk**j),Pe))**(1/j)/momentse[i,0]
        momentsu[i,j] = (np.dot(np.transpose(kk**j),Pu))**(1/j)/momentsu[i,0]


    for j in range(0,NDist):
        Ie[j,0] = np.where(kprimeE(j)>=kk)[0]
        Ie[j,0] = Ie[j,0][-1:]
        Iu[j,0] = np.where(kprimeE(j)>=kk)[0]
        Iu[j,0] = Iu[j,0][-1:]

    Ie = np.minimum(Ie,NDist-1)
    Iu = np.minimum(Iu,NDist-1)

    rhoE = (kprimeE-kk[Ie+1])/(kk[Ie]-kk[Ie+1])

    rhoU = (kprimeU-kk[Iu+1])/(kk[Iu]-kk[Iu+1])

    Le1 = np.zeros(shape=(NDist,1))
    Le2 = Le1
    Lu1 = Le1
    Lu2 = Le1

    for jj in range(0,NDist):
        Le1[Ie[jj]] = rhoE[jj]*Pe[jj]+Le1[Ie[jj]]
        Le2[Ie[jj]+1] = (1-rhoE[jj])*Pe[jj]+Le2[Ie[jj]+1]
        Lu1[Iu[jj]] = rhoU[jj]*Pu[jj]+Lu1[Iu[jj]]
        Lu2[Iu[jj]+1] = (1-rhoU[jj])*Pu[jj]+Lu2[Iu[jj]+1]
 
    PPe = Le1+Le2
    PPu = Lu1+Lu2

    if ZSim[i]==1:
        if ZSim[i+1]==1:
            KeSim[i+1] = (P11[0,0]*(1-UnempG)*KpeSim+P11[1,0]*UnempG*KpuSim)/(1-UnempG)
            KuSim[i+1] = (P11[0,1]*(1-UnempG)*KpeSim+P11[1,1]*UnempG*KpuSim)/(UnempG)
            KeFit[i+1] = (P11[0,0]*(1-UnempG)*KpeFit+P11[1,0]*UnempG*KpuFit)/(1-UnempG)
            KuFit[i+1] = (P11(1,2)*(1-UnempG)*KpeFit+P11[1,1]*UnempG*KpuFit)/(UnempG)
            Pe = (P11[0,0]*(1-UnempG)*PPe+P11[1,0]*UnempG*PPu)/(1-UnempG)
            Pu = (P11[0,1]*(1-UnempG)*PPe+P11[1,1]*UnempG*PPu)/(UnempG)
            KeImp[i+1] = np.dot(np.transpose(kk),Pe)
            KuImp(i+1) = np.dot(np.transpose(kk),Pu)

        if ZSim[i+1]==2:
            KeSim[i+1] = (P10[0,0]*(1-UnempG)*KpeSim+P10[1,0]*UnempG*KpuSim)/(1-UnempB)
            KuSim[i+1] = (P10[0,1]*(1-UnempG)*KpeSim+P10[1,1]*UnempG*KpuSim)/(UnempB)
            KeFit[i+1] = (P10[0,0]*(1-UnempG)*KpeFit+P10[1,0]*UnempG*KpuFit)/(1-UnempB)
            KuFit[i+1] = (P10[0,1]*(1-UnempG)*KpeFit+P10[1,1]*UnempG*KpuFit)/(UnempB)
            Pe = (P10[0,0]*(1-UnempG)*PPe+P10[1,0]*UnempG*PPu)/(1-UnempB)
            Pu = (P10[0,1]*(1-UnempG)*PPe+P10[1,1]*UnempG*PPu)/(UnempB)
            KeImp[i+1] = n.dot(np.transpose(kk),Pe)
            KuImp[i+1] = n.dot(np.transpose(kk),Pu)

    if ZSim[i]==2:
        if ZSim[i+1]==1
            KeSim[i+1] = (P01[0,0]*(1-UnempB)*KpeSim+P01[1,0]*UnempB*KpuSim)/(1-UnempG)
            KuSim[i+1] = (P01[0,1]*(1-UnempB)*KpeSim+P01[1,1]*UnempB*KpuSim)/(UnempG)
            KeFit[i+1] = (P01[0,0]*(1-UnempB)*KpeFit+P01[1,0]*UnempB*KpuFit)/(1-UnempG)
            KuFit[i+1] = (P01[0,1]*(1-UnempB)*KpeFit+P01[1,1]*UnempB*KpuFit)/(UnempG)
            Pe = (P01[0,0]*(1-UnempB)*PPe+P01[1,0]*UnempB*PPu)/(1-UnempG)
            Pu = (P01[0,1]*(1-UnempB)*PPe+P01[1,1]*UnempB*PPu)/(UnempG)
            KeImp[i+1] =  n.dot(np.transpose(kk),Pe)
            KuImp(i+1) =  n.dot(np.transpose(kk),Pu)
    
        if ZSim[i+1]==2
            KeSim[i+1] = (P00[0,0]*(1-UnempB)*KpeSim+P00[1,0]*UnempB*KpuSim)/(1-UnempB)
            KuSim[i+1] = (P00[0,1]*(1-UnempB)*KpeSim+P00[1,1]*UnempB*KpuSim)/(UnempB)
            KeFit[i+1] = (P00[0,0]*(1-UnempB)*KpeFit+P00[1,0]*UnempB*KpuFit)/(1-UnempB)
            KuFit[i+1] = (P00[0,1]*(1-UnempB)*KpeFit+P00[1,1]*UnempB*KpuFit)/(UnempB)
            Pe = (P00[0,0]*(1-UnempB)*PPe+P00[1,0]*UnempB*PPu)/(1-UnempB)
            Pu = (P00[0,1]*(1-UnempB)*PPe+P00[1,1]*UnempB*PPu)/(UnempB)
            KeImp[i+1] = n.dot(np.transpose(kk),Pe)
            KuImp[i+1] = n.dot(np.transpose(kk),Pu)


KSim = (1-Unemp[ZSim])*KeSim+Unemp[ZSim]*KuSim
KImp = (1-Unemp[ZSim])*KeImp+Unemp[ZSim]*KuImp
KFit = (1-Unemp[ZSim])*KeFit+Unemp[ZSim]*KuFit

ZZ = [zg;zb] #change
Y = ZZ[ZSim]*(KImp**alpha)*(((1-Unemp[ZSim])*h)**(1-alpha))
C = Y[0:end-1]-KImp[1:end)+(1-delta)*KImp[0:end-1]

Yind = Cind+Kind[1:end]-(1-delta)*Kind[1:end-1]

print 'Correlation of individual and aggregate consumption:',corrcoef([Cind,C])
print 'Correlation of individual consumption and aggregate income:',corrcoef([Cind,Y(1:end-1)])
print 'Correlation of individual consumption and aggregate capital:', corrcoef([Cind,KImp(1:end-1)])
print 'Correlation of individual consumption and individual income:',corrcoef([Cind,Yind])
print 'Correlation of individual consumption and individual capital:',corrcoef([Cind,Kind(1:end-1)])
print 'Standard deviation of individual consumption:', std(Cind)
print 'Standard deviation of individual capital:', std(Kind)
print 'Autocorrelation of individual consumption:',corrcoef([Cind(1:end-3),Cind(2:end-2),Cind(3:end-1),Cind(4:end)])
print 'Autocorrelation of individual capital:',corrcoef([Kind(1:end-3),Kind(2:end-2),Kind(3:end-1),Kind(4:end)])
print 'Autocorrelation of individual consumption growth:', cgrowth = log(Cind(2:end))-log(Cind(1:end-1)), corrcoef([cgrowth(1:end-3),cgrowth(2:end-2),cgrowth(3:end-1),cgrowth(4:end)])
print 'Max error Ke (%)',100*np.maximum(np.absolute(np.log(KeSim)-np.log(KeImp)))
print 'Max error Ku (%)', 100*np.maximum(np.absolute(np.log(KuSim)-np.log(KuImp)))
print 'Mean error Ke (%)', 100*np.mean(np.absolute(np.log(KeSim)-np.log(KeImp)))
print 'Mean error Ku (%)', 100*np.mean(np.absolute(np.log(KuSim)-np.log(KuImp)))
print 'R-Square K', 1-vnp.var(KImp-KFit)/np.var(KImp)
print 'R-Square Ke',1-np.var(KeImp-KeFit)/np.var(KeImp)
print 'R-Square Ku', 1-np.var(KuImp-KuFit)/np.var(KuImp)


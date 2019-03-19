# Run the Simulation
#
from __future__ import division
import os,sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir)

from pseudospectral import *
from scipy import *
from numpy import *
from numpy import *
from MatGrad import *

#from mayavi import mlab
from numexpr import *
from multiprocessing import Pool
from multiprocessing import cpu_count
from scipy.ndimage.filters import *
from FourierInterpol import *


filename = 'NESS1DFINAL6_'

Nx = 601 #not divisible by 3
xmax = 10000
xmin = -10000



x,Dx,D2x = ScaledFourierSpectral(xmin,xmax,Nx)
kwave = pi*2.0/(x.max()-x.min())

xcoarse = ScaledFourierSpectral(xmin,xmax,2*Nx/3)[0]
FiltxF2C = fourierInterpFToC(x,xcoarse)
FiltxC2F= fourierInterpFToC(xcoarse,x)


e = ones((Nx))
j = ones((Nx))

dt = .05

def Filterdata(inpx,inpfun, Filtx,  ax): 
    fun = swapaxes(inpfun, ax , 0)
    filtfun = tensordot(Filtx,fun, ([-1,0]))
    return swapaxes(filtfun, ax , 0)


def init(e,j,x):
    el=1.
    er=.5    
    j = 0*j
    e= .5*(el+er+(er-el)*tanh(25.*sin(kwave*x)))
    return e,j




def source(e,j):
    dxe=MatGrad(x,e,Dx,ax=0)
    dxj=MatGrad(x,j,Dx,ax=0)
    d2xe=MatGrad(x,e,D2x,ax=0)
    d2xj=MatGrad(x,j,D2x,ax=0)
    sourcee = evaluate("-dxj+d2xe")
    sourcej = evaluate("-(dxe-j*j*dxe/e/e+2*j*dxj/e)+d2xj")
    return sourcee,sourcej
    




tcoord = 0.0

BigSteps =250
i = 0
Smallsteps = 100

e,j = init(e,j,x)
e0,j0= init(e,j,x)

#fft = u10[Nx/2]
el = e0
for steps in range(BigSteps):
    i=i+1
    time1=time()
    #print u1.max()
    #print TT.max()
    #print BB.max()
    for st in range(Smallsteps):
        
        k1e,k1j = source(e,j)
        #print k1u.max()
        #print k1T.max()
        #print k1B.max()
        k2e,k2j = source(e+dt/2*k1e,j+dt/2*k1j)
        k3e,k3j = source(e+dt/2*k2e,j+dt/2*k2j)
        k4e,k4j = source(e+dt*k3e,j+dt*k3j)
        e = e+dt/6*(k1e+2*k2e+2*k3e+k4e)
        j = j+dt/6*(k1j+2*k2j+2*k3j+k4j)
    #fft = append(fft,u1[Nx/2])
    el=append(el,e,1)
    
    with file(filename+'e'+'.txt','a') as plotfile:
        plotfile.write('Physical Time: {0}\n'.format(tcoord))
        savetxt(plotfile, e, fmt='%-7.11f')
    with file(filename+'j'+'.txt','a') as plotfile:
        plotfile.write('Physical Time: {0}\n'.format(tcoord))
        savetxt(plotfile, j, fmt='%-7.11f')
    
   
    time2 =time()





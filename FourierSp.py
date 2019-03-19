# FourierSp.py
#
# Compute the Fourier differentiation matrix
#
# Even Number of points is used for the Fourier Grid

import scipy as sp
from operator import mul
from numpy import *
from pylab import *
import scipy.linalg as lg
from numexpr import *

def fourierdiffmat(x):
    n = sp.size(x)
    n2 = int(ceil((n-1)/2.0))
    n1 = int(floor((n-1)/2.0))
    h = pi/n
    jlist = arange(n)
    sg = evaluate("(-1)**jlist")
    dSn = zeros(n)
    if(mod(n,2)==0):
        Elen = 0.5/tan(jlist[1:n2]*h)
        dSn[1:n2] = Elen*array(sg[1:n2])
        dSn[n2+1:n]=flipud(-Elen[0:n1])*array(sg[n2+1:n])
    else:
        #dSn[1:n] = 0.5/sin(jlist[1:n]*h)*sg[1:n]
        Elen = 0.5/sin(jlist[1:n2+1]*h)
        dSn[1:n2+1] = Elen*array(sg[1:n2+1])
        dSn[n2+1:n]=flipud(Elen[0:n1])*array(sg[n2+1:n])
    D = lg.circulant(dSn)
    return D

def fourierdiffmat2(x):
    n = sp.size(x)
    h = 2.0*pi/n
    jlist = arange(n)
    sg = pow(-1,jlist)
    dSn2 = zeros(n)
    if(mod(n,2)==0):
        dSn2[1:] = -sg[1:]*0.5/pow(sin(0.5*jlist[1:]*h),2.0)
        dSn2[0] = -pi*pi/(3.0 * h*h) -1/6.0
        D2 = lg.circulant(dSn2)
    else:
        Dtemp = fourierdiffmat(x)
        D2 = dot(Dtemp,Dtemp)

    return D2

def fouriergrid(n): # Chebyshev gridpoints
    x= -pi + 2.0*pi*arange(n)/(n)
    return x

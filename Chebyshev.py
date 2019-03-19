# Chebyshev.py
#
# Compute the Chebyshev differentiation matrix
#
# ChebyshevOld uses a slightly different formula that seems to be sensitive to
# Rounding-off errors for large N.


import scipy as sp
from operator import mul
from numpy import *
from pylab import *
import scipy.linalg as lg


def diffmat(x): # x denotes points on Chebyshev grid
    n = sp.size(x)
    ee = sp.ones((n,1))
    Xdiff = array(sp.outer(x,ee)-sp.outer(ee,x)+sp.identity(n))
    c = zeros(n)
    c[0]=2.0
    c[1:n-1]=1.0
    c[n-1]=2.0
    c = c*pow(-1,arange(n))
    invcmat = mat(1/c)
    cmat=mat(c)
    cibycj = array(dot(cmat.T,invcmat))
    D = cibycj/Xdiff
    dsum = sum(D, axis=1)
    D=D-diag(dsum)
    return D

def diffmat2(x): # x denotes points on Chebyshev grid
    D = diffmat(x)
    D2 = dot(D,D)
    return D2

def chebspectral(n):
    x = chebgrid(n)

    D = diffmat(x)
    D2 = dot(D,D)
    return x,D, D2

def chebgrid(n): # Chebyshev gridpoints
    x=cos(pi*arange(0,n+1)/(n))
    return x

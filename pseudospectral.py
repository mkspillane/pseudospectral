# pseudospectral.py
#
# Compute the pseudospectral differentiation matrix for Cheb and Fourier Grids
#
# Even Number of points is used for the Fourier Grid
#
from scipy import *
from operator import mul
from numpy import *
from pylab import *
import scipy.linalg as lg
from Chebyshev import *
from FourierSp import *


#    Chebyshev Differentiation Matrix  (this is used for fixed boundary conditions)

def ScaledChebSpectral(endpoint1,endpoint2, n):
    x,D,D2 = chebspectral(n)
    mslope = (endpoint2 - endpoint1)/(x[n]-x[0])
    x = endpoint1 + mslope*(x-x[0])
    D = D/mslope
    D2 = D2/(mslope*mslope)
    return x,D,D2



#    Fourier Differentiation Matrix (this is used for periodic boundary condtions)

def ScaledFourierSpectral(endpoint1, endpoint2, n):

    tempx = fouriergrid(n)
    tempD = fourierdiffmat(tempx)
    tempD2 = fourierdiffmat2(tempx)

    mslope = (endpoint2 - endpoint1)/(2.0*pi)
    x = endpoint1 + mslope*(tempx-tempx[0])
    D = tempD/mslope
    D2 = tempD2/(mslope*mslope)
    return x,D, D2

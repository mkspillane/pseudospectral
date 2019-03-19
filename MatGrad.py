# MatGrad.py
#
#
#
import os,sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir)
from pseudospectral import *
from numpy import *
from Chebyshev import *
from FourierSp import *

def MatGrad(inpx,inpfun, Dinpx, ax): 
    fun = swapaxes(inpfun, ax , 0)
    dfun = tensordot(Dinpx,fun, ([-1,0]))
    return swapaxes(dfun, ax , 0)

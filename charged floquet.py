# Run the Simulation


from pseudospectral import *
from scipy import *
from numpy import *
from MatGrad import *




filename = 'Reimann_B10'

Nx = 401
Ny = 51

xmax = 40
xmin = -40

ymax = 5
ymin = -5

x,Dx,D2x = ScaledFourierSpectral(xmin,xmax,Nx)
kxwave = pi*2.0/(xmax-xmin)

y,Dy,D2y = ScaledFourierSpectral(ymin,ymax,Ny)
kywave = pi*2.0/(ymax-ymin)


xcoarse = ScaledFourierSpectral(xmin,xmax,2*Nx/3)[0]
ycoarse = ScaledFourierSpectral(ymin,ymax,2*Ny/3)[0]
# FiltxF2C = fourierInterpFToC(x,xcoarse)
# FiltyF2C = fourierInterpFToC(y,ycoarse)
# FiltxC2F= fourierInterpFToC(xcoarse,x)
# FiltyC2F= fourierInterpFToC(ycoarse,y)



 
u1 = ones((Nx,Ny))
u2 = ones((Nx,Ny))
p23 = ones((Nx,Ny))
T= ones((Nx,Ny))
q = ones((Nx,Ny))
p33 = ones((Nx,Ny))

sQ=0.

EE=0.0
B=.1
omega = 1
h0=.5

dt=.001


k=1/36*(sqrt(3)*pi-9*log(3)+18)
k2=1/36*(sqrt(3)*pi-9*log(3))


def init(q,u1,u2,p23,p33,T,x,y):
    e0=1
    q0=1
    v0=0
    epsilon = 0.2
    
    dt = e0*ones((Nx*Ny))

    du1 = v0*ones((Nx*Ny))

    du2 = 0*ones((Nx*Ny))
    dp23 = 0*ones((Nx*Ny))
    dp33=0*ones((Nx*Ny))
    dq=q0*ones((Nx*Ny))
    
    dj1 = zeros((Nx*Ny))
    
    X=kron(x,cos(0*kywave*y))
    Y=kron(cos(0*kxwave*x),y)
    
    X=X.reshape(Nx,Ny)
    Y=Y.reshape(Nx,Ny)
    
    T = dt.reshape(Nx,Ny)
    u1 = du1.reshape(Nx,Ny)
    u2 = evaluate("tanh(10*sin(kxwave*X+ .05*sin(kywave*Y)))")
    q = dq.reshape(Nx,Ny)
    
    
    dyu1=MatGrad(y,u1,Dy,ax=1)
    dxu1=MatGrad(x,u1,Dx,ax=0)  
    
    dyu2=MatGrad(y,u2,Dy,ax=1)
    dxu2=MatGrad(x,u2,Dx,ax=0)
       
    
    
    p23 = evaluate("dxu2*(1 + u1**2) + dyu1*(1 + u2**2)")
    p33 = evaluate("2*(dxu2*u1*u2 - (dxu1*(1 + u2**2))/2. + (dyu2*(1 + u2**2))/2.)")
    return q,u1,u2,p23,p33,T


def electric(t):
    
    Ex = evaluate("EE*cos(omega*t)")
    Ey = evaluate("EE*sin(omega*t)")
    
    dEx = evaluate("-omega*EE*sin(omega*t)")
    dEy = evaluate("omega*EE*cos(omega*t)")
    
    return Ex,Ey,dEx,dEy
    

def source(q,u1,u2,p23,p33,T,t,x,y):
    
    
    h = evaluate("h0*T**2")
    tp = evaluate("k*T/h")
    l2= evaluate("(k2*T)")
    
    Ex,Ey,dEx,dEy = electric(t)

    dyT=MatGrad(y,T,Dy,ax=1)
    dy2T=MatGrad(y,T,D2y,ax=1)
    dxT=MatGrad(x,T,Dx,ax=0)
    dx2T=MatGrad(x,T,D2x,ax=0)

    dyq=MatGrad(y,q,Dy,ax=1)
    dy2q=MatGrad(y,q,D2y,ax=1)
    dxq=MatGrad(x,q,Dx,ax=0)
    dx2q=MatGrad(x,q,D2x,ax=0) 
    
    dyu1=MatGrad(y,u1,Dy,ax=1)
    dy2u1=MatGrad(y,u1,D2y,ax=1)
    dxu1=MatGrad(x,u1,Dx,ax=0)
    dx2u1=MatGrad(x,u1,D2x,ax=0)    
    
    dyu2=MatGrad(y,u2,Dy,ax=1)
    dy2u2=MatGrad(y,u2,D2y,ax=1)
    dxu2=MatGrad(x,u2,Dx,ax=0)
    dx2u2=MatGrad(x,u2,D2x,ax=0) 
    
    dyp33=MatGrad(y,p33,Dy,ax=1)
    dy2p33=MatGrad(y,p33,D2y,ax=1)
    dxp33=MatGrad(x,p33,Dx,ax=0)
    dx2p33=MatGrad(x,p33,D2x,ax=0) 
    
    dyp23=MatGrad(y,p23,Dy,ax=1)
    dy2p23=MatGrad(y,p23,D2y,ax=1)
    dxp23=MatGrad(x,p23,Dx,ax=0)
    dx2p23=MatGrad(x,p23,D2x,ax=0) 

    


    a11= evaluate("-3*T**3*u1 + (2*p33*u1 - 2*p23*u2)/(1 + u2**2)")
    a12= evaluate("(2*p23*u1*(-1 + u2**2) - u2*(2*p33*(1 + u1**2) + 3*T**3*(1 + u2**2)**2))/(1 + u2**2)**2")
    a13= evaluate("(-3*T**2*(2 + 3*u1**2 + 3*u2**2))/2.")
    a14= evaluate("0*u1")
    a15= evaluate("(-2*u1*u2)/(1 + u2**2)")
    a16= evaluate("(u1**2 - u2**2)/(1 + u2**2)")
    
    a21= evaluate("(3*T**3*(1 + u2**2)*(1 + u1**2 + u2**2)*(1 + 2*u1**2 + u2**2) + 2*p23*u1*u2*(3 + 2*u1**2 + 3*u2**2) - 2*p33*(1 + 2*u1**4 + u2**2 + 3*u1**2*(1 + u2**2)))/(2.*(1 + u2**2)*(1 + u1**2 + u2**2)**1.5)")
    a22= evaluate("(u1*u2*(3*T**3*(1 + u2**2)**2*(1 + u1**2 + u2**2) + 2*p33*(1 + u1**2)*(3 + 2*u1**2 + 3*u2**2)) + 2*p23*(-2*u1**4*(-1 + u2**2) + (1 + u2**2)**2 - 3*u1**2*(-1 + u2**4)))/(2.*(1 + u2**2)**2*(1 + u1**2 + u2**2)**1.5)")
    a23= evaluate("(9*T**2*u1*sqrt(1 + u1**2 + u2**2))/2.")
    a24= evaluate("0*u1")
    a25= evaluate("(u2*(1 + 2*u1**2 + u2**2))/((1 + u2**2)*sqrt(1 + u1**2 + u2**2))")
    a26= evaluate("-((u1*(1 + u1**2))/((1 + u2**2)*sqrt(1 + u1**2 + u2**2)))")
    
    a31= evaluate("(2*p23*(1 + u2**2) + u1*u2*(-2*p33 + 3*T**3*(1 + u1**2 + u2**2)))/(2.*(1 + u1**2 + u2**2)**1.5)")
    a32= evaluate("(2*p33*(1 + u1**2) - 2*p23*u1*u2 + 3*T**3*(1 + u1**2 + u2**2)*(1 + u1**2 + 2*u2**2))/(2.*(1 + u1**2 + u2**2)**1.5)")
    a33= evaluate("(9*T**2*u2*sqrt(1 + u1**2 + u2**2))/2.")
    a34= evaluate("0*u1")
    a35= evaluate("u1/sqrt(1 + u1**2 + u2**2)")
    a36= evaluate("u2/sqrt(1 + u1**2 + u2**2)")
    
    a41= evaluate("Ex*sQ + (q*u1)/sqrt(1 + u1**2 + u2**2)")
    a42= evaluate("Ey*sQ + (q*u2)/sqrt(1 + u1**2 + u2**2)")
    a43= evaluate("0*u1")
    a44= evaluate("sqrt(1 + u1**2 + u2**2)")
    a45= evaluate("0*u1")
    a46= evaluate("0*u1")
    
    a51= evaluate("(2*h*u2*(h + p33*tp + 2*p33*tp*u1**2 + h*u2**2) + p23*u1*(l2 + h*tp*(1 - 4*u2**2)))/(2.*h*sqrt(1 + u1**2 + u2**2))")
    a52= evaluate("(2*h*(h - p33*tp)*(u1 + u1**3) + p23*(l2 + h*tp)*u2 + 2*h*(h - 2*p33*tp)*(u1 + u1**3)*u2**2 + p23*(l2 + h*tp*(1 + 4*u1**2))*u2**3)/(2.*h*(1 + u2**2)*sqrt(1 + u1**2 + u2**2))")
    a53= evaluate("0*u1")
    a54= evaluate("0*u1")
    a55= evaluate("tp*sqrt(1 + u1**2 + u2**2)")
    a56= evaluate("0*u1")
    
    a61= evaluate("(l2*p33*u1 - 2*h**2*u1*(1 + u2**2) + h*tp*(-4*p23*u2*(1 + u2**2) + p33*u1*(3 + 4*u2**2)))/(2.*h*sqrt(1 + u1**2 + u2**2))")
    a62= evaluate("(u2*(l2*p33 - h*tp*(p33 + 4*p33*u1**2 - 4*p23*u1*u2) + 2*h**2*(1 + 2*u1**2 + u2**2)))/(2.*h*sqrt(1 + u1**2 + u2**2))")
    a63= evaluate("0*u1")
    a64= evaluate("0*u1")
    a65= evaluate("0*u1")
    a66= evaluate("tp*sqrt(1 + u1**2 + u2**2)")
    
    
    b1=evaluate("-(((dyu1*u1 + dyu2*u2)*(p23*u1 + p33*u2))/(1 + u1**2 + u2**2)**1.5) + (dyu1*p23 + dyu2*p33 + dyp23*u1 + dyp33*u2)/sqrt(1 + u1**2 + u2**2) - Ex*(q*u1 - B*sQ*u2 + Ex*sQ*sqrt(1 + u1**2 + u2**2)) - Ey*(B*sQ*u1 + q*u2 + Ey*sQ*sqrt(1 + u1**2 + u2**2)) + ((dxu1*u1 + dxu2*u2)*(p33*(u1 + u1**3) - p23*u2*(1 + 2*u1**2 + u2**2)))/((1 + u2**2)*(1 + u1**2 + u2**2)**1.5) - (2*dxu2*u2*(-(p33*(u1 + u1**3)) + p23*u2*(1 + 2*u1**2 + u2**2)))/((1 + u2**2)**2*sqrt(1 + u1**2 + u2**2)) + (-(dxp33*u1) - dxp33*u1**3 + dxp23*u2 + 2*dxp23*u1**2*u2 + dxp23*u2**3 - dxu1*(p33 + 3*p33*u1**2 - 4*p23*u1*u2) + dxu2*p23*(1 + 2*u1**2 + 3*u2**2))/((1 + u2**2)*sqrt(1 + u1**2 + u2**2)) + (3*T**2*(dxu1*T*(1 + 2*u1**2 + u2**2) + u1*(dxu2*T*u2 + 3*dxT*(1 + u1**2 + u2**2))))/(2.*sqrt(1 + u1**2 + u2**2)) + (3*T**2*(dyu2*T*(1 + u1**2 + 2*u2**2) + u2*(dyu1*T*u1 + 3*dyT*(1 + u1**2 + u2**2))))/(2.*sqrt(1 + u1**2 + u2**2))")
    b2=evaluate("Ex*(Ex*sQ*u1 + Ey*sQ*u2 + q*sqrt(1 + u1**2 + u2**2)) - B*(B*sQ*u1 + q*u2 + Ey*sQ*sqrt(1 + u1**2 + u2**2)) - (2*dyp23*(1 + u2**2)**2 + 3*T**3*(2*dxu1*u1 + dyu2*u1 + dyu1*u2)*(1 + u2**2)**2 + 3*T**2*(dxT + 3*dxT*u1**2 + 3*dyT*u1*u2)*(1 + u2**2)**2 - 2*(dxp33*(1 + u1**2)*(1 + u2**2) - 2*((-(dxu1*p33*u1) + dxu1*p23*u2 + dxp23*u1*u2)*(1 + u2**2) + dxu2*(p33*(1 + u1**2)*u2 + p23*(u1 - u1*u2**2)))))/(2.*(1 + u2**2)**2)")
    b3=evaluate("(-2*dxp23 - 2*dyp33 - 3*dyT*T**2 - 3*dxu2*T**3*u1 - 2*B**2*sQ*u2 + 2*Ey**2*sQ*u2 - 3*dxu1*T**3*u2 - 6*dyu2*T**3*u2 - 9*dxT*T**2*u1*u2 - 9*dyT*T**2*u2**2 + 2*Ex*sQ*(Ey*u1 + B*sqrt(1 + u1**2 + u2**2)) + 2*q*(B*u1 + Ey*sqrt(1 + u1**2 + u2**2)))/2.")
    b4=evaluate("-(dxu1*q) - dyu2*q + B*dxu2*sQ - B*dyu1*sQ - dxq*u1 - dEx*sQ*u1 - dyq*u2 - dEy*sQ*u2 - (dxu1*Ex*sQ*u1)/sqrt(1 + u1**2 + u2**2) - (dyu1*Ey*sQ*u1)/sqrt(1 + u1**2 + u2**2) - (dxu2*Ex*sQ*u2)/sqrt(1 + u1**2 + u2**2) - (dyu2*Ey*sQ*u2)/sqrt(1 + u1**2 + u2**2)")
    b5=evaluate("-(p23*(4*dxu2*h*tp*u1**3*u2**3 - 2*h*tp*u1*u2*(1 + u2**2)*(dxu2 + dyu1 + 2*dyu1*u2**2) + (1 + u2**2)*((dxu1 + dyu2)*l2*(1 + u2**2) + h*(2*(1 + u2**2) + 3*dxu1*tp*(1 + u2**2) + dyu2*tp*(3 + u2**2))) + u1**2*((dxu1 + dyu2)*l2*(1 + u2**2) + h*(2*(1 + u2**2) + dxu1*tp*(1 - 3*u2**2 - 4*u2**4) + dyu2*tp*(3 + 3*u2**2 + 4*u2**4)))) + 2*h*(dyu1*(1 + u2**2)*(p33*tp*(1 + 2*u1**2)*u2**2 + h*(1 + u2**2)*(1 + u1**2 + u2**2)) + dxu2*(1 + u1**2)*(h*(1 + u2**2)*(1 + u1**2 + u2**2) - p33*tp*u1**2*(1 + 2*u2**2)) + tp*(dxp23*u1*(1 + u2**2)*(1 + u1**2 + u2**2) + u2*(dyp23*(1 + u2**2)*(1 + u1**2 + u2**2) + p33*u1*(dxu1*(1 + 2*u1**2)*(1 + u2**2) - dyu2*(1 + u1**2)*(1 + 2*u2**2))))))/(2.*h*(1 + u2**2)*(1 + u1**2 + u2**2))")
    b6=evaluate("(-(dyu2*(l2*p33*(1 + u1**2 + u2**2) + 2*h**2*(1 + u2**2)*(1 + u1**2 + u2**2) +  h*tp*(4*p23*u1*u2**3 - p33*(-3 + u2**2 + u1**2*(-3 + 4*u2**2))))) +  dxu1*(-(l2*p33*(1 + u1**2 + u2**2)) + 2*h**2*(1 + u2**2)*(1 + u1**2 + u2**2) - h*tp*(-4*p23*u1*u2*(1 + u2**2) + p33*(3*(1 + u2**2) + u1**2*(3 + 4*u2**2)))) - 2*h*(dxp33*tp*u1*(1 + u1**2 + u2**2) + p33*(1 + u1**2 - 2*dxu2*tp*u1**3*u2 + u2**2 +  2*tp*u1*u2*(-dxu2 + dyu1*u2**2)) + u2*(dyp33*tp*(1 + u1**2 + u2**2) + 2*(-(dyu1*p23*tp*u2*(1 + u2**2)) + dxu2*u1*(p23*tp*u1*u2 + h*(1 + u1**2 + u2**2))))))/(2.*h*(1 + u1**2 + u2**2))")
            
    det = evaluate("a44*(a13*a26*a35*a52*a61 - a13*a25*a36*a52*a61 - a13*a26*a32*a55*a61 + a12*a26*a33*a55*a61 + a13*a22*a36*a55*a61 - a12*a23*a36*a55*a61 - a13*a26*a35*a51*a62 + a13*a25*a36*a51*a62 + a13*a26*a31*a55*a62 - a11*a26*a33*a55*a62 - a13*a21*a36*a55*a62 + a11*a23*a36*a55*a62 - a15*(a26*a33 - a23*a36)*(a52*a61 - a51*a62) + a16*(a33*a55*(-(a22*a61) + a21*a62) + a25*a33*(a52*a61 - a51*a62) + a23*(-(a35*a52*a61) + a32*a55*a61 + a35*a51*a62 - a31*a55*a62)) + a15*(a23*a32*a51 - a22*a33*a51 - a23*a31*a52 + a21*a33*a52)*a66 + ((a25*a33 - a23*a35)*(a12*a51 - a11*a52) + (a12*a23*a31 - a11*a23*a32 - a12*a21*a33 + a11*a22*a33)*a55 + a13*(-(a25*a32*a51) + a22*a35*a51 + a25*a31*a52 - a21*a35*a52 - a22*a31*a55 + a21*a32*a55))*a66)")
     
    Ai11= evaluate("a44*(-(a26*a33*a55*a62) + a33*(-(a25*a52) + a22*a55)*a66 + a23*(a36*a55*a62 + a35*a52*a66 - a32*a55*a66))")
    Ai12= evaluate("(a16*a33 - a13*a36)*a44*a55*a62 + a44*(a15*a33*a52 - a13*a35*a52 + a13*a32*a55 - a12*a33*a55)*a66")
    Ai13= evaluate("a44*(-(a16*a23*a55*a62) + a23*(-(a15*a52) + a12*a55)*a66 + a13*(a26*a55*a62 + a25*a52*a66 - a22*a55*a66))")
    Ai14= evaluate("0*a21")
    Ai15= evaluate("(-(a16*a25*a33) + a15*a26*a33 + a16*a23*a35 - a13*a26*a35 - a15*a23*a36 + a13*a25*a36)*a44*a62 + (a15*a23*a32 - a13*a25*a32 - a15*a22*a33 + a12*a25*a33 + a13*a22*a35 - a12*a23*a35)*a44*a66")
    Ai16= evaluate("(a16*a25*a33 - a15*a26*a33 - a16*a23*a35 + a13*a26*a35 + a15*a23*a36 - a13*a25*a36)*a44*a52 + (a16*a23*a32 - a13*a26*a32 - a16*a22*a33 + a12*a26*a33 + a13*a22*a36 - a12*a23*a36)*a44*a55")
    
    Ai21= evaluate("(a26*a33 - a23*a36)*a44*a55*a61 + a44*(a25*a33*a51 - a23*a35*a51 + a23*a31*a55 - a21*a33*a55)*a66")
    Ai22= evaluate("a44*(-(a16*a33*a55*a61) + a33*(-(a15*a51) + a11*a55)*a66 + a13*(a36*a55*a61 + a35*a51*a66 - a31*a55*a66))")
    Ai23= evaluate("(a16*a23 - a13*a26)*a44*a55*a61 + a44*(a15*a23*a51 - a13*a25*a51 + a13*a21*a55 - a11*a23*a55)*a66")
    Ai24= evaluate("0*a21")
    Ai25= evaluate("(a16*a25*a33 - a15*a26*a33 - a16*a23*a35 + a13*a26*a35 + a15*a23*a36 - a13*a25*a36)*a44*a61 + (-(a15*a23*a31) + a13*a25*a31 + a15*a21*a33 - a11*a25*a33 - a13*a21*a35 + a11*a23*a35)*a44*a66")
    Ai26= evaluate("(-(a16*a25*a33) + a15*a26*a33 + a16*a23*a35 - a13*a26*a35 - a15*a23*a36 + a13*a25*a36)*a44*a51 + (-(a16*a23*a31) + a13*a26*a31 + a16*a21*a33 - a11*a26*a33 - a13*a21*a36 + a11*a23*a36)*a44*a55")
    
    Ai31= evaluate("a44*(a36*(-(a25*a52*a61) + a22*a55*a61 + a25*a51*a62 - a21*a55*a62) + a26*(a35*a52*a61 - a32*a55*a61 - a35*a51*a62 + a31*a55*a62) + (-(a25*a32*a51) + a22*a35*a51 + a25*a31*a52 - a21*a35*a52 - a22*a31*a55 + a21*a32*a55)*a66)")
    Ai32= evaluate("a44*(a36*(a15*a52*a61 - a12*a55*a61 - a15*a51*a62 + a11*a55*a62) + a16*(-(a35*a52*a61) + a32*a55*a61 + a35*a51*a62 - a31*a55*a62) + (a15*a32*a51 - a12*a35*a51 - a15*a31*a52 + a11*a35*a52 + a12*a31*a55 - a11*a32*a55)*a66)")
    Ai33= evaluate("a44*(a26*(-(a15*a52*a61) + a12*a55*a61 + a15*a51*a62 - a11*a55*a62) + a16*(a25*a52*a61 - a22*a55*a61 - a25*a51*a62 + a21*a55*a62) + (-(a15*a22*a51) + a12*a25*a51 + a15*a21*a52 - a11*a25*a52 - a12*a21*a55 +  a11*a22*a55)*a66)")
    Ai34= evaluate("0*a21")
    Ai35= evaluate("a44*(-((a26*a35 - a25*a36)*(a12*a61 - a11*a62)) + a16*(-(a25*a32*a61) + a22*a35*a61 + a25*a31*a62 - a21*a35*a62) + (-(a12*a25*a31) + a11*a25*a32 + a12*a21*a35 - a11*a22*a35)*a66 + a15*(a26*a32*a61 - a22*a36*a61 - a26*a31*a62 + a21*a36*a62 + a22*a31*a66 - a21*a32*a66))")
    Ai36= evaluate("a44*((a26*a35 - a25*a36)*(a12*a51 - a11*a52) + a15*(-(a26*a32*a51) + a22*a36*a51 + a26*a31*a52 - a21*a36*a52) + (-(a12*a26*a31) + a11*a26*a32 + a12*a21*a36 - a11*a22*a36)*a55 + a16*(a25*a32*a51 - a22*a35*a51 - a25*a31*a52 + a21*a35*a52 + a22*a31*a55 - a21*a32*a55))")
          
    Ai41= evaluate("-((a26*a33 - a23*a36)*a55*(a42*a61 - a41*a62)) + (-((a25*a33 - a23*a35)*(a42*a51 - a41*a52)) + (a23*a32*a41 - a22*a33*a41 - a23*a31*a42 + a21*a33*a42)*a55)*a66")
    Ai42= evaluate("(a16*a33 - a13*a36)*a55*(a42*a61 - a41*a62) + ((a15*a33 - a13*a35)*(a42*a51 - a41*a52) + (-(a13*a32*a41) + a12*a33*a41 + a13*a31*a42 - a11*a33*a42)*a55)*a66")
    Ai43= evaluate("-((a16*a23 - a13*a26)*a55*(a42*a61 - a41*a62)) + (-((a15*a23 - a13*a25)*(a42*a51 - a41*a52)) + (a13*a22*a41 - a12*a23*a41 - a13*a21*a42 + a11*a23*a42)*a55)*a66")
    Ai44= evaluate("a13*a26*a35*a52*a61 - a13*a25*a36*a52*a61 - a13*a26*a32*a55*a61 + a12*a26*a33*a55*a61 + a13*a22*a36*a55*a61 - a12*a23*a36*a55*a61 - a13*a26*a35*a51*a62 + a13*a25*a36*a51*a62 + a13*a26*a31*a55*a62 - a11*a26*a33*a55*a62 - a13*a21*a36*a55*a62 + a11*a23*a36*a55*a62 - a15*(a26*a33 - a23*a36)*(a52*a61 - a51*a62) + a16*(a33*a55*(-(a22*a61) + a21*a62) + a25*a33*(a52*a61 - a51*a62) + a23*(-(a35*a52*a61) + a32*a55*a61 + a35*a51*a62 - a31*a55*a62)) + a15*(a23*a32*a51 - a22*a33*a51 - a23*a31*a52 + a21*a33*a52)*a66 + ((a25*a33 - a23*a35)*(a12*a51 - a11*a52) + (a12*a23*a31 - a11*a23*a32 - a12*a21*a33 + a11*a22*a33)*a55 + a13*(-(a25*a32*a51) + a22*a35*a51 + a25*a31*a52 - a21*a35*a52 - a22*a31*a55 + a21*a32*a55))*a66")
    Ai45= evaluate("-((a16*a25*a33 - a15*a26*a33 - a16*a23*a35 + a13*a26*a35 + a15*a23*a36 -  a13*a25*a36)*(a42*a61 - a41*a62)) + ((-(a15*a23*a32) + a13*a25*a32 + a15*a22*a33 - a12*a25*a33 - a13*a22*a35 + a12*a23*a35)*a41 + (a15*a23*a31 - a13*a25*a31 - a15*a21*a33 + a11*a25*a33 + a13*a21*a35 - a11*a23*a35)*a42)*a66")
    Ai46= evaluate("(a16*a25*a33 - a15*a26*a33 - a16*a23*a35 + a13*a26*a35 + a15*a23*a36 - a13*a25*a36)*(a42*a51 - a41*a52) + ((-(a16*a23*a32) + a13*a26*a32 + a16*a22*a33 - a12*a26*a33 - a13*a22*a36 + a12*a23*a36)*a41 + (a16*a23*a31 - a13*a26*a31 - a16*a21*a33 + a11*a26*a33 + a13*a21*a36 - a11*a23*a36)*a42)*a55")
        
    Ai51= evaluate("-((a26*a33 - a23*a36)*a44*(a52*a61 - a51*a62)) + a44*(a23*a32*a51 - a22*a33*a51 - a23*a31*a52 + a21*a33*a52)*a66")
    Ai52= evaluate("(a16*a33 - a13*a36)*a44*(a52*a61 - a51*a62) + a44*(-(a13*a32*a51) + a12*a33*a51 + a13*a31*a52 - a11*a33*a52)*a66")
    Ai53= evaluate("-((a16*a23 - a13*a26)*a44*(a52*a61 - a51*a62)) + a44*(a13*a22*a51 - a12*a23*a51 - a13*a21*a52 + a11*a23*a52)*a66")
    Ai54= evaluate("0*a21")
    Ai55= evaluate("a44*((a26*a33 - a23*a36)*(a12*a61 - a11*a62) + a16*(a23*a32*a61 - a22*a33*a61 - a23*a31*a62 + a21*a33*a62) + (a12*a23*a31 - a11*a23*a32 - a12*a21*a33 + a11*a22*a33)*a66 + a13*(-(a26*a32*a61) + a22*a36*a61 + a26*a31*a62 - a21*a36*a62 - a22*a31*a66 + a21*a32*a66))")
    Ai56= evaluate("a44*((-(a16*a23*a32) + a13*a26*a32 + a16*a22*a33 - a12*a26*a33 - a13*a22*a36 + a12*a23*a36)*a51 + (a16*a23*a31 - a13*a26*a31 - a16*a21*a33 +  a11*a26*a33 + a13*a21*a36 - a11*a23*a36)*a52)")
         
    Ai61= evaluate("a44*(a33*a55*(-(a22*a61) + a21*a62) + a25*a33*(a52*a61 - a51*a62) + a23*(-(a35*a52*a61) + a32*a55*a61 + a35*a51*a62 - a31*a55*a62))")
    Ai62= evaluate("a44*(a33*a55*(a12*a61 - a11*a62) + a15*a33*(-(a52*a61) + a51*a62) + a13*(a35*a52*a61 - a32*a55*a61 - a35*a51*a62 + a31*a55*a62))")
    Ai63= evaluate("a44*(a23*a55*(-(a12*a61) + a11*a62) + a15*a23*(a52*a61 - a51*a62) + a13*(-(a25*a52*a61) + a22*a55*a61 + a25*a51*a62 - a21*a55*a62))")
    Ai64= evaluate("0*a21")
    Ai65= evaluate("a44*((-(a15*a23*a32) + a13*a25*a32 + a15*a22*a33 - a12*a25*a33 - a13*a22*a35 + a12*a23*a35)*a61 + (a15*a23*a31 - a13*a25*a31 - a15*a21*a33 +  a11*a25*a33 + a13*a21*a35 - a11*a23*a35)*a62)")
    Ai66= evaluate("a44*((a25*a33 - a23*a35)*(a12*a51 - a11*a52) + a15*(a23*a32*a51 - a22*a33*a51 - a23*a31*a52 + a21*a33*a52) + (a12*a23*a31 - a11*a23*a32 - a12*a21*a33 + a11*a22*a33)*a55 + a13*(-(a25*a32*a51) + a22*a35*a51 + a25*a31*a52 - a21*a35*a52 - a22*a31*a55 + a21*a32*a55))")
           
    sourceu1 = evaluate("(Ai11*b1 + Ai12*b2 + Ai13*b3 + Ai14*b4 + Ai15*b5 + Ai16*b6)/det")
    sourceu2 = evaluate("(Ai21*b1 +Ai22*b2 + Ai23*b3 + Ai24*b4 + Ai25*b5 + Ai26*b6)/det")
    sourceT = evaluate("(Ai31*b1 +Ai32*b2 + Ai33*b3 + Ai34*b4 + Ai35*b5 + Ai36*b6)/det")
    sourceq = evaluate("(Ai41*b1 +Ai42*b2 + Ai43*b3 + Ai44*b4 + Ai45*b5 + Ai46*b6)/det")
    sourcep23 = evaluate("(Ai51*b1 +Ai52*b2 + Ai53*b3 + Ai54*b4 + Ai55*b5 + Ai56*b6)/det")
    sourcep33 = evaluate("(Ai61*b1 +Ai62*b2 + Ai63*b3 + Ai64*b4 + Ai65*b5 + Ai66*b6)/det")
    
    
    return sourceu1,sourceu2,sourceT,sourceq,sourcep23,sourcep33
    



tcoord = 0.0

BigSteps =400
i = 0
Smallsteps = 100


q,u1,u2,p23,p33,T = init(q,u1,u2,p23,p33,T,x,y)

q0,u10,u20,p230,p330,T0 = init(q,u1,u2,p23,p33,T,x,y)
ex,ey,dex,dey = electric(0)

tt=-dt



maxU = evaluate("sqrt(u1**2+u2**2)").max()
max1 = u1.max()
max2 = u2.max()
maxT = T.max()
maxQ = q.max()
maxJE = evaluate("(ex*u1+ey*u2)/EE/sqrt(u1**2+u2**2)").max()

deltaT = T.max()-T.min()
deltaU = evaluate("sqrt(u1**2+u2**2)").max()-evaluate("sqrt(u1**2+u2**2)").min()

#fft = u10[Nx/2]

for steps in range(BigSteps):
    
    tt=tt+dt
    
    
    i=i+1
    time1=time()
    #print u1.max()
    #print TT.max()
    #print BB.max()
    for st in range(Smallsteps):
        tt= tt+dt
        k1u1,k1u2,k1T,k1q,k1p23,k1p33 = source(q,u1,u2,p23,p33,T,tt,x,y)
        k2u1,k2u2,k2T,k2q,k2p23,k2p33 = source(q+dt/2*k1q,u1+dt/2*k1u1,u2+dt/2*k1u2,p23+dt/2*k1p23,p33+dt/2*k1p33,T+dt/2*k1T,tt+dt/2,x,y)
        k3u1,k3u2,k3T,k3q,k3p23,k3p33 = source(q+dt/2*k2q,u1+dt/2*k2u1,u2+dt/2*k2u2,p23+dt/2*k2p23,p33+dt/2*k2p33,T+dt/2*k2T,tt+dt/2,x,y)
        k4u1,k4u2,k4T,k4q,k4p23,k4p33 = source(q+dt/2*k3q,u1+dt/2*k3u1,u2+dt/2*k3u2,p23+dt/2*k3p23,p33+dt/2*k3p33,T+dt/2*k3T,tt+dt,x,y)


        q = q+dt/6*(k1q+2*k2q+2*k3q+k4q)
        u1 = u1+dt/6*(k1u1+2*k2u1+2*k3u1+k4u1)
        u2 = u2+dt/6*(k1u2+2*k2u2+2*k3u2+k4u2)
        p23 = p23+dt/6*(k1p23+2*k2p23+2*k3p23+k4p23)
        p33 = p33+dt/6*(k1p33+2*k2p33+2*k3p33+k4p33)
        T = T+dt/6*(k1T+2*k2T+2*k3T+k4T)      
        
        
        


    maxT = append(maxT,T.max())
    maxU = append(maxU,evaluate("sqrt(u1**2+u2**2)").max())
    max1 = append(max1,u1.max())
    max2 = append(max2,u2.max())
    
    deltaT = append(deltaT,T.max()-T.min())
    deltaU = append(deltaU,evaluate("sqrt(u1**2+u2**2)").max()-evaluate("sqrt(u1**2+u2**2)").min())
    
    
    with file(filename+'T'+'.txt','a') as plotfile:
        #plotfile.write('Physical Time: {0}\n'.format(tcoord))
        savetxt(plotfile, T,delimiter=",")
    with file(filename+'u1'+'.txt','a') as plotfile:
        #plotfile.write('Physical Time: {0}\n'.format(tcoord))
        savetxt(plotfile, u1, delimiter=",")
    with file(filename+'u2'+'.txt','a') as plotfile:
        #plotfile.write('Physical Time: {0}\n'.format(tcoord))
        savetxt(plotfile, u2, delimiter=",")
    with file(filename+'q'+'.txt','a') as plotfile:
        #plotfile.write('Physical Time: {0}\n'.format(tcoord))
        savetxt(plotfile, q, delimiter=",")


    time2 =time()




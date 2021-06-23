import dolfin as df
import numpy as np

def get_residual_form(u, v, rho_e, method ='RAMP'):

    df.dx = df.dx(metadata={"quadrature_degree":4}) 
    # stiffness = rho_e/(1 + 8. * (1. - rho_e))

    if method =='SIMP':
        stiffness = rho_e**3
    else:
        stiffness = rho_e/(1 + 8. * (1. - rho_e))

    # print('the value of stiffness is:', rho_e.vector().get_local())
    # Kinematics
    k=3e1
    E = k * stiffness
    nu = 0.3
    mu, lmbda = (E/(2*(1 + nu))), (E*nu/((1 + nu)*(1 - 2*nu)))

    d = len(u)
    I = df.Identity(d)  # Identity tensor
    F = I + df.grad(u)  # Deformation gradient
    C = F.T*F  # Right Cauchy-Green tensor

    E_ = 0.5*(C-I)
    S = 2.0*mu*E_ + lmbda*df.tr(E_)*df.Identity(d)
    psi = 0.5*df.inner(S,E_)

    # Total potential energy
    '''The first term in this equation provided this error'''
    Pi = psi*df.dx

    res = df.derivative(Pi, u, v)
    
    return res



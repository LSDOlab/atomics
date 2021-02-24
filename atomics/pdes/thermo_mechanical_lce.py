import dolfin as df
import pygmsh

def get_residual_form(u, v, rho_e, phi_angle, k, alpha, method='RAMP'):
    if method =='SIMP':
        C = rho_e**3
    else:
        C = rho_e/(1 + 8. * (1. - rho_e))
 

    E = k
    # C is the design variable, its values is from 0 to 1

    nu = 0.49 # Poisson's ratio

    lambda_ = E * nu/(1. + nu)/(1 - 2 * nu)
    mu = E / 2 / (1 + nu) #lame's parameters

    # Th = df.Constant(5e1)
    Th = df.Constant(1.)
    # Th = df.Constant(5e0)

    w_ij = 0.5 * (df.grad(u) + df.grad(u).T)
    v_ij = 0.5 * (df.grad(v) + df.grad(v).T)

    S = df.as_matrix([[-2., 0., 0. ],
                    [0. , 1., 0. ],
                    [0. , 0., 1.]])

    L = df.as_matrix([[ df.cos(phi_angle), df.sin(phi_angle), 0. ],
                    [-df.sin(phi_angle), df.cos(phi_angle), 0. ],
                    [ 0. , 0., 1. ]])


    alpha_e = alpha*C
    eps = w_ij - alpha_e*Th*L.T*S*L 

    d = len(u)

    sigm = lambda_*df.div(u)*df.Identity(d) + 2*mu*eps
    a = df.inner(sigm, v_ij) * df.dx 
    
    return a



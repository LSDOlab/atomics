import dolfin as df


def get_residual_form(u, v, rho_e, E = 1):
    C = rho_e/(1 + 8. * (1. - rho_e))

    E = 1. * C # C is the design variable, its values is from 0 to 1

    nu = 0.3 # Poisson's ratio

    lambda_ = E * nu/(1. + nu)/(1 - 2 * nu)
    mu = E / 2 / (1 + nu) #lame's parameters


    w_ij = 0.5 * (df.grad(u) + df.grad(u).T)
    v_ij = 0.5 * (df.grad(v) + df.grad(v).T)
    
    d = len(u)

    sigm = lambda_*df.div(u)*df.Identity(d) + 2*mu*w_ij

    a = df.inner(sigm, v_ij) * df.dx 
    
    return a
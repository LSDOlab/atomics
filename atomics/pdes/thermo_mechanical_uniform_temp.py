import dolfin as df


def get_residual_form(u, v, rho_e, Th, k = 199.5e9, alpha = 15.4e-6):
    C = rho_e/(1 + 8. * (1. - rho_e))

    E = k * C # C is the design variable, its values is from 0 to 1

    nu = 0.3 # Poisson's ratio
    Th = Th - df.Constant(20.)
    Th = 0.

    lambda_ = E * nu/(1. + nu)/(1 - 2 * nu)
    mu = E / 2 / (1 + nu) #lame's parameters

    # Th = df.Constant(7)

    w_ij = 0.5 * (df.grad(u) + df.grad(u).T)
    v_ij = 0.5 * (df.grad(v) + df.grad(v).T)

    d = len(u)

    sigm = (lambda_ * df.tr(w_ij) - alpha * (3. * lambda_ + 2. * mu) * Th) * df.Identity(d) + 2 * mu * w_ij

    a = df.inner(sigm, v_ij) * df.dx 
    
    return a


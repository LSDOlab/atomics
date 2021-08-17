import dolfin as df
import numpy as np


def get_residual_form(u, v, rho_e, method='RAMP'):

    df.dx = df.dx(metadata={"quadrature_degree": 4})
    # stiffness = rho_e/(1 + 8. * (1. - rho_e))

    if method == 'SIMP':
        stiffness = rho_e**3
    else:  #RAMP
        stiffness = rho_e / (1 + 8. * (1. - rho_e))

    # print('the value of stiffness is:', rho_e.vector().get_local())
    # Kinematics
    k = 3e1
    E = k * stiffness
    nu = 0.3
    mu, lmbda = (E / (2 * (1 + nu))), (E * nu / ((1 + nu) * (1 - 2 * nu)))

    d = len(u)
    I = df.Identity(d)  # Identity tensor
    F = I + df.grad(u)  # Deformation gradient
    C = F.T * F  # Right Cauchy-Green tensor

    E_ = 0.5 * (C - I)  # Green--Lagrange strain
    S = 2.0 * mu * E_ + lmbda * df.tr(E_) * df.Identity(
        d)  # stress tensor (C:eps)
    psi = 0.5 * df.inner(S, E_)  # 0.5*eps:C:eps

    # Total potential energy
    Pi = psi * df.dx
    # Solve weak problem obtained by differentiating Pi:
    res = df.derivative(Pi, u, v)
    return res

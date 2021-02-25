Neo-Hookean nonliear elastic problem
============================================

1. PDE
-------------

.. code-block:: python

    import dolfin as df


    def get_residual_form(u, v, rho_e, tractionBC, T, k = 10.):
        stiffness = rho_e/(1 + 8. * (1. - rho_e))
        # print('the value of stiffness is:', rho_e.vector().get_local())
        # Kinematics
        d = len(u)
        I = df.Identity(d)             # Identity tensor
        F = I + df.grad(u)             # Deformation gradient
        C = F.T*F                      # Right Cauchy-Green tensor

        # Invariants of deformation tensors
        Ic = df.tr(C)
        J  = df.det(F)

        E = k * stiffness # rho_e is the design variable, its values is from 0 to 1

        nu = 0.3 # Poisson's ratio

        lambda_ = E * nu/(1. + nu)/(1 - 2 * nu)
        mu = E / 2 / (1 + nu) #lame's parameters

        # Stored strain energy density (compressible neo-Hookean model)
        psi = (mu/2)*(Ic - 3) - mu*df.ln(J) + (lambda_/2)*(df.ln(J))**2
        # print('the length of psi is:',len(psi.vector()))

        B  = df.Constant((0.0, 0.0)) 

        # Total potential energy
        '''The first term in this equation provided this error'''
        Pi = psi*df.dx - df.dot(B, u)*df.dx - df.dot(T, u)*tractionBC 

        res = df.derivative(Pi, u, v)
        
        return res



Defining a new PDE
----------------------
ATOmiCS wraps ``FEniCS`` for the solution of the PDEs.
The input for FEniCS is the variational (weak) form.
Thus, to implement a physics problem that is not included in ATOmiCS,
the users need to implement the variational form using FEniCS unified form language (UFL) and add it to the ATOmiCS PDE folder
(``atomics/atomics/pdes``).
We demonstrate this process using an example that adds the st. Venant--Kirchhoff model to the ATOmiCS the PDE folder.

    1.1 Derive the variational form

        St. Venant--Kirchhoff model is a type of simple hyperelastic problem.
        The variational form of this kind of hyperelastic problem can be derived
        by minimizing the potential energy on domain :math:`\Omega`
        
        .. math::
            \Psi = \int_{\Omega} \psi(u) \, {\rm d} x
            - \int_{\Omega} b \cdot u \, {\rm d} x
            - \int_{\partial\Omega} t \cdot u \, {\rm d} s,

        where :math:`\psi` is the strain energy density for st. Venant--Kirchhoff model;
        :math:`b` and :math:`t` are the body and the traction force, respectively;
        :math:`u` is the displacements function.

        We minimize this potential energy by making its directional derivative
        with respect to the change in :math:`u`. 

        .. math::
            D_{v} \Psi = \left.
            \frac{d \Psi(u + \epsilon v)}{d\epsilon} \right|_{\epsilon = 0}=0.

        We compute the :math:`\psi(u)` according to the st. Venant--Kirchhoff model.
        We consider the deformation gradient 
        
        .. math::
            F = I + \nabla u;
        
        the right Cauchy--Green tensor

        .. math::
            C = F^{T} F;

        the Green--Lagrange strain tensor

        .. math::
            E = \frac{1}{2} (C - I)

        the stress tensor

        .. math::
            S = 2 \mu E + \lambda {\rm trace}(E)I;

        and finally, the strain energy density 

        .. math::
            \psi(u) =  \frac{1}{2}  S:E

        this final weak form is also equivalent to

        .. math::
            D_{v}  \int_{\Omega} \psi(u) \, {\rm d} x
            - \int_{\Omega} b \cdot v \
            - \int_{\partial\Omega} t \cdot v \, {\rm d} s 
            =0

    1.2 Write the variational form in ``atomics/atomics/pdes/<filename>.py`` using UFL

        We write the bilinear term as below:

        .. code-block:: python

            import dolfin as df
            import numpy as np
            
            def get_residual_form(u, v, rho_e, method='RAMP'):
                df.dx = df.dx(metadata={"quadrature_degree": 4})
                if method == 'SIMP':
                    stiffness = rho_e**3
                else:  #RAMP
                    stiffness = rho_e / (1 + 8. * (1. - rho_e))

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
            
        Since the number of integration terms in the linear term
        
        .. math::
            \int_{\Omega} b \cdot u \, {\rm d} x
            + \int_{\partial\Omega} t \cdot u \, {\rm d} s,

        and the subdomain
        where they are applied on subject to are problem specific. 
        We define this term in the  ``run_file``.
        The user can refer to :ref:`step-label` on how to define a subdomain.

        .. code-block:: python

            residual_form -= df.dot(f, v) * dss(6)




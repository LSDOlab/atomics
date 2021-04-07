Thermoelastic problem
==========================

1. PDE
-------------

The variational form for the thermoelastic problem is

.. math::   
    \int_{\Omega} \kappa \nabla T \nabla \hat{T} d x
    + \int_{\Omega} \sigma:\nabla v d x
    -\int_{\partial \Omega}\kappa(T \cdot n) \hat{T} d \partial \Omega 
    -\int_{\partial \Omega}(\sigma \cdot \eta) \cdot v d s=0,

where the :math:`\sigma`, :math:`v` are the stress tenser and the test functions for the displacements; 
:math:`T` and :math:`\hat{T}` are the function and the test functions for the temperature field.


1. Code
-------------
.. code-block:: python

  import dolfin as df

  def get_residual_form(u, v, rho_e, T, T_hat, KAPPA, k, alpha, mode='plane_stress', method='RAMP'):
      if method=='RAMP':
          C = rho_e/(1 + 8. * (1. - rho_e))
      else:
          C = rho_e**3

      E = k * C 
      # C is the design variable, its values is from 0 to 1

      nu = 0.3 # Poisson's ratio


      lambda_ = E * nu/(1. + nu)/(1 - 2 * nu)
      mu = E / 2 / (1 + nu) #lame's parameters

      if mode == 'plane_stress':
          lambda_ = 2*mu*lambda_/(lambda_+2*mu)

      I = df.Identity(len(u))
      w_ij = 0.5 * (df.grad(u) + df.grad(u).T) - alpha * I * T
      v_ij = 0.5 * (df.grad(v) + df.grad(v).T)

      d = len(u)

      sigm = lambda_*df.div(u)*df.Identity(d) + 2*mu*w_ij 

      a = df.inner(sigm, v_ij) * df.dx + \
          df.dot(C*KAPPA* df.grad(T),  df.grad(T_hat)) * df.dx
      
      return a
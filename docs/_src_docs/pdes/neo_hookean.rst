Neo-Hookean nonliear elastic problem
============================================
1. PDE
-------------
The variational form for the nonliear elastic problem is

.. math:: 
  \Pi = \int_{\Omega} \psi(u) \, {\rm d} x
  - \int_{\Omega} B \cdot u \, {\rm d} x
  - \int_{\partial\Omega} T \cdot u \, {\rm d} s  ,

where the :math:`\sigma`, :math:`v` are the stress tenser and the test functions. 

We add virtual material to the high strain regions by adding a term to the strain energy density:

.. math::
    \psi_{add} = (1-E_{\text{max}}\hat{\rho})(C_{1,e}(I_C-3)+(C_{2,e}(I_C-3))^2),

where :math:`C_{1,e}`, :math:`C_{2,e}`, and :math:`I_C` are coefficients.


2. code
-------------

.. code-block:: python

    import dolfin as df
    import numpy as np

    def get_residual_form(u, v, rho_e,V_density, tractionBC, T, iteration_number,additive ='strain',k = 8., method ='RAMP'):

        df.dx = df.dx(metadata={"quadrature_degree":4}) 
        # stiffness = rho_e/(1 + 8. * (1. - rho_e))

        if method =='SIMP':
            stiffness = rho_e**3
        else:
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
        stiffen_pow=1.
        threshold_vol= 1.

        eps_star= 0.05
        # print("eps_star--------")

        if additive == 'strain':
            print("additive == strain")

            if iteration_number == 1:
                print('iteration_number == 1')
                eps = df.sym(df.grad(u))
                eps_dev = eps - 1/3 * df.tr(eps) * df.Identity(2)
                eps_eq = df.sqrt(2.0 / 3.0 * df.inner(eps_dev, eps_dev))
                # eps_eq_proj = df.project(eps_eq, density_function_space)   
                ratio = eps_eq / eps_star
                ratio_proj  = df.project(ratio, V_density) 

                c1_e = k*(5.e-2)/(1 + 8. * (1. - (5.e-2)))/6

                c2_e = df.Function(V_density)
                c2_e.vector().set_local(5e-4 * np.ones(V_density.dim()))

                fFile = df.HDF5File(df.MPI.comm_world,"c2_e_proj.h5","w")
                fFile.write(c2_e,"/f")
                fFile.close()

                fFile = df.HDF5File(df.MPI.comm_world,"ratio_proj.h5","w")
                fFile.write(ratio_proj,"/f")
                fFile.close()
                iteration_number += 1
                E = k * stiffness 
                phi_add = (1 - stiffness)*( (c1_e*(Ic-3)) + (c2_e*(Ic-3))**2)

            else:
                ratio_proj = df.Function(V_density)
                fFile = df.HDF5File(df.MPI.comm_world,"ratio_proj.h5","r")
                fFile.read(ratio_proj,"/f")
                fFile.close()


                c2_e = df.Function(V_density)
                fFile = df.HDF5File(df.MPI.comm_world,"c2_e_proj.h5","r")
                fFile.read(c2_e,"/f")
                fFile.close()
                c1_e = k*(5.e-2)/(1 + 8. * (1. - (5.e-2)))/6

                c2_e = df.conditional(df.le(ratio_proj,eps_star), c2_e * df.sqrt(ratio_proj), 
                                        c2_e *(ratio_proj**3))
                phi_add = (1 - stiffness)*( (c1_e*(Ic-3)) + (c2_e*(Ic-3))**2)
                E = k * stiffness

                c2_e_proj =df.project(c2_e, V_density) 
                print('c2_e projected -------------')
                
                eps = df.sym(df.grad(u))
                eps_dev = eps - 1/3 * df.tr(eps) * df.Identity(2)
                eps_eq = df.sqrt(2.0 / 3.0 * df.inner(eps_dev, eps_dev))
                # eps_eq_proj = df.project(eps_eq, V_density)   
                ratio = eps_eq / eps_star
                ratio_proj  = df.project(ratio, V_density) 

                fFile = df.HDF5File(df.MPI.comm_world,"c2_e_proj.h5","w")
                fFile.write(c2_e_proj,"/f")
                fFile.close()

                fFile = df.HDF5File(df.MPI.comm_world,"ratio_proj.h5","w")
                fFile.write(ratio_proj,"/f")
                fFile.close()

        elif additive == 'vol':
            print("additive == vol")
            stiffness = stiffness/(df.det(F)**stiffen_pow)

            E = k * stiffness    

        elif additive == 'False':
            print("additive == False")
            E = k * stiffness # rho_e is the design variable, its values is from 0 to 1

        nu = 0.4 # Poisson's ratio

        lambda_ = E * nu/(1. + nu)/(1 - 2 * nu)
        mu = E / 2 / (1 + nu) #lame's parameters

        # Stored strain energy density (compressible neo-Hookean model)
        psi = (mu/2)*(Ic - 3) - mu*df.ln(J) + (lambda_/2)*(df.ln(J))**2
        # print('the length of psi is:',len(psi.vector()))
        if additive == 'strain':
            psi+=phi_add
        B  = df.Constant((0.0, 0.0)) 

        # Total potential energy
        '''The first term in this equation provided this error'''
        Pi = psi*df.dx - df.dot(B, u)*df.dx - df.dot(T, u)*tractionBC 

        res = df.derivative(Pi, u, v)
        
        return res
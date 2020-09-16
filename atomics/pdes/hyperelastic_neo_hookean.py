import dolfin as df


def get_residual_form(u, v, rho_e, tractionBC, T=df.Constant((0.0, -1.)), k = 10.):
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


# if __name__ == '__main__':
    # import numpy as np

    # NUM_ELEMENTS_X = 80
    # NUM_ELEMENTS_Y = 40
    # LENGTH_X = 160.
    # LENGTH_Y = 80.

    # mesh = df.RectangleMesh.create(
    #     [df.Point(0.0, 0.0), df.Point(LENGTH_X, LENGTH_Y)],
    #     [NUM_ELEMENTS_X, NUM_ELEMENTS_Y],
    #     df.CellType.Type.quadrilateral,
    # )

    # # Define the traction condition:
    # # here traction force is applied on the middle of the right edge
    # class TractionBoundary(df.SubDomain):
    #     def inside(self, x, on_boundary):
    #         return ((abs(x[1] - LENGTH_Y/2) < LENGTH_Y/NUM_ELEMENTS_Y + df.DOLFIN_EPS) and (abs(x[0] - LENGTH_X ) < df.DOLFIN_EPS*1.5e15))

    # # Define the traction boundary
    # sub_domains = df.MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
    # upper_edge = TractionBoundary()
    # upper_edge.mark(sub_domains, 6)
    # dss = df.Measure('ds')(subdomain_data=sub_domains)
    # tractionBC = dss(6)


    # density_function_space = df.FunctionSpace(mesh, 'DG', 0)
    # density_function = df.Function(density_function_space)
    # density_function.vector().set_local(np.ones(density_function_space.dim()))

    # displacements_function_space = df.VectorFunctionSpace(mesh, 'Lagrange', 1)
    # displacements_function = df.Function(displacements_function_space)
    # displacements_trial_function = df.TrialFunction(displacements_function_space)
    # v = df.TestFunction(displacements_function_space)
    
    # residual_form = get_residual_form(
    #     displacements_function, 
    #     v, 
    #     density_function,
    #     tractionBC
    # )
    # bcs = df.DirichletBC(displacements_function_space, df.Constant((0.0, 0.0)), '(abs(x[0]-0.) < DOLFIN_EPS)')
    # Dres = df.derivative(residual_form, displacements_function)
    # df.solve(residual_form == 0, displacements_function, bcs, J=Dres)
    # df.File("u.pvd") << displacements_function

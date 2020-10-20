import dolfin as df


def get_residual_form(u, v, rho_e, tractionBC, T, additive ='False',k = 20.):
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
    threshold_vol= 0.84
    threshold_strain= 0.2

    eps_star= 1.
    if additive == 'strain':
        # eps_eq_proj = df.Function(rho_e.function_space())
        # fFile = df.HDF5File(df.MPI.comm_world,"eps_eq_proj.h5","r")
        # fFile.read(f2,"/f")
        # fFile.close()
        # TODO: initiate the zero iteration value of the variables
        eps = df.sym(df.grad(u))
        eps_dev = eps - 1/3 * df.tr(eps) * df.Identity(2)
        eps_eq = df.sqrt(2.0 / 3.0 * df.inner(eps_dev, eps_dev))
        eps_eq_proj = df.project(eps_eq, density_function_space)   
        ratio = eps / eps_eq

        c1_e = k*(5.e-2)/(1 + 8. * (1. - (5.e-2)))/6
        ratio = eps_eq_proj/eps_star

        c2_e = df.conditional(df.le(ratio,threshold_strain), c2_e * df.sqrt(ratio), eps_eq_proj *(ratio**3))
        phi_add = (1 - stiffness)*( (c1_e*(Ic-3)) + (c2_e*(Ic-3))**2)
        E = k * stiffness

        eps = df.sym(df.grad(u))
        # TensorFunctionSpace(mesh,"DG",0) 
        eps_dev = eps - 1/3 * df.tr(eps) * df.Identity(2)
        eps_eq = df.sqrt(2.0 / 3.0 * df.inner(eps_dev, eps_dev))
        eps_eq_proj = df.project(eps_eq, density_function_space)   
        ratio = eps / eps_eq

        fFile = df.HDF5File(df.MPI.comm_world,"eps_eq_proj.h5","w")
        fFile.write(eps_eq_proj,"/f")
        fFile.close()

    elif additive == 'vol':
        print("additive == vol")
        stiffness = stiffness/(df.det(F)**stiffen_pow)

        # stiffness = df.conditional(df.le(df.det(F),threshold_vol), stiffness/(df.det(F)**stiffen_pow), stiffness)
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




if __name__ == '__main__':
    import numpy as np

    NUM_ELEMENTS_X = 120
    NUM_ELEMENTS_Y = 30
    LENGTH_X = 0.12
    LENGTH_Y = 0.03

    mesh = df.RectangleMesh.create(
        [df.Point(0.0, 0.0), df.Point(LENGTH_X, LENGTH_Y)],
        [NUM_ELEMENTS_X, NUM_ELEMENTS_Y],
        df.CellType.Type.quadrilateral,
    )

    # Define the traction condition:
    # here traction force is applied on the middle of the right edge
    class TractionBoundary(df.SubDomain):
        def inside(self, x, on_boundary):
            return ((abs(x[1] - LENGTH_Y/2) < 4 * LENGTH_Y/NUM_ELEMENTS_Y + df.DOLFIN_EPS) and (abs(x[0] - LENGTH_X ) < df.DOLFIN_EPS*1.5e15))

    # Define the traction boundary
    sub_domains = df.MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
    upper_edge = TractionBoundary()
    upper_edge.mark(sub_domains, 6)
    dss = df.Measure('ds')(subdomain_data=sub_domains)
    tractionBC = dss(6)


    density_function_space = df.FunctionSpace(mesh, 'DG', 0)
    density_function = df.Function(density_function_space)
    density_function.vector().set_local(np.ones(density_function_space.dim()))

    displacements_function_space = df.VectorFunctionSpace(mesh, 'Lagrange', 1)
    displacements_function = df.Function(displacements_function_space)
    displacements_trial_function = df.TrialFunction(displacements_function_space)
    v = df.TestFunction(displacements_function_space)
    k = 3e9
    residual_form = get_residual_form(
        displacements_function, 
        v, 
        density_function,
        tractionBC,
        df.Constant((0.0, -0.9)),
        
    )
    bcs = df.DirichletBC(displacements_function_space, df.Constant((0.0, 0.0)), '(abs(x[0]-0.) < DOLFIN_EPS)')
    Dres = df.derivative(residual_form, displacements_function)

    problem = df.NonlinearVariationalProblem(residual_form, displacements_function, bcs, Dres)
    solver  = df.NonlinearVariationalSolver(problem)
    solver.parameters['nonlinear_solver']='snes' 
    solver.parameters["snes_solver"]["line_search"] = 'bt' 
    solver.parameters["snes_solver"]["linear_solver"]='mumps' # "cg" "gmres"
    solver.parameters["snes_solver"]["maximum_iterations"]=400
    # solver.parameters["mumps"]["relative_tolerance"]=1e-9
    # solver.parameters["snes_solver"]["linear_solver"]["maximum_iterations"]=1000
    solver.parameters["snes_solver"]["error_on_nonconvergence"] = False
    solver.solve()
    # def dev(s):
    #     return s-tr(s) * Identity(3)/3.0
    # def von_mises(s):
    #     return sqrt(3.0/2.0 * inner(dev(s), dev(s)))
    # C = F.T*F 
    # E=1/2(C−I)
    eps = df.sym(df.grad(displacements_function))
    # TensorFunctionSpace(mesh,"DG",0) 
    eps_dev = eps - 1/3 * df.tr(eps) * df.Identity(2)
    eps_eq = df.sqrt(2.0 / 3.0 * df.inner(eps_dev, eps_dev))
    eps_eq_proj = df.project(eps_eq, density_function_space)   
    ratio = eps / eps_eq

    # df.solve(residual_form == 0, displacements_function, bcs, J=Dres)
    fFile = df.HDF5File(df.MPI.comm_world,"f.h5","w")
    fFile.write(eps_eq_proj,"/f")
    fFile.close()


    f2 = df.Function(density_function_space)
    fFile = df.HDF5File(df.MPI.comm_world,"f.h5","r")
    fFile.read(f2,"/f")
    fFile.close()

    df.File("u.pvd") << displacements_function
    df.File("eps_eq_proj.pvd") << eps_eq_proj
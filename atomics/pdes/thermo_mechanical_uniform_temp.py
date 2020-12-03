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


if __name__ == '__main__':
    import numpy as np

    NUM_ELEMENTS_X = 40
    NUM_ELEMENTS_Y = 40
    LENGTH_X = 0.05
    LENGTH_Y = 0.05

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
    K= 2.e6
    ALPHA = -1.e-3
    residual_form = get_residual_form(
        displacements_function, 
        v, 
        density_function,
        K,
        ALPHA
    )


    bcs = df.DirichletBC(displacements_function_space, df.Constant((0.0, 0.0)), '(abs(x[0]-0.) < DOLFIN_EPS)')
    Dres = df.derivative(residual_form, displacements_function)

    problem = df.NonlinearVariationalProblem(residual_form, displacements_function, bcs, Dres)
    solver  = df.NonlinearVariationalSolver(problem)
    solver.parameters['nonlinear_solver']='newton' 
    # solver.parameters["snes_solver"]["line_search"] = 'bt' 
    # solver.parameters["snes_solver"]["linear_solver"]='mumps' # "cg" "gmres"
    # solver.parameters["snes_solver"]["maximum_iterations"]=400
    # # solver.parameters["mumps"]["relative_tolerance"]=1e-9
    # # solver.parameters["snes_solver"]["linear_solver"]["maximum_iterations"]=1000
    # solver.parameters["snes_solver"]["error_on_nonconvergence"] = False
    solver.solve()

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
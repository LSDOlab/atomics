import dolfin as df
import pygmsh

def get_residual_form(u, v, rho_e, phi_angle, k, alpha):
    C = rho_e/(1 + 8. * (1. - rho_e))
    # C = rho_e**3

    E = k# C is the design variable, its values is from 0 to 1

    nu = 0.49 # Poisson's ratio

    lambda_ = E * nu/(1. + nu)/(1 - 2 * nu)
    mu = E / 2 / (1 + nu) #lame's parameters

    # Th = df.Constant(5e1)
    Th = df.Constant(1.)
    # Th = df.Constant(5e0)

    w_ij = 0.5 * (df.grad(u) + df.grad(u).T)
    v_ij = 0.5 * (df.grad(v) + df.grad(v).T)

    S = df.as_matrix([[-2., 0., 0. ],
                    [0. , 1., 0. ],
                    [0. , 0., 1.]])
    # phi_angle = np.pi/2
    # phi_angle = df.Function(rho_e.function_space())
    # phi_angle.vector().set_local(np.random.random(rho_e.function_space().dim()))

    L = df.as_matrix([[ df.cos(phi_angle), df.sin(phi_angle), 0. ],
                    [-df.sin(phi_angle), df.cos(phi_angle), 0. ],
                    [ 0. , 0., 1. ]])

    # L_T = as_matrix([[ df.cos(phi_angle), -df.sin(phi_angle), 0. ],
    #                 [ df.sin(phi_angle),  df.cos(phi_angle), 0. ],
    #                 [ 0. , 0., 1. ]])
    alpha_e = alpha*C
    eps = w_ij - alpha_e*Th*L.T*S*L 

    d = len(u)

    # sigm = (lambda_ * df.tr(w_ij) - alpha * (3. * lambda_ + 2. * mu) * Th) * S + 2 * mu * w_ij
    # sigm = (lambda_ * df.tr(w_ij) - alpha * (3. * lambda_ + 2. * mu) * Th) * L.T* S*L + 2 * mu * w_ij
    sigm = lambda_*df.div(u)*df.Identity(d) + 2*mu*eps
    a = df.inner(sigm, v_ij) * df.dx 
    
    return a


if __name__ == '__main__':
    import numpy as np
    '''
    1. Define constants
    '''
    # parameters for the film
    LENGTH  =  2.5e-3
    WIDTH   =  5e-3
    THICKNESS = 5e-6
    START_X = -2.5e-3
    START_Y = -2.5e-3
    START_Z = -2.5e-6
    NUM_ELEMENTS_X = NUM_ELEMENTS_Y = 50
    NUM_ELEMENTS_Z = 4
    K = 5.e0
    ALPHA = 2.5e-3

    degree = 2.5
    angle = np.pi/180 * degree

    '''
    2. Define mesh
    '''
    mesh = df.BoxMesh.create(
        [df.Point(-LENGTH, -WIDTH/2, 0), df.Point(LENGTH, WIDTH/2, THICKNESS)],
        [NUM_ELEMENTS_X, NUM_ELEMENTS_Y, NUM_ELEMENTS_Z],
        df.CellType.Type.hexahedron,
    )

    '''
    3. Define bcs (middle lines to preserve symmetry)
    '''
    class MidHBoundary(df.SubDomain):
        def inside(self, x, on_boundary):
            return ( abs(x[1] - (START_Y-START_Y)) < df.DOLFIN_EPS )

    class MidVBoundary(df.SubDomain):
        def inside(self, x, on_boundary):
            return ( abs(x[0] - (START_X-START_X)) < df.DOLFIN_EPS)

    class MidZBoundary(df.SubDomain):
        def inside(self, x, on_boundary):
            return ( abs(x[0] - (START_X-START_X)) < df.DOLFIN_EPS *1e12
                        and abs(x[2] + 0) < df.DOLFIN_EPS *1e9
                        and abs(x[1] - (START_Y-START_Y)) < df.DOLFIN_EPS*1e12)


    density_function_space = df.FunctionSpace(mesh, 'DG', 0)
    density_function = df.Function(density_function_space)
    density_val = np.ones(density_function_space.dim())
    

    density_function.vector().set_local(density_val)

    angle_function_space = df.FunctionSpace(mesh, 'DG', 0)
    angle_function = df.Function(angle_function_space)
    angle_val = np.zeros(angle_function_space.dim())
    # angle_val[0:400] = 0.
    # angle_val[400:800] = np.pi/6
    # angle_val[800:1200] = np.pi/3
    # angle_val[1200:1600] = np.pi/2

    angle_val[0:100] = 0.
    angle_val[400:500] = 0.
    angle_val[800:900] = 0.
    angle_val[1200:1300] = 0.

    angle_val[100:200] = np.pi/4
    angle_val[500:600] = np.pi/4
    angle_val[900:1000] = np.pi/4
    angle_val[1300:1400] = np.pi/4

    angle_val[200:300] = np.pi/4 * 3
    angle_val[600:700] = np.pi/4 * 3
    angle_val[1000:1100] = np.pi/4 * 3
    angle_val[1100:1200] = np.pi/4 * 3

    angle_val[300:400] = np.pi
    angle_val[700:800] = np.pi
    angle_val[1100:1200] = np.pi
    angle_val[1500:1600] = np.pi

    angle_function.vector().set_local(angle_val)

    displacements_function_space = df.VectorFunctionSpace(mesh, 'Lagrange', 1)
    displacements_function = df.Function(displacements_function_space)
    displacements_trial_function = df.TrialFunction(displacements_function_space)
    v = df.TestFunction(displacements_function_space)
    K= 5.e6
    ALPHA = 2.5e-3
    residual_form = get_residual_form(
        displacements_function, 
        v, 
        density_function,
        angle_function,
        K,
        ALPHA
    )


    Dres = df.derivative(residual_form, displacements_function)
    df.set_log_active(True)
    # bc = []
    # bc = df.DirichletBC(displacements_function_space, df.Constant((0.0, 0.0, 0.0)), LeftBottomClampBoundary())
    # bc = df.DirichletBC(displacements_function_space, df.Constant((0.0, 0.0, 0.0)), MiddleClampBoundary())

    bc_displacements_v = df.DirichletBC(displacements_function_space.sub(0), 
                                    df.Constant((0.0)), 
                                    MidVBoundary())
    bc_displacements_h = df.DirichletBC(displacements_function_space.sub(1), 
                                        df.Constant((0.0)), 
                                        MidHBoundary())
    bc_displacements_z = df.DirichletBC(displacements_function_space.sub(2), 
                                        df.Constant((0.0)), 
                                        MidZBoundary())
    bcs = [bc_displacements_v,bc_displacements_h,bc_displacements_z]

    bc = bcs


    # df.DirichletBC(displacements_function_space, df.Constant((0.0, 0.0, 0.0)), LeftBottomClampBoundary())
    # df.solve(Dres==-residual_form, displacements_function, bc, solver_parameters={"newton_solver":{"maximum_iterations":1, "error_on_nonconvergence":False}})


    problem = df.NonlinearVariationalProblem(residual_form, displacements_function, bcs, Dres)
    solver  = df.NonlinearVariationalSolver(problem)
    solver.parameters['nonlinear_solver']='snes' 
    solver.parameters["snes_solver"]["line_search"] = 'bt' 
    solver.parameters["snes_solver"]["linear_solver"]='mumps' # "cg" "gmres"
    # solver.parameters["snes_solver"]["maximum_iterations"]=400
    # solver.parameters["mumps"]["relative_tolerance"]=1e-15
    # # solver.parameters["snes_solver"]["linear_solver"]["maximum_iterations"]=1000
    # solver.parameters["snes_solver"]["error_on_nonconvergence"] = False
    solver.solve()



    desired_disp = df.Expression(( "-(1-cos(angle))*x[0]",
                                "0.0",
                                "abs(x[0])*sin(angle)"), 
                                angle=angle,
                                degree=1 )
    desired_disp = df.project(desired_disp, displacements_function_space )

    df.File("test_LCE/u20b.pvd") << displacements_function
    df.File("test_LCE/desired_u.pvd") << desired_disp

    fFile = df.HDF5File(df.MPI.comm_world,"angle_scalar.h5","w")
    fFile.write(angle_function,"/f")
    fFile.close()
    # fFile = df.HDF5File(df.MPI.comm_world,"eps_eq_proj_1000.h5","r")
    # fFile.read(f2,"/f")
    # fFile.close()
    # S = np.array([[-2., 0., 0. ],
    #                 [0. , 1., 0. ],
    #                 [0. , 0., 1. ]])
# phi_angle = np.pi/2
# L = np.array([[ np.cos(phi_angle), np.sin(phi_angle), 0. ],
#                 [-np.sin(phi_angle), np.cos(phi_angle), 0. ],
#                 [ 0. , 0., 1. ]])

# L_T = np.array([[ np.cos(phi_angle), -np.sin(phi_angle), 0. ],
#                 [ np.sin(phi_angle),  np.cos(phi_angle), 0. ],
#                 [ 0. , 0., 1. ]])

# S = np.array([[-2., 0., 0. ],
#                 [0. , 1., 0. ],
#                 [0. , 0., 1. ]])
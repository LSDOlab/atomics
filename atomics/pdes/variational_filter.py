import dolfin as df


def get_residual_form_variational_filter(rho_e_unfiltered, rho_e, C=df.Constant(7e-1)):
    v = df.TestFunction(rho_e_unfiltered.function_space())
    h = df.CellDiameter(rho_e_unfiltered.function_space().mesh())
    res_filter = (rho_e-rho_e_unfiltered)*v*df.dx + C*df.avg(h)*df.jump(rho_e)*df.jump(v)*df.dS
    # solve(res_filter==0, filtered_density)
    return res_filter#, filtered_density




if __name__ == '__main__':
    import numpy as np

    NUM_ELEMENTS_X = 40
    NUM_ELEMENTS_Y = 40
    LENGTH_X = 0.05
    LENGTH_Y = 0.05

    # mesh = df.RectangleMesh.create(
    #     [df.Point(0.0, 0.0), df.Point(LENGTH_X, LENGTH_Y)],
    #     [NUM_ELEMENTS_X, NUM_ELEMENTS_Y],
    #     df.CellType.Type.quadrilateral,
    # )
    mesh = df.UnitSquareMesh(10,10)

    # Define the traction condition:
    # here traction force is applied on the middle of the right edge
    # class TractionBoundary(df.SubDomain):
    #     def inside(self, x, on_boundary):
    #         return ((abs(x[1] - LENGTH_Y/2) < 4 * LENGTH_Y/NUM_ELEMENTS_Y + df.DOLFIN_EPS) and (abs(x[0] - LENGTH_X ) < df.DOLFIN_EPS*1.5e15))

    # # Define the traction boundary
    # sub_domains = df.MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
    # upper_edge = TractionBoundary()
    # upper_edge.mark(sub_domains, 6)
    # dss = df.Measure('ds')(subdomain_data=sub_domains)
    # tractionBC = dss(6)


    density_function_space = df.FunctionSpace(mesh, 'DG', 0)
    density_function = df.Function(density_function_space)
    density_function_unfiltered = df.Function(density_function_space)
    density_function_unfiltered.vector().set_local(np.random.random(density_function_space.dim()))

    # displacements_function_space = df.VectorFunctionSpace(mesh, 'Lagrange', 1)
    # displacements_function = df.Function(displacements_function_space)
    # displacements_trial_function = df.TrialFunction(displacements_function_space)
    v = df.TestFunction(density_function_space)
    K= 2.e6
    ALPHA = -1.e-3
    residual_form = get_residual_form_variational_filter(
        density_function_unfiltered, 
        density_function,
        C=df.Constant(100000)

    )


    # bcs = df.DirichletBC(displacements_function_space, df.Constant((0.0, 0.0)), '(abs(x[0]-0.) < DOLFIN_EPS)')
    Dres = df.derivative(residual_form, density_function)

    # problem = df.NonlinearVariationalProblem(residual_form, density_function, Dres)
    # solver  = df.NonlinearVariationalSolver(problem)
    # solver.parameters['nonlinear_solver']='newton' 
    # # solver.parameters["snes_solver"]["line_search"] = 'bt' 
    # # solver.parameters["snes_solver"]["linear_solver"]='mumps' # "cg" "gmres"
    # # solver.parameters["snes_solver"]["maximum_iterations"]=400
    # # # solver.parameters["mumps"]["relative_tolerance"]=1e-9
    # # # solver.parameters["snes_solver"]["linear_solver"]["maximum_iterations"]=1000
    # # solver.parameters["snes_solver"]["error_on_nonconvergence"] = False
    # solver.solve()
    df.solve(residual_form==0, density_function)
    import matplotlib.pyplot as plt

    df.plot(density_function_unfiltered)

    plt.figure(2)
    df.plot(density_function)
    plt.show()

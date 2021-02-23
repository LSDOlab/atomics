import dolfin as df
import pygmsh

def get_residual_form(u, v, rho_e, T, T_hat, KAPPA, k, alpha, mode='plane_stress', method='RAMP'):
    if method=='RAMP':
        C = rho_e/(1 + 8. * (1. - rho_e))
    else:
        C = rho_e**3

    E = k * C # C is the design variable, its values is from 0 to 1

    nu = 0.3 # Poisson's ratio
    # Th = Th - df.Constant(20.)


    lambda_ = E * nu/(1. + nu)/(1 - 2 * nu)
    mu = E / 2 / (1 + nu) #lame's parameters

    if mode == 'plane_stress':
        lambda_ = 2*mu*lambda_/(lambda_+2*mu)

    # Th = df.Constant(7)
    I = df.Identity(len(u))
    w_ij = 0.5 * (df.grad(u) + df.grad(u).T) - alpha * I * T
    v_ij = 0.5 * (df.grad(v) + df.grad(v).T)

    d = len(u)

    sigm = lambda_*df.div(u)*df.Identity(d) + 2*mu*w_ij 

    a = df.inner(sigm, v_ij) * df.dx + \
        df.dot(C*KAPPA* df.grad(T),  df.grad(T_hat)) * df.dx
    print("get a-------")
    
    return a


if __name__ == '__main__':
    import meshio
    import numpy as np
    import pygmsh

    # parameters for box
    MESH_L  =  1.e-2
    LENGTH  =  20.0e-2
    WIDTH   =  20.0e-2
    HIGHT   =  5.e-2
    START_X = -10.0e-2
    START_Y = -10.0e-2
    START_Z = -2.5e-2

    # parameters for cylindars (cells)
    num_cell_x   =  5
    num_cell_y   =  5
    num_cells = num_cell_x*num_cell_y
    first_cell_x = -8e-2
    first_cell_y = -8e-2
    end_cell_x   =  8e-2
    end_cell_y   =  8e-2
    x = np.linspace(first_cell_x, end_cell_x, num_cell_x)
    y = np.linspace(first_cell_y, end_cell_y, num_cell_y)
    xv, yv = np.meshgrid(x, y)
    bottom_z     = -2.5e-2
    radius       =  0.01
    axis_cell    = [0.0, 0.0, HIGHT]
    # bottom_z = -2.5e-3

    # constants for temperature field
    KAPPA = 235
    AREA_CYLINDER = 2 * np.pi * radius * HIGHT
    AREA_SIDE = WIDTH * HIGHT
    POWER = 90.
    T_0 = 20.
    q = df.Constant((POWER/AREA_CYLINDER)) # bdry heat flux

    # constants for thermoelastic model
    K = 69e9
    ALPHA = 13e-6

    #-----------------Generate--mesh----------------
    with pygmsh.occ.Geometry() as geom:
        geom.characteristic_length_min = 0.004
        geom.characteristic_length_max = 0.004
        disk_dic = {}
        disks = []

        rectangle = geom.add_rectangle([START_X, START_Y, 0.], LENGTH, WIDTH)
        for i in range(num_cells):
            name = 'disk' + str(i)
            disk_dic[name] = geom.add_disk([xv.flatten()[i], yv.flatten()[i], 0.], 0.01)
            disks.append(disk_dic[name])

        geom.boolean_difference(rectangle, geom.boolean_union(disks))

        mesh = geom.generate_mesh()
        mesh.write("test_2d.vtk")



    f_l = df.Constant(( 1.e6/AREA_SIDE, 0.)) 
    f_r = df.Constant((-1.e6/AREA_SIDE, 0.)) 
    f_b = df.Constant(( 0.,  1.e6/AREA_SIDE)) 
    f_t = df.Constant(( 0., -1.e6/AREA_SIDE))

    filename = 'test_2d.vtk'
    mesh = meshio.read(
        filename,  
        file_format="vtk" 
    )
    points = mesh.points
    cells = mesh.cells
    meshio.write_points_cells(
        "test_2d.xml",
        points,
        cells,
        )

    import os
    os.system('gmsh -2 test_2d.vtk -format msh2')
    os.system('dolfin-convert test_2d.msh mesh_2d.xml')
    mesh = df.Mesh("mesh_2d.xml")

    import matplotlib.pyplot as plt
    df.plot(mesh)
    plt.show()

    #-----------define-heating-boundary-------
    class HeatBoundary(df.SubDomain):
        def inside(self, x, on_boundary):
            cond_list = []
            for i in range(num_cells):
                cond = (abs(( x[0]-(xv.flatten()[i]) )**2 + ( x[1]-(yv.flatten()[i]) )**2) < (radius**2) + df.DOLFIN_EPS)
                cond_list = cond_list or cond
            return cond_list

    #-----------define-surrounding-heat-sink-boundary-------
    class SurroundingBoundary(df.SubDomain):
        def inside(self, x, on_boundary):
            return ( abs(x[0] -   START_X)  < df.DOLFIN_EPS or
                    abs(x[0] - (-START_X)) < df.DOLFIN_EPS or  
                    abs(x[1] -   START_Y)  < df.DOLFIN_EPS or
                    abs(x[1] - (-START_Y)) < df.DOLFIN_EPS)

    # Mark the HeatBoundary ass dss(6)
    sub_domains = df.MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
    heat_edge = HeatBoundary()
    heat_edge.mark(sub_domains, 6)
    # dss = df.Measure('ds')(subdomain_data=sub_domains)

    class LeftBoundary(df.SubDomain):
        def inside(self, x, on_boundary):
            return (abs(x[0] - START_X)< df.DOLFIN_EPS)

    class RightBoundary(df.SubDomain):
        def inside(self, x, on_boundary):
            return (abs(x[0] + START_X)< df.DOLFIN_EPS)

    class BottomBoundary(df.SubDomain):
        def inside(self, x, on_boundary):
            return (abs(x[1] - START_Y)< df.DOLFIN_EPS)

    class TopBoundary(df.SubDomain):
        def inside(self, x, on_boundary):
            return (abs(x[1] + START_Y)< df.DOLFIN_EPS)

    class CenterBoundary(df.SubDomain):
        def inside(self, x, on_boundary):
            return (abs(( x[0] - 0.)**2 + ( x[1] - 0.)**2) < (radius**2) + df.DOLFIN_EPS)


    # Mark the traction boundaries 8 10 12 14
    # sub_domains = df.MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
    left_edge  = LeftBoundary()
    right_edge = RightBoundary()
    bottom_edge = BottomBoundary()
    top_edge = TopBoundary()
    left_edge.mark(sub_domains, 8)
    right_edge.mark(sub_domains, 10)
    bottom_edge.mark(sub_domains, 12)
    top_edge.mark(sub_domains, 14)

    dss = df.Measure('ds')(subdomain_data=sub_domains)

    # df.File('solutions_2d/domains.pvd') << sub_domains
    # define mixed function space
    d = mesh.geometry().dim()
    cell = mesh.ufl_cell()
    displacement_fe = df.VectorElement("CG",cell,1)
    temperature_fe = df.FiniteElement("CG",cell,1)

    mixed_fs = df.FunctionSpace(mesh, df.MixedElement([displacement_fe,temperature_fe]))

    mixed_function = df.Function(mixed_fs)
    displacements_function,temperature_function = df.split(mixed_function)
    # mixed_function.split()

    v,T_hat = df.TestFunctions(mixed_fs)

    density_function_space = df.FunctionSpace(mesh, 'DG', 0)
    density_function = df.Function(density_function_space)
    density_function.vector().set_local(np.ones(density_function_space.dim())*0.999)

    residual_form = get_residual_form(
        displacements_function, 
        v, 
        density_function,
        temperature_function,
        T_hat,
        KAPPA,
        K,
        ALPHA
    )
    residual_form -= ( df.dot(f_l, v) * dss(8) + df.dot(f_r, v) * dss(10) + df.dot(f_b, v) * dss(12) 
                        + df.dot(f_t, v) * dss(14) + q*T_hat*dss(6) )
    print("get residual_form-------")

    bc_displacements = df.DirichletBC(mixed_fs.sub(0).sub(0), df.Constant((0.0)), CenterBoundary())
    bc_displacements_1 = df.DirichletBC(mixed_fs.sub(0).sub(1), df.Constant((0.0)), CenterBoundary())

    bc_temperature = df.DirichletBC(mixed_fs.sub(1), df.Constant(T_0), SurroundingBoundary())
    bcs = [bc_displacements, bc_temperature, bc_displacements_1]

    Dres = df.derivative(residual_form, mixed_function)
    print("Dres-------")

    # df.solve(residual_form==0, mixed_function, bcs)

    problem = df.NonlinearVariationalProblem(residual_form, mixed_function, bcs, Dres)
    solver  = df.NonlinearVariationalSolver(problem)
    solver.parameters['nonlinear_solver']='newton' 

    solver.solve()

    displacements_function_val, temperature_function_val= mixed_function.split()
    
    # df.File("mixed_displacements_2d.pvd") << displacements_function_val
    # df.File("mixed_temperature_2d.pvd") << temperature_function_val

    C = density_function/(1 + 8. * (1. - density_function))

    E = K * C # C is the design variable, its values is from 0 to 1

    nu = 0.3 # Poisson's ratio
    # Th = Th - df.Constant(20.)


    lambda_ = E * nu/(1. + nu)/(1 - 2 * nu)
    mu = E / 2 / (1 + nu) #lame's parameters


    lambda_ = 2*mu*lambda_/(lambda_+2*mu)

    # Th = df.Constant(7)
    I = df.Identity(len(displacements_function))
    w_ij = 0.5 * (df.grad(displacements_function) + df.grad(displacements_function).T) - ALPHA * I * temperature_function
    v_ij = 0.5 * (df.grad(v) + df.grad(v).T)

    d = len(displacements_function)

    def sigma(u):
        sigm = lambda_*df.div(u)*df.Identity(d) + 2*mu*w_ij 
        return sigm

    import matplotlib
    import matplotlib.pyplot as plt
    # Plot stress
    s = sigma(displacements_function) - (1./3)*df.tr(sigma(displacements_function))*df.Identity(d)  # deviatoric stress
    von_Mises = df.sqrt(3./2*df.inner(s, s))

    
    V = df.FunctionSpace(mesh, 'DG', 0)
    von_Mises = df.project(von_Mises, V)
    von_Mises_max = von_Mises.vector().get_local().max()
    von_Mises_min = von_Mises.vector().get_local().min()
    # der = df.derivative(von_Mises, mixed_function)
    von_Mises_form = (1/df.CellVolume(mesh)) * von_Mises * df.TestFunction(V) * df.dx
    von_Mises_val = df.assemble(von_Mises_form).get_local()
    der = df.derivative(von_Mises_form, mixed_function)
    der_val = df.assemble(der).array()
    df.plot(von_Mises, title='von Mises stress')
    plt.show()
    df.File("von_Mises.pvd") << von_Mises

    derivative_petsc_sparse = df.as_backend_type(df.assemble(der)).mat()
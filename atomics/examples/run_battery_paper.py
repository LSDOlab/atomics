import dolfin as df
import meshio
import numpy as np
import pygmsh

import openmdao.api as om

from atomics.api import PDEProblem, AtomicsGroup
from atomics.pdes.thermo_mechanical_uniform_temp import get_residual_form
from atomics.general_filter_comp import GeneralFilterComp

# parameters for box
MESH_L  =  3.e-2
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

# constants for thermoelastic model
K = 69e9
ALPHA = 13e-6



# ''' 
# 1 (1-2 generate mesh-vtk format)
#-----------------Generate--mesh----------------
with pygmsh.occ.Geometry() as geom:
    geom.characteristic_length_min = 0.03
    geom.characteristic_length_max = 0.03

    cylinder_dic = {}
    cylinders = []

    # add box
    rectangle = geom.add_box([START_X, START_Y, START_Z], [LENGTH, WIDTH, HIGHT], MESH_L)

    # add cylindars (cells)  
    for i in range(num_cells):
        name = 'cylinder' + str(i)
        cylinder_dic[name] = geom.add_cylinder(
                                                [xv.flatten()[i], yv.flatten()[i], bottom_z], 
                                                axis_cell, radius
                                                )
        cylinders.append(cylinder_dic[name])

    geom.boolean_difference(rectangle, geom.boolean_union(cylinders))
    mesh = geom.generate_mesh()
    mesh.write("test.vtk")

# 2
# '''

# '''
# 3 (3-4 thermal analysis (model the heating power as-(bdry heat fluxes*Area)):
#                       -read mesh
#                       -define heating bdry
# )

#-----------------read--mesh-------------
filename = 'test.vtk'
mesh = meshio.read(
    filename,  
    file_format="vtk"  
)
points = mesh.points
cells = mesh.cells
meshio.write_points_cells(
    "test.xml",
    points,
    cells,
    )

mesh = df.Mesh("test.xml")


#-----------define-heating-boundary-------
class HeatBoundary(df.SubDomain):
    def inside(self, x, on_boundary):
        cond_list = []
        for i in range(num_cells):
            cond = (abs(( x[0]-(xv.flatten()[i]) )**2 + ( x[1]-(yv.flatten()[i]) )**2) < (radius**2) + df.DOLFIN_EPS)
            cond_list = cond_list or cond
        return cond_list

# Mark the HeatBoundary ass dss(6)
sub_domains = df.MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
heat_edge = HeatBoundary()
heat_edge.mark(sub_domains, 6)
dss = df.Measure('ds')(subdomain_data=sub_domains)

# df.File('solutions/domains.pvd') << sub_domains

temperature_f_s = df.FunctionSpace(mesh, 'Lagrange', 1)

#-----------sourounding-boundary--as--constants---
class SurroundingBoundary(df.SubDomain):
    def inside(self, x, on_boundary):
        return ( abs(x[0] -   START_X)  < df.DOLFIN_EPS or
                 abs(x[0] - (-START_X)) < df.DOLFIN_EPS or  
                 abs(x[1] -   START_Y)  < df.DOLFIN_EPS or
                 abs(x[1] - (-START_Y)) < df.DOLFIN_EPS)

bc = df.DirichletBC(temperature_f_s, df.Constant(T_0), SurroundingBoundary())

T = df.Function(temperature_f_s)
T_hat = df.TestFunction(temperature_f_s)
f = df.Constant((POWER/AREA_CYLINDER))
a = KAPPA*df.inner(df.grad(T), df.grad(T_hat))*df.dx
L = f*T_hat*dss(6)

u = df.Function(temperature_f_s)
df.solve(a - L==0, T, bc)
file = df.File("poisson.pvd")
file << T
# import matplotlib.pyplot as plt
# df.plot(u)
# plt.show()

fFile = df.HDF5File(df.MPI.comm_world,"temperature.h5","w")
fFile.write(u,"/f")
fFile.close()
4
# ''' 
# -----------read-temperature--data---

temperature = df.Function(temperature_f_s)
fFile = df.HDF5File(df.MPI.comm_world,"temperature.h5","r")
fFile.read(temperature,"/f")
fFile.close()

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

# c_edge = CenterBoundary()


left_edge.mark(sub_domains, 8)
right_edge.mark(sub_domains, 10)
top_edge.mark(sub_domains, 12)
bottom_edge.mark(sub_domains, 14)

# # c_edge.mark(sub_domains, 16)

dss = df.Measure('ds')(subdomain_data=sub_domains)

# PDE problem
pde_problem = PDEProblem(mesh)

f_l = df.Constant(( 1.e6/AREA_SIDE, 0.,   0.))
f_r = df.Constant((-1.e6/AREA_SIDE, 0.,   0.))
f_b = df.Constant(( 0.,   1.e6/AREA_SIDE, 0.))
f_t = df.Constant(( 0.,  -1.e6/AREA_SIDE, 0.))

# Add input to the PDE problem:
# name = 'density', function = density_function (function is the solution vector here)
density_function_space = df.FunctionSpace(mesh, 'DG', 0)
density_function = df.Function(density_function_space)
pde_problem.add_input('density', density_function)


# Add states to the PDE problem (line 58):
# name = 'displacements', function = displacements_function (function is the solution vector here)
# residual_form = get_residual_form(u, v, rho_e) from atomics.pdes.thermo_mechanical_uniform_temp
# *inputs = density (can be multiple, here 'density' is the only input)
displacements_function_space = df.VectorFunctionSpace(mesh, 'Lagrange', 1)
displacements_function = df.Function(displacements_function_space)
v = df.TestFunction(displacements_function_space)
residual_form = get_residual_form(
    displacements_function, 
    v, 
    density_function,
    temperature,
    K,
    ALPHA,
)
residual_form -= ( df.dot(f_l, v) * dss(8) + df.dot(f_r, v) * dss(10) + df.dot(f_b, v) * dss(14) + df.dot(f_t, v) * dss(12) )
pde_problem.add_state('displacements', displacements_function, residual_form, 'density')


# Add output-avg_density to the PDE problem:
volume = df.assemble(df.Constant(1.) * df.dx(domain=mesh))
avg_density_form = density_function / (df.Constant(1. * volume)) * df.dx(domain=mesh)
pde_problem.add_scalar_output('avg_density', avg_density_form, 'density')

# Add output-compliance to the PDE problem:
compliance_form = df.dot(f_l, displacements_function) * dss(8) +df.dot(f_r, displacements_function) * dss(10) +\
                    df.dot(f_b, displacements_function) * dss(12) + df.dot(f_t, displacements_function) * dss(14)
pde_problem.add_scalar_output('compliance', compliance_form, 'displacements')

# Add boundary conditions to the PDE problem:
pde_problem.add_bc(df.DirichletBC(displacements_function_space, df.Constant((0.0, 0.0, 0.0)), CenterBoundary()))


# Define the OpenMDAO problem and model

prob = om.Problem()

num_dof_density = pde_problem.inputs_dict['density']['function'].function_space().dim()

comp = om.IndepVarComp()
comp.add_output(
    'density_unfiltered', 
    shape=num_dof_density, 
    val=np.random.random(num_dof_density) * 0.86,
)
prob.model.add_subsystem('indep_var_comp', comp, promotes=['*'])


comp = GeneralFilterComp(density_function_space=density_function_space)
prob.model.add_subsystem('general_filter_comp', comp, promotes=['*'])


group = AtomicsGroup(pde_problem=pde_problem)
prob.model.add_subsystem('atomics_group', group, promotes=['*'])

prob.model.add_design_var('density_unfiltered',upper=1, lower=1e-4)
prob.model.add_objective('compliance')
prob.model.add_constraint('avg_density',upper=0.80)

prob.driver = driver = om.pyOptSparseDriver()
driver.options['optimizer'] = 'SNOPT'
driver.opt_settings['Verify level'] = 0
driver.opt_settings['Major iterations limit'] = 500
driver.opt_settings['Minor iterations limit'] = 100000
driver.opt_settings['Iterations limit'] = 100000000
driver.opt_settings['Major step limit'] = 2.0

driver.opt_settings['Major feasibility tolerance'] = 1.0e-6
driver.opt_settings['Major optimality tolerance'] =2.e-12

prob.setup()
prob.run_model()

# prob.check_partials(compact_print=True)
# print(prob['compliance']); exit()

prob.run_driver()

#save the solution vector
df.File('solutions/displacement.pvd') << displacements_function
df.File('solutions/stiffness_th_55.pvd') << density_function


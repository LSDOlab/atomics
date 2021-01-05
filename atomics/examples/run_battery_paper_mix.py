import dolfin as df
import meshio
import numpy as np
import pygmsh

import openmdao.api as om

from atomics.api import PDEProblem, AtomicsGroup
from atomics.pdes.thermo_mechanical_mix_2d_stress import get_residual_form
from atomics.general_filter_comp import GeneralFilterComp

'''
This code tries to replicate the Kamapadi 2020 results
with a 2d linear plane stress model
'''

'''
1. Define constants
'''

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
radius       =  0.01
axis_cell    = [0.0, 0.0, HIGHT]

# constants for temperature field
KAPPA = 235
AREA_CYLINDER = 2 * np.pi * radius * HIGHT
AREA_SIDE = WIDTH * HIGHT
POWER = 90.
T_0 = 20.
q = df.Constant((POWER/AREA_CYLINDER)) # bdry heat flux


# constants for thermoelastic model
K = 69e7
ALPHA = 13e-6
f_l = df.Constant(( 1.e6/AREA_SIDE, 0.)) 
f_r = df.Constant((-1.e6/AREA_SIDE, 0.)) 
f_b = df.Constant(( 0.,  1.e6/AREA_SIDE)) 
f_t = df.Constant(( 0., -1.e6/AREA_SIDE))

'''
2. Define mesh
'''

#-----------------Generate--mesh----------------
with pygmsh.occ.Geometry() as geom:
    geom.characteristic_length_min = 0.002
    geom.characteristic_length_max = 0.002
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


#-----------------read--mesh-------------
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

'''
3. Define traction bc subdomains
'''

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

df.File('solutions_2d/domains.pvd') << sub_domains

'''
4. Define PDE problem
'''

# PDE problem
pde_problem = PDEProblem(mesh)

'''
4. 1. Add input to the PDE problem
'''
# name = 'density', function = density_function (function is the solution vector here)
density_function_space = df.FunctionSpace(mesh, 'DG', 0)
density_function = df.Function(density_function_space)
pde_problem.add_input('density', density_function)

'''
4. 2. Add states
'''
# Define mixed function space-split into temperature and displacement FS
d = mesh.geometry().dim()
cell = mesh.ufl_cell()
displacement_fe = df.VectorElement("CG",cell,1)
temperature_fe = df.FiniteElement("CG",cell,1)

mixed_fs = df.FunctionSpace(mesh, df.MixedElement([displacement_fe,temperature_fe]))

mixed_function = df.Function(mixed_fs)
displacements_function,temperature_function = df.split(mixed_function)

v,T_hat = df.TestFunctions(mixed_fs)

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
pde_problem.add_state('mixed_states', mixed_function, residual_form, 'density')

'''
4. 3. Add outputs
'''

# Add output-avg_density to the PDE problem:
volume = df.assemble(df.Constant(1.) * df.dx(domain=mesh))
avg_density_form = density_function / (df.Constant(1. * volume)) * df.dx(domain=mesh)
pde_problem.add_scalar_output('avg_density', avg_density_form, 'density')
print("Add output-avg_density-------")

# Add output-compliance to the PDE problem:

compliance_form = df.dot(f_l, displacements_function) * dss(8) +df.dot(f_r, displacements_function) * dss(10) +\
                    df.dot(f_b, displacements_function) * dss(12) + df.dot(f_t, displacements_function) * dss(14)
pde_problem.add_scalar_output('compliance', compliance_form, 'mixed_states')
print("Add output-compliance-------")


'''
4. 3. Add bcs
'''

bc_displacements = df.DirichletBC(mixed_fs.sub(0).sub(0), df.Constant((0.0)), CenterBoundary())
bc_displacements_1 = df.DirichletBC(mixed_fs.sub(0).sub(1), df.Constant((0.0)), CenterBoundary())

bc_temperature = df.DirichletBC(mixed_fs.sub(1), df.Constant(T_0), SurroundingBoundary())

# Add boundary conditions to the PDE problem:
pde_problem.add_bc(bc_displacements)
pde_problem.add_bc(bc_displacements_1)
pde_problem.add_bc(bc_temperature)


# Define the OpenMDAO problem and model

prob = om.Problem()

num_dof_density = pde_problem.inputs_dict['density']['function'].function_space().dim()

comp = om.IndepVarComp()
comp.add_output(
    'density_unfiltered', 
    shape=num_dof_density, 
    val=np.ones(num_dof_density),
    # val=np.random.random(num_dof_density) * 0.86,
)
prob.model.add_subsystem('indep_var_comp', comp, promotes=['*'])


comp = GeneralFilterComp(density_function_space=density_function_space)
prob.model.add_subsystem('general_filter_comp', comp, promotes=['*'])


group = AtomicsGroup(pde_problem=pde_problem)
prob.model.add_subsystem('atomics_group', group, promotes=['*'])

prob.model.add_design_var('density_unfiltered',upper=1, lower=1e-4)
prob.model.add_objective('compliance')
prob.model.add_constraint('avg_density',upper=0.70)

prob.driver = driver = om.pyOptSparseDriver()
driver.options['optimizer'] = 'SNOPT'
driver.opt_settings['Verify level'] = 0
driver.opt_settings['Major iterations limit'] = 5000
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

displacements_function_val, temperature_function_val= mixed_function.split()

#save the solution vector
df.File('solutions/displacement_whole.pvd') << displacements_function_val
df.File('solutions/temperature_whole.pvd') << temperature_function_val

df.File('solutions/stiffness_whole.pvd') << density_function


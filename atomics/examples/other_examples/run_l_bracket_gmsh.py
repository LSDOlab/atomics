import dolfin as df
import numpy as np

import openmdao.api as om

from atomics.api import PDEProblem, AtomicsGroup
from atomics.pdes.linear_elastic import get_residual_form # here we select a linear elastic PDE
from atomics.general_filter_comp import GeneralFilterComp

import meshio

# import L-shaped bracket from gmsh vtk file and use meshio to convert to xml file
# (TODO: XDMF not working)
np.random.seed(0)
filename = 'test_gmsh_fine'
mesh = meshio.read(filename, file_format="vtk")
meshio.write_points_cells("fenics_mesh_l_bracket.xml", mesh.points, mesh.cells)

mesh = df.Mesh("fenics_mesh_l_bracket.xml")

# Define the mesh and create the PDE problem
NUM_ELEMENTS_X = 80
NUM_ELEMENTS_Y = 40
LENGTH_X = 20.
LENGTH_Y = 10.
AVG_ELEMENT_SIZE = (mesh.hmax() + mesh.hmin()) / 2.

# PDE problem
pde_problem = PDEProblem(mesh)

# Add input to the PDE problem:
# name = 'density', function = density_function (function is the solution vector here)
density_function_space = df.FunctionSpace(mesh, 'DG', 0)
density_function = df.Function(density_function_space)
pde_problem.add_input('density', density_function)

# Add states to the PDE problem:
# name = 'displacements', function = displacements_function (function is the solution vector here)
# residual_form = get_residual_form(u, v, rho_e) from atomics.pdes.linear_elastic
# *inputs = density (can be multiple, here 'density' is the only input)
displacements_function_space = df.VectorFunctionSpace(mesh, 'Lagrange', 1)
displacements_function = df.Function(displacements_function_space)
v = df.TestFunction(displacements_function_space)
residual_form = get_residual_form(
    displacements_function,
    v,
    density_function,
    method='SIMP'
)

# Define traction boundary for the traction force
class TractionBoundary(df.SubDomain):
    def inside(self, x, on_boundary):
        return ((abs(x[1] - LENGTH_Y) < AVG_ELEMENT_SIZE * 2.) and
                (abs(x[0] - LENGTH_X ) < df.DOLFIN_EPS))

# Define the traction boundary
sub_domains = df.MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
upper_edge = TractionBoundary()
upper_edge.mark(sub_domains, 6)
dss = df.Measure('ds')(subdomain_data=sub_domains)
f = df.Constant((0, -1. / 4 , 0.))  # define the traction force

residual_form -= df.dot(f, v) * dss(6)
pde_problem.add_state('displacements', displacements_function, residual_form, 'density')

# Add output-avg_density to the PDE problem:
volume = df.assemble(df.Constant(1.) * df.dx(domain=mesh))
avg_density_form = density_function / (df.Constant(1. * volume)) * df.dx(domain=mesh)
pde_problem.add_scalar_output('avg_density', avg_density_form, 'density')

# Add output-compliance to the PDE problem:
compliance_form = df.dot(f, displacements_function) * dss(6)
pde_problem.add_scalar_output('compliance', compliance_form, 'displacements')

# Add boundary conditions to the PDE problem:
pde_problem.add_bc(df.DirichletBC(displacements_function_space,
                    df.Constant((0.0, 0.0, 0.0)), '(abs(x[1]-30.) < DOLFIN_EPS)'))

# Define the OpenMDAO problem and model
prob = om.Problem()
num_dof_density = pde_problem.inputs_dict['density']['function'].function_space().dim()

# Add design variables---density on each element:
comp = om.IndepVarComp()
comp.add_output(
    'density_unfiltered',
    shape=num_dof_density,
    val=np.random.random(num_dof_density) * 0.86,
)
prob.model.add_subsystem('indep_var_comp', comp, promotes=['*'])

# add the filter and specifying the filter radius--num_element_filtered=2
comp = GeneralFilterComp(density_function_space=density_function_space, num_element_filtered=2)
prob.model.add_subsystem('general_filter_comp', comp, promotes=['*'])

group = AtomicsGroup(pde_problem=pde_problem)
prob.model.add_subsystem('atomics_group', group, promotes=['*'])

prob.model.add_design_var('density_unfiltered',upper=1, lower=1e-4)
prob.model.add_objective('compliance')
prob.model.add_constraint('avg_density',upper=0.40)

if True:
    prob.driver = driver = om.pyOptSparseDriver()
    driver.options['optimizer'] = 'SNOPT'
    driver.opt_settings['Verify level'] = 0

    driver.opt_settings['Major iterations limit'] = 100000
    driver.opt_settings['Minor iterations limit'] = 100000
    driver.opt_settings['Iterations limit'] = 100000000
    driver.opt_settings['Major step limit'] = 2.0

    driver.opt_settings['Major feasibility tolerance'] = 1.0e-6
    driver.opt_settings['Major optimality tolerance'] =2.e-10
else:
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'

prob.setup()
prob.run_model()
prob.run_driver()


#save the solution vector
df.File('solutions/displacement.pvd') << displacements_function
df.File('solutions/density_l_bracket_fine.pvd') << density_function                   
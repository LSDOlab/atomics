import dolfin as df

import openmdao.api as om

from atomics.api import PDEProblem, StatesComp
from atomics.pdes.thermo_mechanical_uniform_temp import get_residual_form

from cartesian_density_filter_comp import CartesianDensityFilterComp


# Define the mesh and create the PDE problem

num_elements_x = 60
num_elements_y = 40
length_x = .06
length_y = .04
k = 2

mesh = df.RectangleMesh.create(
    [df.Point(0.0, 0.0), df.Point(length_x, length_y)],
    [num_elements_x, num_elements_y],
    df.CellType.Type.quadrilateral,
)

class BottomBoundary(df.SubDomain):
    def inside(self, x, on_boundary):
        l_x = 0.06
        return (abs(x[1]-0.) < df.DOLFIN_EPS_LARGE and abs(x[0] - l_x/2)< 2.5e-3)

sub_domains = df.MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
upper_edge = BottomBoundary()
upper_edge.mark(sub_domains, 6)
dss = df.Measure('ds')(subdomain_data=sub_domains)
f = df.Constant((0, -1.))

pde_problem = PDEProblem(mesh)

density_function_space = df.FunctionSpace(mesh, 'DG', 0)
density_function = df.Function(density_function_space)
pde_problem.add_input('density', density_function)

displacements_function_space = df.VectorFunctionSpace(mesh, 'Lagrange', 1)
displacements_function = df.Function(displacements_function_space)
v = df.TestFunction(displacements_function_space)
residual_form = get_residual_form(
    displacements_function, 
    v, 
    density_function,
)
residual_form -= df.dot(f, v) * dss(6)
pde_problem.add_state('displacements', displacements_function, residual_form, 'density')

avg_density_function_space = df.FunctionSpace(mesh, 'R', 0)
avg_density_function = df.Function(avg_density_function_space)
volume = df.assemble(df.Constant(1.) * df.dx(domain=mesh))
avg_density_expression = density_function / (df.Constant(1. * volume)) * df.dx(domain=mesh)
pde_problem.add_output('avg_density', avg_density_function, avg_density_expression, 'density')

compliance_function_space = df.FunctionSpace(mesh, 'R', 0)
compliance_function = df.Function(compliance_function_space)
compliance_expression = df.dot(f, displacements_function) * dss(6)
pde_problem.add_output('compliance', compliance_function, compliance_expression, 'compliance')

pde_problem.add_bc(df.DirichletBC(displacements_function_space, df.Constant((0.0, 0.0)), '(abs(x[0]-0.) < DOLFIN_EPS)'))
pde_problem.add_bc(df.DirichletBC(displacements_function_space, df.Constant((0.0, 0.0)), '(abs(x[0]-0.06) < DOLFIN_EPS)'))

# num_dof_density = V_density.dim()

# Define the OpenMDAO problem and model

prob = om.Problem()

comp = om.IndepVarComp()
comp.add_output('density_unfiltered', shape=pde_problem.inputs_dict['density']['function'].function_space().dim())
prob.model.add_subsystem('indep_var_comp', comp, promotes=['*'])

comp = CartesianDensityFilterComp(
    length_x=length_x,
    length_y=length_y,
    num_nodes_x=num_elements_x + 1,
    num_nodes_y=num_elements_y + 1,
    num_dvs=pde_problem.inputs_dict['density']['function'].function_space().dim(), 
    radius=.04*2/32,
)
prob.model.add_subsystem('density_filter_comp', comp, promotes=['*'])

comp = StatesComp(
    pde_problem=pde_problem,
    state_name='displacements',
)
prob.model.add_subsystem('states_comp', comp, promotes=['*'])

prob.setup()
prob.run_model()
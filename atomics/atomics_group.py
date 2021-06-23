import openmdao.api as om

from atomics.pde_problem import PDEProblem
from atomics.states_comp import StatesComp
# from atomics.states_comp_nonlinear import StatesComp
from atomics.scalar_output_comp import ScalarOutputsComp
from atomics.field_output_comp import FieldOutputsComp

class AtomicsGroup(om.Group):
    """
    The AtomicsGroup is a OpenMDAO Group object that wraps the assembly of OpenMDAO components.
    Within ``AtomicsGroup``, the users can choose the ``pde_problem``, 
    ``linear_solver_`` (for the solution total derivatives), the`` problem_type``, 
    and whether to turn on the ``visualization`` option.
    problem.
    Parameters
    ----------
    ``linear_solver_``   solver for the total derivatives
    default='petsc_cg_ilu', 
    values=['fenics_direct', 'scipy_splu', 'fenics_krylov', 'petsc_gmres_ilu', 'scipy_cg','petsc_cg_ilu']

    ``problem_type``    solver for the FEA problem
    default='nonlinear_problem', 
    values=['linear_problem', 'nonlinear_problem', 'nonlinear_problem_load_stepping']

    ``visualization``    whether to save the iteration histories
    default='False', 
    values=['True', 'False'],
    """
    
    def initialize(self):
        self.options.declare('pde_problem', types=PDEProblem)
        self.options.declare(
            'linear_solver_', default='petsc_cg_ilu', 
            values=['fenics_direct', 'scipy_splu', 'fenics_krylov', 'petsc_gmres_ilu', 'scipy_cg','petsc_cg_ilu'],
        )
        self.options.declare(
            'problem_type', default='nonlinear_problem', 
            values=['linear_problem', 'nonlinear_problem', 'nonlinear_problem_load_stepping'],
        )
        self.options.declare(
            'visualization', default='False', 
            values=['True', 'False'],
        )

    def setup(self):
        pde_problem = self.options['pde_problem']
        linear_solver_ = self.options['linear_solver_']
        problem_type = self.options['problem_type']
        visualization = self.options['visualization']


        for state_name in pde_problem.states_dict:
            comp = StatesComp(
                pde_problem=pde_problem,
                state_name=state_name,
                linear_solver_=linear_solver_,
                problem_type=problem_type,
                visualization=visualization
            )
            self.add_subsystem('{}_states_comp'.format(state_name), comp, promotes=['*'])

        for scalar_output_name in pde_problem.scalar_outputs_dict:
            comp = ScalarOutputsComp(
                pde_problem=pde_problem,
                scalar_output_name=scalar_output_name, 
            )
            self.add_subsystem('{}_scalar_outputs_comp'.format(scalar_output_name), comp, promotes=['*'])
        
        for field_output_name in pde_problem.field_outputs_dict:
            comp = FieldOutputsComp(
                pde_problem=pde_problem,
                field_output_name=field_output_name, 
            )
            self.add_subsystem('{}_field_outputs_comp'.format(field_output_name), comp, promotes=['*'])
import openmdao.api as om

from atomics.pde_problem import PDEProblem
from atomics.states_comp import StatesComp
# from atomics.states_comp_nonlinear import StatesComp
from atomics.scalar_output_comp import ScalarOutputsComp
from atomics.field_output_comp import FieldOutputsComp


class AtomicsGroup(om.Group):

    def initialize(self):
        self.options.declare('pde_problem', types=PDEProblem)

    def setup(self):
        pde_problem = self.options['pde_problem']

        for state_name in pde_problem.states_dict:
            comp = StatesComp(
                pde_problem=pde_problem,
                state_name=state_name,
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
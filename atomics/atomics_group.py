import openmdao.api as om

from atomics.pde_problem import PDEProblem
from atomics.states_comp import StatesComp
from atomics.output_comp import ScalarOutputsComp


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
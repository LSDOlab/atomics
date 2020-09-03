from __future__ import division
import dolfin as df
from six.moves import range
import numpy as np 
import openmdao.api as om

from atomics.pde_problem import PDEProblem

class OutputsComp(om.ExplicitComponent):
    """
    Calculates the outputs (constraints/objective).
    Parameters
    ----------
    input_name[pde_problem.states_dict[output_name]['inputs']] : numpy array
    Returns
    -------
    Output_name
    """
    def initialize(self):
        self.options.declare('pde_problem', types=PDEProblem)
        self.options.declare('output_name', types=str)

    def setup(self):
        pde_problem = self.options['pde_problem']
        output_name = self.options['output_name']
        output_function = pde_problem.outputs_dict[output_name]['function']
        # IS this adding all the things as input rather than the ones for outputs comp?
        for input_name in pde_problem.outputs_dict[output_name]['inputs']:
            input_function = pde_problem.inputs_dict[input_name]['function']
            self.add_input(input_name, shape=input_function.function_space().dim())
        self.add_output(output_name, shape=output_function.function_space().dim())

        for input_name in pde_problem.outputs_dict[output_name]['inputs']:
            input_function = pde_problem.inputs_dict[input_name]['function']
            dOutput_dinput = self.compute_derivative(output_name, input_function)
            self.declare_partials(output_name, input_name, rows=dOutput_dinput.row, cols=dOutput_dinput.col)

    def compute_derivative(self, arg_name, arg_function):
        pde_problem = self.options['pde_problem']
        state_name = self.options['state_name']

        residual_form = pde_problem.states_dict[state_name]['residual_form']

        derivative_form = df.derivative(residual_form, arg_function)
        derivative_petsc_sparse = df.as_backend_type(df.assemble(derivative_form)).mat()
        derivative_csr = csr_matrix(derivative_petsc_sparse.getValuesCSR()[::-1], shape=derivative_petsc_sparse.size)

        return derivative_csr.tocoo()
              
    def compute(self, inputs, outputs):

        self.fea.rho_e.vector().set_local(inputs['rho_e'])
        # outputs['volume_fraction'] = assemble(self.fea.avg_density(self.fea.C))
        # outputs['volume_fraction'] = self.fea.avg_density(self.fea.C)
        outputs['volume_fraction'] = assemble(self.fea.avg_density(self.fea.rho_e))
        # sum(inputs['E_e'])/self.fea.volume

    def compute_partials(self, inputs, partials):
        # print('Run Constraintscomp compute_partials()-----------------------')
        self.fea.rho_e.vector().set_local(inputs['rho_e'])

        V_f = self.fea.avg_density(self.fea.rho_e)

        dV_f_dC = 1./self.fea.volume
        # print(dV_f_dC)
        # dCe_dC = assemble(derivative(Ce, self.fea.C))
        # print("derivative",derivative(V_f,self.fea.rho_e) )
        partials['volume_fraction', 'rho_e'] = assemble(derivative(V_f,self.fea.rho_e)) #.get_local()



if __name__ == '__main__':
    from set_fea_full import set_fea as fea
    from openmdao.api import Problem, Group
    from openmdao.api import IndepVarComp
    group = Group()
    num_elements  = 40
    fea = set_fea(num_elements=num_elements)
    comp = IndepVarComp()
    comp.add_output('rho_e', shape=fea.num_var, val=np.ones((fea.num_var))*0.5)
    comp.add_output('displacements', shape=fea.num_dof, val=np.random.random((fea.num_dof)))

    group.add_subsystem('input', comp, promotes=['*'])


    comp = Constraintscomp(fea=fea)
    group.add_subsystem('Constraintscomp', comp, promotes=['*'])
    prob = Problem()
    prob.model = group
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    # print(prob['displacements'])
    # print('check_partials:')
    prob.check_partials(compact_print=True)    
    # prob.check_partials(compact_print=False)
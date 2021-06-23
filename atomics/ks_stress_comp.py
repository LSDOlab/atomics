from __future__ import division
import dolfin as df
from six.moves import range
import numpy as np 
import openmdao.api as om

from atomics.pde_problem import PDEProblem


class KSStressComp(om.ExplicitComponent):
    """
    ScalarOutputsComp wraps up a the scalar output (constraints/objective)
    from a FEniCS form.
    Parameters
    ----------
    pde_problem PDEProblem
        PDEProblem is a class containing the mesh and the dictionaries of
        the boundary conditions, inputs, states, and outputs.
    scalar_output_name : str
        the name of the scalar output
    Returns
    -------
    outputs[scalar_output_name] numpy array   
        the scalar output
    Returns
    -------
    Output_name
    """
    def initialize(self):
        self.options.declare('pde_problem', types=PDEProblem)
        self.options.declare('rho', types=float)
        self.options.declare('expression')

    def setup(self):
        pde_problem = self.options['pde_problem']
        self.add_input('mixed_states', shape=pde_problem.states_dict['mixed_states']['function'].function_space().dim())
        self.add_input('density', shape=pde_problem.inputs_dict['density']['function'].function_space().dim())
        self.add_output('von_mises_max')
        self.declare_partials('*', '*')
        self.density_function_space = pde_problem.inputs_dict['density']['function'].function_space()
        self.mesh = self.density_function_space.mesh()

    def compute_derivative(self, inputs):
        pde_problem = self.options['pde_problem']
        expression = self.options['expression']
        self._set_values(inputs)

        von_Mises_max = df.project(expression, self.density_function_space).vector().get_local().max()
        rho = self.options['rho']
        self.form = (
            1/df.CellVolume(self.mesh) * df.exp(rho * (expression-von_Mises_max) )
            ) * df.dx
        print('von Mises max:--------', von_Mises_max)
        derivative_form_s = df.derivative(self.form, pde_problem.states_dict['mixed_states']['function'])
        derivative_numpy_s = 1/(rho * df.assemble(self.form)) * df.assemble(derivative_form_s).get_local()

        derivative_form_d = df.derivative(self.form, pde_problem.inputs_dict['density']['function'])
        derivative_form_d = 1/(rho * df.assemble(self.form)) * df.assemble(derivative_form_d).get_local()
        return derivative_numpy_s, derivative_form_d

    def _set_values(self, inputs):
        pde_problem = self.options['pde_problem']
        pde_problem.states_dict['mixed_states']['function'].vector().set_local(inputs['mixed_states'])
        pde_problem.inputs_dict['density']['function'].vector().set_local(inputs['density'])
              
    def compute(self, inputs, outputs):
        pde_problem = self.options['pde_problem']
        # form = self.options['form']
        rho = self.options['rho']
        expression = self.options['expression']

        self._set_values(inputs)
        von_Mises_max = df.project(expression, self.density_function_space).vector().get_local().max()

        self.form = (
            1/df.CellVolume(self.mesh) * df.exp(rho * (expression-von_Mises_max) )
            ) * df.dx 
        outputs['von_mises_max'] = 1/rho * df.ln(df.assemble(self.form)) + von_Mises_max

    def compute_partials(self, inputs, partials):
        pde_problem = self.options['pde_problem']

        self._set_values(inputs)

        derivative_numpy_s, derivative_form_d = self.compute_derivative(inputs)
        partials['von_mises_max', 'mixed_states'] = derivative_numpy_s
        partials['von_mises_max', 'density'] = derivative_form_d

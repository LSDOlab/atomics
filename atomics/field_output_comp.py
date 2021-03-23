from __future__ import division
import dolfin as df
from six.moves import range
from six import iteritems
import numpy as np 
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix

import openmdao.api as om

from atomics.pde_problem import PDEProblem


class FieldOutputsComp(om.ExplicitComponent):
    """
    FieldOutputsComp wraps up a the field (a scalar on each element) output 
    (used as constraints/objective) from a FEniCS form.
    Parameters
    ----------
    pde_problem   PDEProblem
        PDEProblem is a class containing the dictionaries of
        the boundary conditions, inputs, states, and outputs.
    field_output_name : str
        the name of the field output
    Returns
    -------
    outputs[field_output_name] numpy array
    """

    def initialize(self):
        self.options.declare('pde_problem', types=PDEProblem)
        self.options.declare('field_output_name', types=str)

    def setup(self):
        pde_problem = self.options['pde_problem']
        field_output_name = self.options['field_output_name']
        form = pde_problem.field_outputs_dict[field_output_name]['form']

        self.argument_functions_dict = argument_functions_dict = dict()
        for argument_name in pde_problem.field_outputs_dict[field_output_name]['arguments']:
            if argument_name in pde_problem.inputs_dict:
                argument_functions_dict[argument_name] = pde_problem.inputs_dict[argument_name]['function']
            elif argument_name in pde_problem.states_dict:
                argument_functions_dict[argument_name] = pde_problem.states_dict[argument_name]['function']
            else:
                print(argument_name)
                raise Exception()

        for argument_name, argument_function in iteritems(self.argument_functions_dict):
            self.add_input(argument_name, shape=argument_functions_dict[argument_name].function_space().dim())
        self.add_output(field_output_name, shape=df.assemble(form).get_local().shape) 

        for argument_name, argument_function in iteritems(self.argument_functions_dict):
            dR_dinput = self.compute_derivative(argument_name, argument_function)
            self.declare_partials(field_output_name, argument_name, rows=dR_dinput.row, cols=dR_dinput.col)

    def compute_derivative(self, argument_name, argument_function):
        pde_problem = self.options['pde_problem']
        field_output_name = self.options['field_output_name']

        form = pde_problem.field_outputs_dict[field_output_name]['form']

        derivative_form = df.derivative(form, argument_function, df.TrialFunction(argument_function.function_space()))
        derivative_petsc_sparse = df.as_backend_type(df.assemble(derivative_form)).mat()
        derivative_csr = csr_matrix(derivative_petsc_sparse.getValuesCSR()[::-1], shape=derivative_petsc_sparse.size)

        return derivative_csr.tocoo()

    def _set_values(self, inputs):
        pde_problem = self.options['pde_problem']
        field_output_name = self.options['field_output_name']
 

        for argument_name, argument_function in iteritems(self.argument_functions_dict):
            argument_function.vector().set_local(inputs[argument_name])

    def compute(self, inputs, outputs):
        pde_problem = self.options['pde_problem']
        field_output_name = self.options['field_output_name']

        self._set_values(inputs)

        form = pde_problem.field_outputs_dict[field_output_name]['form']

        outputs[field_output_name] = df.assemble(form).get_local()

    def compute_partials(self, inputs, partials):
        pde_problem = self.options['pde_problem']
        field_output_name = self.options['field_output_name']

        self._set_values(inputs)

        for argument_name, argument_function in iteritems(self.argument_functions_dict):
            dF_dinput = self.compute_derivative(argument_name, argument_function)

            partials[field_output_name, argument_name] = dF_dinput.data
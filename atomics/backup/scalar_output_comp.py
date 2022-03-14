from __future__ import division
import dolfin as df
from six.moves import range
from six import iteritems
import numpy as np
import openmdao.api as om

from atomics.pde_problem import PDEProblem

from pytikz.snopt_history_parser import SNOPTHistoryParser
from pytikz.matplotlib_utils import use_latex_fonts, get_plt_no_show, save_fig, adjust_spines


class ScalarOutputsComp(om.ExplicitComponent):
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
        self.options.declare('scalar_output_name', types=str)

    def setup(self):
        pde_problem = self.options['pde_problem']
        scalar_output_name = self.options['scalar_output_name']

        self.argument_functions_dict = argument_functions_dict = dict()
        for argument_name in pde_problem.scalar_outputs_dict[
                scalar_output_name]['arguments']:
            if argument_name in pde_problem.inputs_dict:
                argument_functions_dict[
                    argument_name] = pde_problem.inputs_dict[argument_name][
                        'function']
            elif argument_name in pde_problem.states_dict:
                argument_functions_dict[
                    argument_name] = pde_problem.states_dict[argument_name][
                        'function']
            else:
                print(argument_name)
                raise Exception()

        for argument_name, argument_function in iteritems(
                self.argument_functions_dict):
            print('argument_name input_________________-', argument_name)
            self.add_input(argument_name,
                           shape=argument_functions_dict[argument_name].
                           function_space().dim())
        self.add_output(scalar_output_name)

        self.declare_partials(scalar_output_name, '*')

        # itr_pre = np.loadtxt('itr.out')
        # parser = SNOPTHistoryParser()
        # parser.load_file(
        #     '/home/jyan_linux/Downloads/Software/ozone2/ozone2/Examples/steady_state_TO/SNOPT_print.out'
        # )
        # hist = parser.parse()
        # hist['FuncEvals'] = np.array(0.)

    def compute_derivative(self, argument_name, argument_function):
        pde_problem = self.options['pde_problem']
        scalar_output_name = self.options['scalar_output_name']

        form = pde_problem.scalar_outputs_dict[scalar_output_name]['form']

        derivative_form = df.derivative(form, argument_function)
        derivative_numpy = df.assemble(derivative_form).get_local()

        return derivative_numpy

    def _set_values(self, inputs):
        pde_problem = self.options['pde_problem']
        scalar_output_name = self.options['scalar_output_name']

        for argument_name, argument_function in iteritems(
                self.argument_functions_dict):
            argument_function.vector().set_local(inputs[argument_name])

    def compute(self, inputs, outputs):

        parser = SNOPTHistoryParser()
        parser.load_file(
            '/home/jyan_linux/Downloads/Software/ozone2/ozone2/Examples/steady_state_TO/SNOPT_print.out'
        )
        hist = parser.parse()
        # success = parser.check_success()
        # num_eval = parser.get_num_eval()
        print('number of function eveluation:', hist['FuncEvals'])
        prev = np.loadtxt(
            '/home/jyan_linux/Downloads/Software/ozone2/ozone2/Examples/steady_state_TO/itr.out'
        )
        np.savetxt(
            '/home/jyan_linux/Downloads/Software/ozone2/ozone2/Examples/steady_state_TO/itr.out',
            hist['FuncEvals'])
        now = np.loadtxt(
            '/home/jyan_linux/Downloads/Software/ozone2/ozone2/Examples/steady_state_TO/itr.out'
        )

        if prev.size != now.size:

            print('not EQUAL!!!!!!!!!!!!!', hist['Optimality'])
            for argument_name, argument_function in iteritems(
                    self.argument_functions_dict):
                print(argument_name)
                print(inputs[argument_name].mean())
        else:
            print('EEEEEEEEEEEEEEEEEEEEEEEEEQUAL!!!!!!!!!!!!!',
                  hist['Optimality'])
        #     for argument_name, argument_function in iteritems(
        #             self.argument_functions_dict):
        #         print(argument_name)
        #         print(inputs[argument_name].mean())

        pde_problem = self.options['pde_problem']
        scalar_output_name = self.options['scalar_output_name']

        self._set_values(inputs)

        form = pde_problem.scalar_outputs_dict[scalar_output_name]['form']
        # print('output:-----------')
        # print( df.assemble(form))

        outputs[scalar_output_name] = df.assemble(form)

    def compute_partials(self, inputs, partials):
        pde_problem = self.options['pde_problem']
        scalar_output_name = self.options['scalar_output_name']

        self._set_values(inputs)

        for argument_name, argument_function in iteritems(
                self.argument_functions_dict):
            derivative_numpy = self.compute_derivative(argument_name,
                                                       argument_function)

            partials[scalar_output_name,
                     argument_name][0, :] = derivative_numpy

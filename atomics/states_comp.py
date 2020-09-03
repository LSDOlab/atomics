from __future__ import division
import dolfin as df
from six.moves import range

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from petsc4py import PETSc

import openmdao.api as om

from atomics.pde_problem import PDEProblem


class StatesComp(om.ImplicitComponent):
    """
    The implicit component that wraps the FEniCS PDE solver.
    This component calls the set_fea.py that solves a linear elastic 
    problem.
    Parameters
    ----------
    rho_e[self.fea.dvs] : numpy array
        density
    Returns
    -------
    displacements[self.fea.num_dof] : numpy array
        nodel displacement vector
    """

    def initialize(self):
        self.options.declare('pde_problem', types=PDEProblem)
        self.options.declare('state_name', types=str)
        self.options.declare(
            'linear_solver', default='fenics_direct', 
            values=['fenics_direct', 'scipy_splu', 'fenics_krylov', 'petsc_gmres_ilu'],
        )

    def setup(self):
        pde_problem = self.options['pde_problem']
        state_name = self.options['state_name']
        state_function = pde_problem.states_dict[state_name]['function']

        for input_name in pde_problem.states_dict[state_name]['inputs']:
            input_function = pde_problem.inputs_dict[input_name]['function']
            self.add_input(input_name, shape=input_function.function_space().dim())
        self.add_output(state_name, shape=state_function.function_space().dim())
        dR_dstate = self.compute_derivative(state_name, state_function)
 
        self.declare_partials(state_name, state_name, rows=dR_dstate.row, cols=dR_dstate.col)
        for input_name in pde_problem.states_dict[state_name]['inputs']:
            input_function = pde_problem.inputs_dict[input_name]['function']
            dR_dinput = self.compute_derivative(state_name, input_function)
            self.declare_partials(state_name, input_name, rows=dR_dinput.row, cols=dR_dinput.col)

    def compute_derivative(self, arg_name, arg_function):
        pde_problem = self.options['pde_problem']
        state_name = self.options['state_name']

        residual_form = pde_problem.states_dict[state_name]['residual_form']

        derivative_form = df.derivative(residual_form, arg_function)
        derivative_petsc_sparse = df.as_backend_type(df.assemble(derivative_form)).mat()
        derivative_csr = csr_matrix(derivative_petsc_sparse.getValuesCSR()[::-1], shape=derivative_petsc_sparse.size)

        return derivative_csr.tocoo()

    def _set_values(self, inputs, outputs):
        pde_problem = self.options['pde_problem']
        state_name = self.options['state_name']
        state_function = pde_problem.states_dict[state_name]['function']

        state_function.vector().set_local(outputs[state_name])
        for input_name in pde_problem.states_dict[state_name]['inputs']:
            input_function = pde_problem.inputs_dict[input_name]['function']
            input_function.vector().set_local(inputs[input_name])

    def apply_nonlinear(self, inputs, outputs, residuals):
        pde_problem = self.options['pde_problem']
        state_name = self.options['state_name']

        residual_form = pde_problem.states_dict[state_name]['residual_form']

        self._set_values(inputs, outputs)
        residuals[state_name] = df.assemble(residual_form).get_local()

    def solve_nonlinear(self, inputs, outputs):
        pde_problem = self.options['pde_problem']
        state_name = self.options['state_name']

        state_function = pde_problem.states_dict[state_name]['function']
        residual_form = pde_problem.states_dict[state_name]['residual_form']

        self._set_values(inputs, outputs)

        self.derivative_form = df.derivative(residual_form, state_function)

        df.set_log_active(True)
        df.solve(residual_form==0, state_function, bcs=pde_problem.bcs_list, J=self.derivative_form,
              solver_parameters={"newton_solver":{"maximum_iterations":1, "error_on_nonconvergence":False}})

        self.L = -residual_form

        outputs[state_name] = state_function.vector().get_local()

    def linearize(self, inputs, outputs, partials):
        pde_problem = self.options['pde_problem']
        state_name = self.options['state_name']
        state_function = pde_problem.states_dict[state_name]['function']

        self.dR_dstate = self.compute_derivative(state_name, state_function)
        partials[state_name,state_name] = self.dR_dstate.data

        for input_name in pde_problem.states_dict[state_name]['inputs']:
            input_function = pde_problem.inputs_dict[input_name]['function']
            dR_dinput = self.compute_derivative(state_name, input_function)
            partials[state_name,input_name] = dR_dinput.data

    # should I write those linear_solver options outside/seperately
    def solve_linear(self, d_outputs, d_residuals, mode):

        linear_solver = self.options['linear_solver']

        pde_problem = self.options['pde_problem']
        state_name = self.options['state_name']
        state_function = pde_problem.states_dict[state_name]['function']

        residual_form = pde_problem.states_dict[state_name]['residual_form']
        A, _ = df.assemble_system(self.derivative_form, - residual_form, pde_problem.bcs_list)

        if linear_solver=='fenics_direct':

            rhs_ = df.Function(state_function.function_space())
            dR = df.Function(state_function.function_space())

            rhs_.vector().set_local(d_outputs[state_name])

            for bc in pde_problem.bcs_list:
                bc.apply(A)
            Am = df.as_backend_type(A).mat()
            ATm = Am.transpose()
            AT =  df.PETScMatrix(ATm)

            df.solve(AT,dR.vector(),rhs_.vector()) 
            d_residuals[state_name] =  dR.vector().get_local()

        elif linear_solver=='scipy_splu':
            for bc in pde_problem.bcs_list:
                bc.apply(A)
            Am = df.as_backend_type(A).mat()
            ATm = Am.transpose()
            ATm_csr = csr_matrix(ATm.getValuesCSR()[::-1], shape=Am.size)
            lu = splu(ATm_csr.tocsc())
            d_residuals[state_name] = lu.solve(d_outputs[state_name],trans='T')


        elif linear_solver=='fenics_Krylov':

            rhs_ = df.Function(state_function.function_space())
            dR = df.Function(state_function.function_space())

            rhs_.vector().set_local(d_outputs[state_name])

            for bc in pde_problem.bcs_list:
                bc.apply(A)
            Am = df.as_backend_type(A).mat()
            ATm = Am.transpose()
            AT =  df.PETScMatrix(ATm)

            solver = df.KrylovSolver('gmres', 'ilu')
            prm = solver.parameters          
            prm["maximum_iterations"]=1000000
            prm["divergence_limit"] = 1e2
            solver.solve(AT,dR.vector(),rhs_.vector())

            d_residuals[state_name] =  dR.vector().get_local()

        elif linear_solver=='petsc_gmres_ilu':
            ksp = PETSc.KSP().create() 
            ksp.setType(PETSc.KSP.Type.GMRES)
            ksp.setTolerances(rtol=5e-11)

            for bc in pde_problem.bcs_list:
                bc.apply(A)
            Am = df.as_backend_type(A).mat()

            ksp.setOperators(Am)

            ksp.setFromOptions()
            pc = ksp.getPC()
            pc.setType("ilu")

            size = state_function.function_space().dim()

            dR = PETSc.Vec().create()
            dR.setSizes(size)
            dR.setType('seq')
            dR.setValues(range(size), d_residuals[state_name])
            dR.setUp()

            du = PETSc.Vec().create()
            du.setSizes(size)
            du.setType('seq')
            du.setValues(range(size), d_outputs[state_name])
            du.setUp()

            if mode == 'fwd':
                ksp.solve(dR,du)
                d_outputs[state_name] = du.getValues(range(size))
            else:
                ksp.solveTranspose(du,dR)
                d_residuals[state_name] = dR.getValues(range(size))
                






if __name__ == '__main__':   
    pass
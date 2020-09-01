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
            values=['fenics_direct', 'scipy_splu', 'petsc_gmres_ilu'],
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

        derivative_form = df.derivative(residual_form, state_function)

        df.set_log_active(True)
        df.solve(residual_form==0, state_function, bcs=pde_problem.bcs_list, J=derivative_form,
              solver_parameters={"newton_solver":{"maximum_iterations":1, "error_on_nonconvergence":False}})

        self.L = -residual_form

        outputs[state_name] = state_function.vector().get_local()

    def linearize(self, inputs, outputs, partials):

        J = derivative(self.fea.pdeRes(self.fea.u, self.fea.v, self.fea.rho_e), self.fea.u)

        res = - self.fea.pdeRes(self.fea.u, self.fea.v, self.fea.rho_e)
        self.A, b = assemble_system(J, res, [self.fea.bc_1,self.fea.bc_1])

        self.fea.rho_e.vector().set_local(inputs['rho_e'])
        self.fea.u.vector().set_local(outputs['displacements'])
        
        dR_du_coo, dR_dC_coo = self.fea.compute_derivative(self.fea.u, self.fea.v, self.fea.rho_e)

        self.dR_du_sparse = as_backend_type(self.A).mat()
        
        partials['displacements','rho_e'] = dR_dC_coo.data
        partials['displacements','displacements'] = dR_du_coo.data
    

    # @profile
    def solve_linear(self, d_outputs, d_residuals, mode):

        option = self.options['option']

        dR_du_sparse = self.dR_du_sparse

        
        if option==1:

            ksp = PETSc.KSP().create() 

            ksp.setType(PETSc.KSP.Type.GMRES)
            ksp.setTolerances(rtol=5e-11)
            ksp.setOperators(dR_du_sparse)
            ksp.setFromOptions()

            pc = ksp.getPC()
            pc.setType("ilu")

            size = len(self.fea.V.dofmap().dofs())

            dR = PETSc.Vec().create()
            dR.setSizes(size)
            dR.setType('seq')
            dR.setValues(range(size), d_residuals['displacements'])
            dR.setUp()

            du = PETSc.Vec().create()
            du.setSizes(size)
            du.setType('seq')
            du.setValues(range(size), d_outputs['displacements'])
            du.setUp()

            if mode == 'fwd':
                ksp.solve(dR,du)
                d_outputs['displacements'] = du.getValues(range(size))
            else:
                ksp.solveTranspose(du,dR)
                d_residuals['displacements'] = dR.getValues(range(size))
                
        elif option==2:

            rhs_ = Function(self.fea.V)
            dR = Function(self.fea.V)

            rhs_.vector().set_local(d_outputs['displacements'])

            A = self.A
            for bc in [self.fea.bc_1,self.fea.bc_2]:
                bc.apply(A)
            Am = as_backend_type(A).mat()

            ATm = Am.transpose()
            AT =  PETScMatrix(ATm)

            solve(AT,dR.vector(),rhs_.vector()) # cannot directly use fea.u here, the update for the solution is not compatible
            d_residuals['displacements'] =  dR.vector().get_local()


        elif option==3:
            A = self.A
            for bc in [self.fea.bc_1,self.fea.bc_2]:
                bc.apply(A)
            Am = as_backend_type(A).mat()
            ATm = Am.transpose()
            ATm_csr = csr_matrix(ATm.getValuesCSR()[::-1], shape=Am.size)
            lu = splu(ATm_csr.tocsc())
            d_residuals['displacements'] = lu.solve(d_outputs['displacements'],trans='T')


        elif option==4:

            rhs_ = Function(self.fea.V)
            dR = Function(self.fea.V)

            rhs_.vector().set_local(d_outputs['displacements'])

            A = self.A
            for bc in [self.fea.bc_1,self.fea.bc_2]:
                bc.apply(A)
            Am = as_backend_type(A).mat()

            ATm = Am.transpose()
            AT =  PETScMatrix(ATm)
            set_log_active(True)

            solver = KrylovSolver('gmres', 'ilu')
            prm = solver.parameters          
            prm["maximum_iterations"]=1000000
            prm["divergence_limit"] = 1e2
            # info(parameters,True)
            solver.solve(AT,dR.vector(),rhs_.vector())


            d_residuals['displacements'] =  dR.vector().get_local()


if __name__ == '__main__':    
    from set_fea_full import set_fea as fea

    from openmdao.api import Problem, Group
    from openmdao.api import IndepVarComp

    num_elements = 32
    fea = set_fea(num_elements=num_elements)
    group = Group()
    prob = Problem()

    comp = IndepVarComp()
    comp.add_output('rho_e', shape=fea.num_var, val=np.random.random((fea.num_var))+1e-3)
    group.add_subsystem('input', comp, promotes=['*'])

    comp = StatesComp(fea=fea)
    group.add_subsystem('StatesComp', comp, promotes=['*'])
  
    prob.model = group
    prob.setup()
    prob.run_model()
    prob.check_partials(compact_print=True)
    File('elasticity2/displacement.pvd') << fea.u


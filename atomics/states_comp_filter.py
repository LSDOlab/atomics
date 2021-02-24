from __future__ import division
import dolfin as df
from petsc4py import PETSc

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from petsc4py import PETSc

import openmdao.api as om
from openmdao.api import Problem
from matplotlib import pyplot as plt

import ufl
from atomics.pdes.variational_filter import get_residual_form_variational_filter
# import cProfile, pstats, io

class StatesFilterComp(om.ImplicitComponent):
    """
    The implicit component that wraps a FEniCS filter that soomthing 
    the density basic on the size of the elements

    Parameters
    ----------
    density_unfiltered[self.fea.num_var] : numpy array
        unfiltered density
    Returns
    -------
    density[self.fea.num_var] : numpy array
        filtered density
    """

    def initialize(self):
        self.options.declare('residual')
        self.options.declare('function_space')

        self.options.declare('option', default=2)

    
    def setup(self):

        self.fea = self.options['residual']
        self.function_space = self.options['function_space']
        self.v = df.TestFunction(self.options['function_space'])

        self.add_input('density_unfiltered', shape=self.function_space.dim() )
        self.add_output('density', shape=self.function_space.dim() )
        self.filtered_density = df.Function( self.function_space)
        self.unfiltered_density = df.Function( self.function_space)

        dR_dstate = self.compute_derivative('dR_dstate', self.filtered_density)
        dR_dinput = self.compute_derivative('dR_dinput', self.unfiltered_density)
 
        self.declare_partials('density', 'density_unfiltered', rows=dR_dinput.row, cols=dR_dinput.col)
        self.declare_partials('density', 'density', rows=dR_dstate.row, cols=dR_dstate.col)

    def compute_derivative(self, arg_name, arg_function):
        
        residual_form = self.options['residual'](self.unfiltered_density, 
                                                    self.filtered_density)
        derivative_form = df.derivative(residual_form, arg_function)
        derivative_petsc_sparse = df.as_backend_type(df.assemble(derivative_form)).mat()
        derivative_csr = csr_matrix(derivative_petsc_sparse.getValuesCSR()[::-1], shape=derivative_petsc_sparse.size)

        return derivative_csr.tocoo()


    def apply_nonlinear(self, inputs, outputs, residuals):
        self.unfiltered_density.vector().set_local(inputs['density_unfiltered'])
        self.filtered_density.vector().set_local(outputs['density'])

        residual_form = self.options['residual'](self.unfiltered_density, 
                                                    self.filtered_density)
        residuals['density'] = df.assemble(residual_form).get_local()

    
    def solve_nonlinear(self, inputs, outputs,):

        self.unfiltered_density.vector().set_local(inputs['density_unfiltered'])
        self.filtered_density.vector().set_local(outputs['density'])

        residual_form = self.options['residual'](self.unfiltered_density, 
                                                    self.filtered_density)
        J = df.derivative(residual_form, self.filtered_density)

        df.set_log_active(False)
        df.set_log_active(True)


        df.solve(residual_form==0, self.filtered_density, J=J,
              solver_parameters={"newton_solver":{"maximum_iterations":1, "error_on_nonconvergence":False}})

        outputs['density'] = self.filtered_density.vector().get_local()

    
    def linearize(self, inputs, outputs, partials):
        # print('linearize')
        residual_form = self.options['residual'](self.unfiltered_density, 
                                                    self.filtered_density)
        J = df.derivative(residual_form, self.filtered_density)

        self.A, b = df.assemble_system(J, - residual_form)

        self.unfiltered_density.vector().set_local(inputs['density_unfiltered'])
        self.filtered_density.vector().set_local(outputs['density'])
        
        dR_dstate = self.compute_derivative('dR_dstate', self.filtered_density)
        dR_dinput = self.compute_derivative('dR_dinput', self.unfiltered_density)

        self.dR_du_sparse = df.as_backend_type(self.A).mat()
        
        partials['density','density'] = dR_dstate.data
        partials['density','density_unfiltered'] = dR_dinput.data

    def solve_linear(self, d_outputs, d_residuals, mode):

        option = self.options['option']

        dR_du_sparse = self.dR_du_sparse

        
        if option==1:

            ksp = PETSc.KSP().create() 

            ksp.setType(PETSc.KSP.Type.GMRES)
            ksp.setTolerances(rtol=5e-14)
            ksp.setOperators(dR_du_sparse)
            ksp.setFromOptions()

            pc = ksp.getPC()
            pc.setType("ilu")

            size = len(self.fea.VC.dofmap().dofs())

            dR = PETSc.Vec().create()
            dR.setSizes(size)
            dR.setType('seq')
            dR.setValues(range(size), d_residuals['density'])
            dR.setUp()

            du = PETSc.Vec().create()
            du.setSizes(size)
            du.setType('seq')
            du.setValues(range(size), d_outputs['density'])
            du.setUp()

            if mode == 'fwd':
                ksp.solve(dR,du)
                d_outputs['density'] = du.getValues(range(size))
            else:
                ksp.solveTranspose(du,dR)
                d_residuals['density'] = dR.getValues(range(size))
                print('d_residual[density]', d_residuals['density'])
        elif option==2:
            # print('option 2')

            rhs_ = df.Function(self.function_space)
            dR = df.Function(self.function_space)

            rhs_.vector().set_local(d_outputs['density'])

            A = self.A
            Am = df.as_backend_type(A).mat()

            ATm = Am.transpose()
            AT =  df.PETScMatrix(ATm)

            df.solve(AT,dR.vector(),rhs_.vector()) # cannot directly use fea.u here, the update for the solution is not compatible
            d_residuals['density'] =  dR.vector().get_local()


        elif option==3:
            A = self.A

            Am = df.as_backend_type(A).mat()
            ATm = Am.transpose()
            ATm_csr = csr_matrix(ATm.getValuesCSR()[::-1], shape=Am.size)
            lu = splu(ATm_csr.tocsc())
            d_residuals['density'] = lu.solve(d_outputs['density'],trans='T')


        elif option==4:

            rhs_ = df.Function(self.function_space)
            dR = df.Function(self.function_space)

            rhs_.vector().set_local(d_outputs['density'])

            A = self.A
            Am = df.as_backend_type(A).mat()

            ATm = Am.transpose()
            AT =  df.PETScMatrix(ATm)
            df.set_log_active(True)

            solver = df.KrylovSolver('gmres', 'ilu')
            prm = solver.parameters          
            prm["maximum_iterations"]=1000000
            prm["divergence_limit"] = 1e2
            # info(parameters,True)
            solver.solve(AT,dR.vector(),rhs_.vector())


            d_residuals['displacements'] =  dR.vector().get_local()


if __name__ == '__main__':    

    from openmdao.api import Problem, Group
    from openmdao.api import IndepVarComp
    from atomics.pdes.variational_filter import get_residual_form_variational_filter

    mesh = df.UnitSquareMesh(10,10)
    FS = df.FunctionSpace(mesh, 'DG', 0)
  
    group = Group()
    prob = Problem()

    comp = IndepVarComp()
    comp.add_output('density_unfiltered', shape=FS.dim(), val=np.random.random((FS.dim()))+1e-3)
    group.add_subsystem('input', comp, promotes=['*'])

    comp = StatesFilterComp(residual=get_residual_form_variational_filter, function_space=FS)
    group.add_subsystem('StatesComp', comp, promotes=['*'])
  
    prob.model = group
    prob.setup()
    prob.run_model()
    prob.check_partials(compact_print=True)
    # df.File('unfiltered_density.pvd') << fea.unfiltered_density

    # df.File('filtered_density.pvd') << fea.filtered_density

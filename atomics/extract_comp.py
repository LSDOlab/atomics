import numpy as np

from openmdao.core.explicitcomponent import ExplicitComponent


class ExtractComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('in_name', types=str)
        self.options.declare('out_name', types=str)
        self.options.declare('in_shape', types=int)
        self.options.declare('partial_dof', types=np.ndarray)

    def setup(self):
        in_name = self.options['in_name']
        out_name = self.options['out_name']
        partial_dof = self.options['partial_dof']
        in_shape = self.options['in_shape']
        out_shape = partial_dof.size

        self.add_input(in_name, shape=in_shape)
        self.add_output(out_name, shape=out_shape)

        row_indices = np.arange(partial_dof.size)
        col_indices = partial_dof

        self.declare_partials(of=out_name, wrt=in_name, rows=row_indices, cols=col_indices, val=1.)

    def compute(self, inputs, outputs):
        in_name = self.options['in_name']
        out_name = self.options['out_name']
        partial_dof = self.options['partial_dof']

        outputs[out_name] = inputs[in_name][partial_dof.astype(int)]


if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp
    from ksconstraints_comp import KSConstraintsComp
    in_data = np.loadtxt('test_t.out')
    partial_dof = np.loadtxt('ind.out')
    shape = (10,)
    axis = 0

    prob = Problem()

    comp = IndepVarComp()
    comp.add_output('x', val=in_data)
    prob.model.add_subsystem('ivc', comp, promotes=['*'])

    comp = ExtractComp(
        in_name='x',
        out_name='y',
        in_shape=in_data.size,
        partial_dof=partial_dof,
    )
    prob.model.add_subsystem('ExtractComp', comp, promotes=['*'])

    comp = KSConstraintsComp(
        in_name='y',
        out_name='y_max',
        shape=(partial_dof.size,),
        axis=0,
        rho=50.,
    )
    prob.model.add_subsystem('KSConstraintsComp', comp, promotes=['*'])

    prob.setup()
    prob.run_model()
    prob.check_partials(compact_print=True)
    print(prob['x'], 'x')
    print(prob['y'], 'y')
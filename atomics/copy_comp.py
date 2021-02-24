import numpy as np

from openmdao.core.explicitcomponent import ExplicitComponent


class Copycomp(ExplicitComponent):

    def initialize(self):
        self.options.declare('in_name', types=str)
        self.options.declare('out_name', types=str)
        self.options.declare('in_shape', types=int)
        self.options.declare('num_copies', types=int)

    def setup(self):
        in_name = self.options['in_name']
        out_name = self.options['out_name']
        in_shape = self.options['in_shape']
        num_copies = self.options['num_copies']
        out_shape = num_copies * (in_shape)

        self.add_input(in_name, shape=in_shape)
        self.add_output(out_name, shape=out_shape)
 
        row_indices = np.arange(in_shape * num_copies )
        col_indices = np.tile(np.arange(in_shape), (num_copies))

        self.declare_partials(of=out_name, wrt=in_name, rows=row_indices, cols=col_indices, val=1.)

    def compute(self, inputs, outputs):
        in_name = self.options['in_name']
        out_name = self.options['out_name']
        num_copies = self.options['num_copies']
        
        outputs[out_name] = np.tile(inputs[in_name], num_copies)

if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp

    in_shape = 5
    in_data = np.random.random(in_shape)
    num_copies = 3


    prob = Problem()

    comp = IndepVarComp()
    comp.add_output('x', val=in_data)
    prob.model.add_subsystem('ivc', comp, promotes=['*'])

    comp = Copycomp(
        in_name='x',
        out_name='y',
        in_shape=in_shape,
        num_copies = num_copies,
    )
    prob.model.add_subsystem('Copycomp', comp, promotes=['*'])


    prob.setup()
    prob.run_model()
    prob.check_partials(compact_print=True)
    prob.check_partials(compact_print=False)
    print(prob['x'], 'x')
    print(prob['y'], 'y')
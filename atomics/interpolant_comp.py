import numpy as np

from openmdao.core.explicitcomponent import ExplicitComponent


class InterpolantComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('in_name', types=str)
        self.options.declare('out_name', types=str)
        self.options.declare('in_shape', types=int)
        self.options.declare('num_pts', types=int)

    def setup(self):
        in_name = self.options['in_name']
        out_name = self.options['out_name']
        in_shape = self.options['in_shape']
        num_pts = self.options['num_pts']
        out_shape = self.options['in_shape'] * (num_pts-2)

        self.add_input(in_name, shape=in_shape)
        self.add_output(out_name, shape=out_shape)

        ele_layer = int(in_shape/2)
        ele_list = np.arange(ele_layer)
        val = np.linspace(0, 1., num_pts)   
        row_indices_0 = np.arange(ele_layer * num_pts )
        col_indices_0 = np.tile(ele_list, (num_pts))
        val_1 = np.outer(np.linspace(0,1, num_pts), np.ones(ele_layer)).flatten()
        row_indices_1 = np.arange(ele_layer * num_pts )
        col_indices_1 = (np.tile(ele_list, (num_pts,1)) + ele_layer).flatten()
        val_0 = 1 - val_1

        row_indices = np.concatenate((row_indices_0, row_indices_1))
        col_indices = np.concatenate((col_indices_0, col_indices_1))
        val = np.concatenate((val_0, val_1))

        self.declare_partials(of=out_name, wrt=in_name, rows=row_indices, cols=col_indices, val=val)

    def compute(self, inputs, outputs):
        in_name = self.options['in_name']
        out_name = self.options['out_name']
        in_shape = self.options['in_shape']
        num_pts = self.options['num_pts']
        out_shape = self.options['in_shape'] * (num_pts-2)

        val = np.linspace(0, 1., num_pts)   
        out = np.zeros((num_pts, int(in_shape/2)))
        x_0 = inputs[in_name][np.arange(int(in_shape/2))]
        x_1 = inputs[in_name][np.arange(in_shape-int(in_shape/2), in_shape)]

        for i in np.arange(num_pts):
            out[i, :] = val[i] * x_1 + (1-val[i]) * x_0
        
        outputs[out_name] = out.flatten()

if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp

    in_shape = 4
    in_data = np.random.random(in_shape)
    num_pts = 4


    prob = Problem()

    comp = IndepVarComp()
    comp.add_output('x', val=in_data)
    prob.model.add_subsystem('ivc', comp, promotes=['*'])

    comp = InterpolantComp(
        in_name='x',
        out_name='y',
        in_shape=in_shape,
        num_pts = num_pts,
    )
    prob.model.add_subsystem('InterpolantComp', comp, promotes=['*'])


    prob.setup()
    prob.run_model()
    prob.check_partials(compact_print=True)
    prob.check_partials(compact_print=False)
    print(prob['x'], 'x')
    print(prob['y'], 'y')
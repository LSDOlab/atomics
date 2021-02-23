import numpy as np

from openmdao.core.explicitcomponent import ExplicitComponent


class SymmericAnglecomp(ExplicitComponent):

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

        self.L = int(np.sqrt(in_shape/2))
        # print(self.L)
        self.add_input(in_name, shape=in_shape)
        self.add_output(out_name, shape=out_shape)

        col_temp = np.arange(self.L)
        col_temp_1 = np.hstack((col_temp, col_temp[::-1]))
        # print('col_temp_1', col_temp_1)
        col_temp_mtx_half_temp = np.tile(col_temp_1, (self.L,1))+ np.arange(self.L).reshape(self.L,1)*self.L
        # print('col_temp_mtx_half_temp', col_temp_mtx_half_temp)
        col_temp_mtx_half = np.vstack((col_temp_mtx_half_temp, col_temp_mtx_half_temp[::-1]))
        # print('col_temp_mtx_half', col_temp_mtx_half)

        col_temp_mtx = np.vstack((col_temp_mtx_half, col_temp_mtx_half+int(in_shape/2)))
        # print('col_temp_mtx', col_temp_mtx)

        row_indices = np.arange(in_shape * num_copies )
        col_indices = col_temp_mtx.flatten()

        val_temp = np.outer(np.array([1,-1]), np.ones(self.L))
        val_half_temp = np.outer(np.ones(self.L), val_temp.flatten())
        val_half = np.concatenate((val_half_temp.flatten(), (val_half_temp.flatten())[::-1]))
        self.declare_partials(of=out_name, wrt=in_name, rows=row_indices, cols=col_indices, val=np.tile(val_half, 2))

    def compute(self, inputs, outputs):
        in_name = self.options['in_name']
        out_name = self.options['out_name']
        num_copies = self.options['num_copies']

        in_mat = inputs[in_name].reshape(2, self.L, self.L)
        # print('in',  inputs[in_name])

        # print('in_mat', in_mat.shape)
        # print('in_mat', in_mat)

        out_mat_temp_h1 = np.concatenate((in_mat, np.pi - in_mat[:,:,::-1]), axis=2)
        # print('out_mat_temp', out_mat_temp_h1.shape)
        # print('out_mat_temp', out_mat_temp_h1)
        out_mat_temp_h2 = np.concatenate((np.pi - in_mat[:,::-1,:], in_mat[:,::-1,::-1]), axis=2)

        out_mat = np.hstack((out_mat_temp_h1, out_mat_temp_h2))
        # print('out_shape', out_mat.shape)
        # print('out_shape', out_mat)
        outputs[out_name] = out_mat.flatten()

if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp

    in_shape = 8
    in_data = np.random.random(in_shape)
    num_copies = 4


    prob = Problem()

    comp = IndepVarComp()
    comp.add_output('x', val=in_data)
    prob.model.add_subsystem('ivc', comp, promotes=['*'])

    comp = SymmericRhocomp(
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
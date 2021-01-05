import numpy as np

from openmdao.core.explicitcomponent import ExplicitComponent


class KSConstraintsComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('shape', types=tuple)
        self.options.declare('axis', types=int)
        self.options.declare('out_name', types=str)
        self.options.declare('in_name', types=str)
        self.options.declare('rho', 50.0, desc="Constraint Aggregation Factor.")

    def setup(self):
        in_name = self.options['in_name']
        out_name = self.options['out_name']
        shape = self.options['shape']
        axis = self.options['axis']

        self.total_rank = len(self.options['shape'])
        # print('self.total_rank is:', self.total_rank)
        if self.total_rank == 1:
            shape = shape+(1,)
            self.total_rank = 2

        if self.options['axis'] < 0:
            self.options['axis'] += self.total_rank



        in_shape = tuple(shape)
        out_shape = shape[:axis] + shape[axis+1:]
        # print('in_shape', in_shape)
        # print('out_shape', out_shape)

        self.add_input(in_name, shape=in_shape)
        self.add_output(out_name, shape=out_shape)

        out_indices = np.arange(np.prod(out_shape)).reshape(out_shape)
        in_indices = np.arange(np.prod(in_shape)).reshape(in_shape)

        alphabet = 'abcdefghijkl'

        self.einsum_str = einsum_str = '{},{}->{}'.format(
            alphabet[:axis] + alphabet[axis+1:self.total_rank],
            alphabet[axis],
            alphabet[:self.total_rank],
        )
        self.ones = ones = np.ones(shape[axis])

        rows = np.einsum(
            einsum_str,
            out_indices,
            ones.astype(int),
        ).flatten()
        cols = in_indices.flatten()

        self.declare_partials(of=out_name, wrt=in_name, rows=rows, cols=cols)

    def compute(self, inputs, outputs):

        in_name = self.options['in_name']
        out_name = self.options['out_name']
        shape = self.options['shape']
        axis = self.options['axis']
        rho = self.options['rho']
        if self.total_rank == 1:
            shape = shape+(1,)
            inputs[in_name] = inputs[in_name].reshape(-1,1)
        con_val = inputs[in_name]

        g_max = np.max(con_val, axis=axis)
        g_diff = con_val - np.einsum(
            self.einsum_str,
            g_max,
            self.ones,
        )
        exponents = np.exp(rho * g_diff)
        summation = np.sum(exponents, axis=axis)
        result = g_max + 1.0 / rho * np.log(summation)
        outputs[out_name] = result
        # print('The max of the result is:', result)

        dsum_dg = rho * exponents
        dKS_dsum = 1.0 / (rho * np.einsum(
            self.einsum_str,
            summation,
            self.ones,
        ))
        dKS_dg = dKS_dsum * dsum_dg

        self.dKS_dg = dKS_dg

    def compute_partials(self, inputs, partials):
        in_name = self.options['in_name']
        out_name = self.options['out_name']
        partials[out_name, in_name] = self.dKS_dg.flatten()


if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp

    shape = (10,)
    axis = 0

    prob = Problem()

    comp = IndepVarComp()
    comp.add_output('x', val=np.random.rand(*shape))
    prob.model.add_subsystem('ivc', comp, promotes=['*'])

    comp = KSConstraintsComp(
        in_name='x',
        out_name='y',
        shape=shape,
        axis=axis,
        rho=100.,
    )
    prob.model.add_subsystem('comp', comp, promotes=['*'])

    prob.setup()
    prob.run_model()
    prob.check_partials(compact_print=True)
    print(prob['x'], 'x')
    print(prob['y'], 'y')
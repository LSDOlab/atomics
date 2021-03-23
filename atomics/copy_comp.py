import numpy as np

from openmdao.core.explicitcomponent import ExplicitComponent


class CopyComp(ExplicitComponent):
    """
    CopyComp is a stock component implemented in OpenMDAO
    to make copies of a variable and concatenate with itself.
    Parameters
    ----------
    in_name : str
        the name of the variable we want to copy
    out_name : str
        the output variable name
    in_shape : int
        the shape of the input variable
    num_copies : int
        number of copies
    Returns
    -------
    outputs[out_name] : numpy array
        the inputs[in_name] vector concatenated with its copies
    """

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

import numpy as np

from openmdao.core.explicitcomponent import ExplicitComponent


class ExtractComp(ExplicitComponent):
    """
    ExtractComp is a stock component implemented in OpenMDAO
    to extract some data from a vector
    Parameters
    ----------
    in_name : str
        the name of the variable to extract from
    out_name : str
        the output variable name
    in_shape : int
        the shape of the input variable
    partial_dof : numpy array
        The indices of the extracted data in inputs[in_name] vector
    Returns
    -------
    outputs[out_name] : numpy array
        the inputs[in_name] the vector of the extracted data
    """

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

import numpy as np

from openmdao.core.explicitcomponent import ExplicitComponent


class SymmericAnglecomp(ExplicitComponent):
    """
    SymmericAnglecomp is a stock component implemented in OpenMDAO
    used in atomics.examples.case_studies.case_3_LCE_shape_matching
    to get a mirrored angle profile from its top left quarter. 
    Parameters
    ----------
    in_name : str
        the name of the input variable (angles on the top left)
    out_name : str
        the output variable name
    in_shape : int
        the shape of the input variable
    num_copies : 4
        mirror and flip
    Returns
    -------
    outputs[out_name] : numpy array
        The flattened whole symmetric matrix
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

        self.L = int(np.sqrt(in_shape/2))
        self.add_input(in_name, shape=in_shape)
        self.add_output(out_name, shape=out_shape)

        col_temp = np.arange(self.L)
        col_temp_1 = np.hstack((col_temp, col_temp[::-1]))
        col_temp_mtx_half_temp = np.tile(col_temp_1, (self.L,1))+ np.arange(self.L).reshape(self.L,1)*self.L
        col_temp_mtx_half = np.vstack((col_temp_mtx_half_temp, col_temp_mtx_half_temp[::-1]))
        col_temp_mtx = np.vstack((col_temp_mtx_half, col_temp_mtx_half+int(in_shape/2)))

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
        out_mat_temp_h1 = np.concatenate((in_mat, np.pi - in_mat[:,:,::-1]), axis=2)
        out_mat_temp_h2 = np.concatenate((np.pi - in_mat[:,::-1,:], in_mat[:,::-1,::-1]), axis=2)

        out_mat = np.hstack((out_mat_temp_h1, out_mat_temp_h2))
        outputs[out_name] = out_mat.flatten()

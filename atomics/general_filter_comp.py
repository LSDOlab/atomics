import dolfin as df
import numpy as np
import scipy.sparse 
from scipy import spatial


from openmdao.api import ExplicitComponent


class GeneralFilterComp(ExplicitComponent):
    def initialize(self):
        self.options.declare('density_function_space')

   
    def setup(self):
        density_function_space = self.options['density_function_space']
        NUM_ELEMENTS = density_function_space.dim()


        self.add_input('density_unfiltered', shape=NUM_ELEMENTS)
        self.add_output('density', shape=NUM_ELEMENTS)

        scalar_output_center_coord = density_function_space.tabulate_dof_coordinates()

        ''' Todo: use average element size to define the radius '''

        # filter radius defined as two times the average size

        mesh_size_max = density_function_space.mesh().hmax()
        mesh_size_min = density_function_space.mesh().hmin()

        radius = 2 * ((mesh_size_max + mesh_size_min) /2) 
        # 1.414 is because the hmax is defined as the 
        # greatest distance between any two vertices (sqrt(2))

        weight_ij = []
        col = []
        row = []

        for i in range(NUM_ELEMENTS):
            current_point = scalar_output_center_coord[i,:]
            points_selection = scalar_output_center_coord
            tree = spatial.cKDTree(points_selection)
            idx = tree.query_ball_point(list(current_point), radius)
            nearest_points = points_selection[idx]
            
            weight_sum = sum(radius - np.linalg.norm(current_point - nearest_points,axis = 1))

            for j in idx:
                weight = ( radius - np.linalg.norm(current_point - points_selection[j]))/weight_sum
                row.append(i)
                col.append(j)
                weight_ij.append(weight)       
       
        self.weight_mtx = scipy.sparse.csr_matrix((weight_ij, (row, col)), shape=(NUM_ELEMENTS, NUM_ELEMENTS))

        self.declare_partials('density', 'density_unfiltered',rows=np.array(row), cols=np.array(col),val=np.array(weight_ij))

    def compute(self, inputs, outputs):
        outputs['density'] = self.weight_mtx.dot(inputs['density_unfiltered'])


# if __name__ == '__main__':
#     import dolfin as df
#     from openmdao.api import Problem, Group
#     from openmdao.api import IndepVarComp
#     group = Group()

#     NUM_ELEMENTS_X = 30
#     NUM_ELEMENTS_Y = 20
#     LENGTH_X = .06
#     LENGTH_Y = .04

#     mesh = df.RectangleMesh.create(
#         [df.Point(0.0, 0.0), df.Point(LENGTH_X, LENGTH_Y)],
#         [NUM_ELEMENTS_X, NUM_ELEMENTS_Y],
#         df.CellType.Type.quadrilateral,
#     )

#     density_function_space = df.FunctionSpace(mesh, 'DG', 0)
#     scaler_value = np.zeros((NUM_ELEMENTS_X*NUM_ELEMENTS_Y))
#     scaler_value[300:340] = 1.
#     comp = IndepVarComp()
#     comp.add_output('density_unfiltered', shape=600, val=scaler_value)
#     group.add_subsystem('input', comp, promotes=['*'])

#     comp = GeneralFilterComp(density_function_space=density_function_space)
#     group.add_subsystem('GeneralFilterComp', comp, promotes=['*'])
#     prob = Problem()
#     prob.model = group
#     prob.setup()
#     prob.run_model()
#     prob.check_partials(compact_print=True)
#     prob.check_partials(compact_print=False)

#     import matplotlib.pyplot as plt

#     fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharex=True)
#     ax0.set_title('density_unfiltered')
#     ax0.imshow(prob['density_unfiltered'].reshape(NUM_ELEMENTS_X, NUM_ELEMENTS_Y), cmap='hot', interpolation='nearest')
#     ax1.set_title('density_filtered')
#     ax1.imshow(prob['density'].reshape(NUM_ELEMENTS_X, NUM_ELEMENTS_Y), cmap='hot', interpolation='nearest')
#     plt.show()



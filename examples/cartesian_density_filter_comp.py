import numpy as np
import scipy.sparse 

from openmdao.api import ExplicitComponent


class CartesianDensityFilterComp(ExplicitComponent):

    """
    Compute the filtered density using the simplest linear filter 
    (<= 8 element for 2d; <= 26 elements for 3d).
    Parameters
    ----------
    density_unfiltered[num_dvs] : numpy array
        density for each element before filtering
    Returns
    -------
    density[num_dvs] : numpy array
        density for each element after filtering
    """


    def initialize(self):
        self.options.declare('length_x', types=(int, float),default=160. )#required=True)
        self.options.declare('length_y', types=(int, float), default= 80.)#required=True)
        self.options.declare('num_nodes_x', types=int, default=81)#required=True)
        self.options.declare('num_nodes_y', types=int, default=41)#required=True)
        self.options.declare('num_dvs', types=int, default=3200 )#required=True)
        self.options.declare('radius', types=float, default=4.)#required=True)
        
    def setup(self):
        num_dvs = self.options['num_dvs']
        length_x = self.options['length_x']
        length_y = self.options['length_y']
        num_nodes_x = self.options['num_nodes_x']
        num_nodes_y = self.options['num_nodes_y']
        radius = self.options['radius']

        self.add_input('density_unfiltered', shape=num_dvs)
        self.add_output('density', shape=num_dvs)

        num_elem_x = num_nodes_x - 1
        num_elem_y = num_nodes_y - 1
        num_elem = num_elem_x * num_elem_y

        lx = length_x / float(num_elem_x)
        ly = length_y / float(num_elem_y)

        cnt = 0

        row = []
        col = []
        wij = []
        
        cnt = 0
        ''' NOTE: element is ordered by x-y sequence '''
        for iy in range(0, num_elem_y):
            for ix in range(0, num_elem_x):
                eid_i = iy * num_elem_x + ix
                # centers
                x_i = (ix + 0.5) * lx
                y_i = (iy + 0.5) * ly
                dsum = 0
                dij_list = []
                negh_list = []
                num_neighbor = 0

                for jy in range(max(0, iy - 3), min(num_elem_y, iy + 3)):
                    for jx in range(max(0, ix - 3), min(num_elem_x, ix + 3)):
                        eid_j = jy * num_elem_x + jx
                        x_j = (jx + 0.5) * lx
                        y_j = (jy + 0.5) * ly

                        dij = ((x_i-x_j)**2 + (y_i-y_j)**2)**0.5
                        if dij < radius:
                            num_neighbor += 1
                            dij_list.append(dij)
                            negh_list.append(eid_j)
                            dsum += dij
                
                for tt in range(0, num_neighbor):
                    row.append(eid_i)
                    col.append(negh_list[tt])
                    wij.append(dij_list[tt]/dsum)
                    cnt += 1

        row = np.array(row)
        col = np.array(col)
        wij = np.array(wij)
        self.mtx = scipy.sparse.csr_matrix((wij, (row, col)), shape=(num_elem, num_elem))
                
        self.declare_partials('density', 'density_unfiltered', rows = row, cols = col, val = wij)

    def compute(self, inputs, outputs):
        outputs['density'] = self.mtx.dot(inputs['density_unfiltered'])



if __name__ == '__main__':
    from openmdao.api import Problem, Group
    from openmdao.api import IndepVarComp
    from penalty_comp import Penaltycomp

    num_elements = 3200
    group = Group()
    intial_density = np.zeros((num_elements))
    intial_density[1201:1601] = np.arange(400)
    intial_density[1601:2000] = np.arange(400,1,-1)

    comp = IndepVarComp()
    comp.add_output('density_unfiltered', shape=num_elements, val=intial_density)
    group.add_subsystem('input', comp, promotes=['*'])

    comp = DensityFilterComp()
    group.add_subsystem('DensityFilterComp', comp, promotes=['*'])
    prob = Problem()
    prob.model = group
    prob.setup()
    prob.run_model()
    prob.check_partials(compact_print=True)
    prob.check_partials(compact_print=False)
    import matplotlib.pyplot as plt
    import numpy as np
    E_org = prob['density_unfiltered'].reshape(80, 40)
    E_filered = prob['density'].reshape(80, 40)
    plt.figure(1)
    plt.imshow(E_org, cmap='hot', interpolation='nearest')
    plt.title('origial stiffness (one slice in xy plane)')

    plt.figure(2)
    plt.imshow(E_filered, cmap='hot', interpolation='nearest')
    plt.title('filtered stiffness')
    plt.title('filtered stiffness (one slice in xy plane)')

    plt.figure(3)
    plt.plot(prob['density'])
    plt.title('Stiffness(all element---filtered)')

    plt.figure(4)
    plt.plot(prob['density_unfiltered'])   
    plt.title('Stiffness(all element---original)')
 
    plt.show()

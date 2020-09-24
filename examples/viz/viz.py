from lsdo_viz.api import BaseViz, Frame
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

num_elements = 41
import numpy as np

# from atomics.visualization.atomics_viz import AtomicsViz

class AtomicsViz(BaseViz):

    def setup(self):
        # self.use_latex_fonts()

        self.frame_name_format = 'output_{}'

        self.add_frame(
            Frame(
                height_in=6,
                width_in=7,
                nrows=3,
                ncols=1,
                wspace=0.55,
                hspace=0.55,
            ), 1)

    def add_paraview_plot(self, paraview_script_path):
        
        importlib.import_module(paraview_script_path)
        
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        plt.imshow(mpimg.imread('/home/jyan_linux/Downloads/Software/atomics_jy/atomics/examples/solutions_iterations/density.png'))

    def plot(self, data_dict_list, ind, video=False):
        print("--------------plot-------------------")
        x = data_dict_list[ind]['volume_fraction']
        h = data_dict_list[ind]['compliance']
        rho_e = data_dict_list[ind]['y']
        # (prob['rho_e']/(1 + 8.*(1-prob['rho_e']))
        print('rho_e',type(rho_e))

        self.get_frame(1).clear_all_axes()

        with self.get_frame(1)[0, 0] as ax:
            paraview_script_path = '/home/jyan_linux/Downloads/Software/atomics_jy/atomics/examples/viz/paraview_script.py'
            self.add_paraview_plot(paraview_script_path)

        self.get_frame(1).write()




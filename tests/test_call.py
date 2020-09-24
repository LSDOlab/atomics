from subprocess import call

call(['python3','/home/jyan_linux/Downloads/Software/atomics_jy/atomics/tests/test_paraview.py'])

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
plt.imshow(mpimg.imread('/home/jyan_linux/Downloads/Software/atomics_jy/atomics/python_paraview.png'))
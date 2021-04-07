Getting started
===============

Installing
----------
To install ATOMiCS and run topology optimization problems, users need to follow these steps:

1.  Install `FEniCS <https://fenicsproject.org/download/>`_ partial differential equation (PDE) solver. 

  - For MAC users, the anaconda installation method is recommend.
    For Ubuntu users, please just install according to the Ubuntu installation guidline.
    For Windows users (haven't tested), please try the method of installing Ubuntu subsystem.

2. Install ``OpenMDAO``:

 - The installation of OpenMDAO: ``pip install 'openmdao[all]'`` .

3. Install ``ATOMiCS``:

  -Clone `this repository <https://github.com/LSDOlab/atomics>`_,
  navigate to the atomics directory. 
  Then, and use the command ``pip install -e .`` to install ATOMiCS.

Other recommandations: while the ``scipy`` optimizer in OpenMDAO works for some small-scale problems, we recommend `IPOPT <https://github.com/coin-or/Ipopt/>`_ or `SNOPT <http://ccom.ucsd.edu/~optimizers/downloads/>`_.
Note that ``SNOPT`` is a commercial optimizer.



.. toctree::
  :maxdepth: 2
  :titlesonly:

Getting started
===============

Installing
----------
To install ATOMiCS and run topology optimization problems, you need to follow these steps:

1.  Install FEniCS partial differential equation (PDE) solver from https://fenicsproject.org/download/. 

  - For MAC users, the anaconda installation method is recommend.
    For Ubuntu users, please just install according to the Ubuntu installation guidline.
    For Windows users (haven't tested), please try the method of installing Ubuntu subsystem.

2. Install ``OpenMDAO``:

 - The installation of OpenMDAO: ``pip install 'openmdao[all]'``

3. Install ``ATOMiCS``:

  - ``git clone`` this repository, navigate to the atomics directory. 
  Then, and use the command ``pip install -e .`` to install ATOMiCS.

Other recommandations: while the ``scipy`` optimizer in OpenMDAO works for some small scale problems, we recommend `IPOPT` (https://github.com/coin-or/Ipopt/) or `SNOPT` (http://ccom.ucsd.edu/~optimizers/downloads/).
Note that ``SNOPT`` is a commercial optimizer.

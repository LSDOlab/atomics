# ATOmiCS
ATOmiCS stands for **A**utomated **T**opology **O**ptimization for **m**ultidisciplinary problems using FEn**iCS**. It is a Python module that performs topology optimization for various physics problems with automated derivatives. ATOmiCS is implemented based on [OpenMDAO](https://openmdao.org/) and [FEniCS](https://fenicsproject.org/). The details of ATOmiCS can be found in the following article:

```
@article{yan2022topology,
  title={Topology optimization with automated derivative computation for multidisciplinary design problems},
  author={Yan, Jiayao and Xiang, Ru and Kamensky, David and Tolley, Michael T and Hwang, John T},
  journal={Structural and Multidisciplinary Optimization},
  volume={65},
  number={5},
  pages={1--20},
  year={2022},
  publisher={Springer}
}
```

A preprint of the above article can be found [here](https://github.com/LSDOlab/lsdo_bib/blob/main/pdf/yan2022topology.pdf).

Getting started
===============
For a detailed tutorial of ATOMiCS (automated topology optimization using OpenMDAO and FEniCS), please check our online documentations (lsdolab.github.io/atomics/).

Installing
----------
To install ATOMiCS and run topology optimization problems, you need to follow these steps:

1.  Install FEniCS 2019.1.0 partial differential equation (PDE) solver from https://fenicsproject.org/download/archive/. 

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

.. atomics documentation master file, created by
   sphinx-quickstart on Thu Sep 24 00:25:46 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ATOMiCS's documentation!
===================================
Hi, this is ATOMiCS 0.1 (\ **a**\utomated  **t**\opology  **o**\ptimization  in Open\ **M**\DAO using FEn\ **iCS**\ ). It is an open-source density-based topology optimization toolbox.
ATOMiCS is built on top of the `OpenMDAO framework <https://github.com/openmdao/blue>`_, which is described and documented `here <https://openmdao.org/newdocs/versions/latest>`_.
The solution approach to the PDE problems and the partial derivative computations are wrapped from `FEniCS Project <https://fenicsproject.org/>`_ documented `here <https://fenicsproject.org/olddocs/dolfin/latest/python/>`_.
The details of ATOmiCS can be found in the following article:

.. code-block:: python

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


A preprint of the above article can be found `here <https://github.com/LSDOlab/lsdo_bib/blob/main/pdf/yan2022topology.pdf>`_.




Documentation
-------------

.. toctree::
   :maxdepth: 2
   :titlesonly:

   _src_docs/getting_started
   _src_docs/solution
   _src_docs/examples
   _src_docs/methods
   _src_docs/pdes
   _src_docs/extension
   _src_docs/api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

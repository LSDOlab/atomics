.. _exp-label:


Examples library
================

We perform three case studies corresponding to those in the manuscript: 

* linear and nonlinear cantilever beam topology optimization for compliance minimization;
* battery pack optimizations for compliance minimization and mass minimization under constraints;
* a liquid crystal elastomers (LCE) topology optimization for shape matching. 

We perform these case study to demonstrate the capability of ATOMiCS:
  1. easily changeable PDE and solvers, dealing with nonlinear problems ``(case I)``; 
  2. multiphysics approach, binding with ``GMSH`` and handling unstructured mesh, changeable constraints and objectives ``(case II)``; 
  3. topology optimization for shape matching of smart material, 3D problem, adding other OpenMDAO components and design variables other than densities ``(case III)``.


For features that are not documented or cannot be discribed in detail in the above three case studies, we present other examples including

* a L-shape beam topology optimizaiton using external mesh ``(GMSH)``;
* cantilever beam topology optimization using FEniCS variational filter

--------------------------------------------------------------------

.. toctree::
  :maxdepth: 1
  :titlesonly:

  examples/case_studies
  examples/other_examples


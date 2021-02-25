Steps for solving a topology optimization problem in ATOMiCS
====================================================================

1. Define the mesh
-------------------
ATOMiCS supports FEniCS built-in meshes as well as external mesh of .vtk or .stl type from ``GMSH`` or other mesh generation tool.

    1.1 FEniCS built-in meshes:
        The documentations for FEniCS built-in meshes can be found here.

    1.2. External mesh:
        We use ``meshio`` to convert the external mesh to the formats that FEniCS accepts. An example mesh generated from ``GMSH`` GUI is shown below:
        
.. figure:: doc_gmsh_example.png
    :scale: 40 %
    :align: center        

2. Select a filter
---------------------------------------
.. code-block:: python

   from atomics.general_filter_comp import GeneralFilterComp
   from atomics.pdes.variational_filter import get_residual_form_variational_filter


3. Select a penalizarion scheme
---------------------------------------
.. code-block:: python

    residual_form = get_residual_form(
        ...,
        method=method_name
        # method=<method_name
    )


4. Solve for the states 
---------------------------------------
.. code-block:: python

    from atomics.pdes.<pde_name> import get_residual_form

    residual_form = get_residual_form(
        ...,
        method=method_name
        # method=<method_name
    )

    group = AtomicsGroup(pde_problem=pde_problem, problem_type='linear_problem')
    prob.model.add_subsystem('atomics_group', group, promotes=['*'])

5. Define outputs
---------------------------------------
.. code-block:: python
    
    output_form = ...
    pde_problem.add_scalar_output(<output_name>, <output_form>, <argument_name>)

5. Visualization
---------------------------------------

TODO: add a video

.. toctree::
  :maxdepth: 2
  :titlesonly:
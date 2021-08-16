Case study I: linear elastic cantilever beam topology optimization
===================================================================

The variational form for the linear elastic problem is

.. math:: \int_{\Omega}\sigma:\nabla v d x -\int_{\partial \Omega}(\sigma \cdot \eta) \cdot v d s=0 ,

where the :math:`\sigma`, :math:`v` are the stress tenser and the test functions. 

The code can be downloaded from 
`here <https://github.com/LSDOlab/atomics/blob/master/atomics/examples/case_1_cantilever_beam_opts/run_cantilever_beam.py>`_

1. Code
---------------------------------------

We explain the code in detail in this section.

1.1. Import
~~~~~~~~~~~~~~~~~~~~~~~~~~~
First, import ``dolfin``, ``numpy``, ``openmdao``, ``atomics.api``, ``atomics.pde``, and ``atomics.general_filter_comp``.

.. code-block:: python

  import dolfin as df
  import numpy as np
  import openmdao.api as om
  from atomics.api import PDEProblem, AtomicsGroup
  from atomics.pdes.linear_elastic import get_residual_form
  from atomics.general_filter_comp import GeneralFilterComp


1.2. Define the mesh
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

  np.random.seed(0)

  '''
  Define the mesh
  '''
  NUM_ELEMENTS_X = 80
  NUM_ELEMENTS_Y = 40
  LENGTH_X = 160.
  LENGTH_Y = 80.

  mesh = df.RectangleMesh.create(
      [df.Point(0.0, 0.0), df.Point(LENGTH_X, LENGTH_Y)],
      [NUM_ELEMENTS_X, NUM_ELEMENTS_Y],
      df.CellType.Type.quadrilateral,
  )

1.3. Define the PDE problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

  '''
  3. Setup the PDE problem
  '''
  # PDE problem
  pde_problem = PDEProblem(mesh)

  # Add input to the PDE problem:
  # name = 'density', function = density_function (function is the solution vector here)
  density_function_space = df.FunctionSpace(mesh, 'DG', 0)
  density_function = df.Function(density_function_space)
  pde_problem.add_input('density', density_function)

  # Add states to the PDE problem (line 58):
  # name = 'displacements', function = displacements_function (function is the solution vector here)
  # residual_form = get_residual_form(u, v, rho_e) from atomics.pdes.thermo_mechanical_uniform_temp
  # *inputs = density (can be multiple, here 'density' is the only input)

  displacements_function_space = df.VectorFunctionSpace(mesh, 'Lagrange', 1)
  displacements_function = df.Function(displacements_function_space)
  v = df.TestFunction(displacements_function_space)
  method='SIMP'
  residual_form = get_residual_form(
      displacements_function, 
      v, 
      density_function,
      method=method
  )

  residual_form -= df.dot(f, v) * dss(6)

  '''
  Define the traction boundary conditions
  '''
  # here traction force is applied on the middle of the right edge
  class TractionBoundary(df.SubDomain):
      def inside(self, x, on_boundary):
          return ((abs(x[1] - LENGTH_Y/2) < LENGTH_Y/NUM_ELEMENTS_Y + df.DOLFIN_EPS) and (abs(x[0] - LENGTH_X ) < df.DOLFIN_EPS*1.5e15))

  # Define the traction boundary
  sub_domains = df.MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
  upper_edge = TractionBoundary()
  upper_edge.mark(sub_domains, 6)
  dss = df.Measure('ds')(subdomain_data=sub_domains)
  f = df.Constant((0, -1. / 4 ))

  pde_problem.add_state('displacements', displacements_function, residual_form, 'density')

  # Add output-avg_density to the PDE problem:
  volume = df.assemble(df.Constant(1.) * df.dx(domain=mesh))
  avg_density_form = density_function / (df.Constant(1. * volume)) * df.dx(domain=mesh)
  pde_problem.add_scalar_output('avg_density', avg_density_form, 'density')

  # Add output-compliance to the PDE problem:
  compliance_form = df.dot(f, displacements_function) * dss(6)
  pde_problem.add_scalar_output('compliance', compliance_form, 'displacements')

  # Add Dirichlet boundary conditions to the PDE problem:
  pde_problem.add_bc(df.DirichletBC(displacements_function_space, df.Constant((0.0, 0.0)), '(abs(x[0]-0.) < DOLFIN_EPS)'))

1.4. Set up the OpenMDAO model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

    # Define the OpenMDAO problem and model

    prob = om.Problem()

    num_dof_density = pde_problem.inputs_dict['density']['function'].function_space().dim()

    comp = om.IndepVarComp()
    comp.add_output(
        'density_unfiltered', 
        shape=num_dof_density, 
        val=np.random.random(num_dof_density) * 0.86,
    )
    prob.model.add_subsystem('indep_var_comp', comp, promotes=['*'])

    comp = GeneralFilterComp(density_function_space=density_function_space)
    prob.model.add_subsystem('general_filter_comp', comp, promotes=['*'])


    group = AtomicsGroup(pde_problem=pde_problem)
    prob.model.add_subsystem('atomics_group', group, promotes=['*'])

    prob.model.add_design_var('density_unfiltered',upper=1, lower=1e-4)
    prob.model.add_objective('compliance')
    prob.model.add_constraint('avg_density',upper=0.40)

    # set up the optimizer
    if True:
        prob.driver = driver = om.pyOptSparseDriver()
        driver.options['optimizer'] = 'SNOPT'
        driver.opt_settings['Verify level'] = 0

        driver.opt_settings['Major iterations limit'] = 100000
        driver.opt_settings['Minor iterations limit'] = 100000
        driver.opt_settings['Iterations limit'] = 100000000
        driver.opt_settings['Major step limit'] = 2.0

        driver.opt_settings['Major feasibility tolerance'] = 1.0e-6
        driver.opt_settings['Major optimality tolerance'] =2.e-10
    else:
        prob.driver = om.ScipyOptimizeDriver() 
        prob.driver.options['optimizer'] = 'SLSQP' 

    prob.setup()
    prob.run_model()
    # print(prob['compliance']); exit()
    prob.run_driver()


    #save the solution vector
    if method =='SIMP':
        penalized_density  = df.project(density_function**3, density_function_space) 
    else:
        penalized_density  = df.project(density_function/(1 + 8. * (1. - density_function)), density_function_space) 

    df.File('solutions/case_1/cantilever_beam/displacement.pvd') << displacements_function
    df.File('solutions/case_1/cantilever_beam/penalized_density.pvd') << penalized_density

2. Results (density plot)
---------------------------------------

The users can visualize the optimized densities by opening the ``<name>.pvd`` from Paraview.

    .. figure:: doc_case1_1_result.png
        :scale: 40 %
        :align: center
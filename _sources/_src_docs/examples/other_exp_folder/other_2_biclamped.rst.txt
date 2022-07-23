biclamped beam topology optimization 
==========================================


The variational form for the linear elastic problem is

.. math:: \int_{\Omega}\sigma:\nabla v d x -\int_{\partial \Omega}(\sigma \cdot \eta) \cdot v d s=0 ,

where the :math:`\sigma`, :math:`v` are the stress tenser and the test functions. 

The code can be downloaded from 
`here <https://github.com/LSDOlab/atomics/blob/master/atomics/examples/other_examples/run_biclamped_thermoelastic.py>`_

1. Code
---------------------------------------
.. code-block:: python

    import dolfin as df

    import numpy as np

    import openmdao.api as om

    from atomics.api import PDEProblem, AtomicsGroup
    from atomics.pdes.thermo_mechanical_uniform_temp import get_residual_form

    # from cartesian_density_filter_comp import CartesianDensityFilterComp
    from atomics.general_filter_comp import GeneralFilterComp


    np.random.seed(0)

    # Define the mesh and create the PDE problem
    NUM_ELEMENTS_X = 80
    NUM_ELEMENTS_Y = 40
    LENGTH_X = 2.
    LENGTH_Y = 1.
    K = 199.5e9
    ALPHA = 15.4e-6

    mesh = df.RectangleMesh.create(
        [df.Point(0.0, 0.0), df.Point(LENGTH_X, LENGTH_Y)],
        [NUM_ELEMENTS_X, NUM_ELEMENTS_Y],
        df.CellType.Type.quadrilateral,
    )

    # Define the boundary condition
    class BottomBoundary(df.SubDomain):
        def inside(self, x, on_boundary):
            return (abs(x[1] - 0.) < df.DOLFIN_EPS_LARGE and abs(x[0] - LENGTH_X / 2) < 2. * LENGTH_X / NUM_ELEMENTS_X)

    # Define the traction boundary
    sub_domains = df.MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
    upper_edge = BottomBoundary()
    upper_edge.mark(sub_domains, 6)
    dss = df.Measure('ds')(subdomain_data=sub_domains)
    f = df.Constant((0, -5.e6/(4. * LENGTH_X / NUM_ELEMENTS_X)))

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
    residual_form = get_residual_form(
        displacements_function, 
        v, 
        density_function,
        K,
        ALPHA
    )
    residual_form -= df.dot(f, v) * dss(6)
    pde_problem.add_state('displacements', displacements_function, residual_form, 'density')

    # Add output-avg_density to the PDE problem:
    volume = df.assemble(df.Constant(1.) * df.dx(domain=mesh))
    avg_density_form = density_function / (df.Constant(1. * volume)) * df.dx(domain=mesh)
    pde_problem.add_scalar_output('avg_density', avg_density_form, 'density')

    # Add output-compliance to the PDE problem:
    compliance_form = df.dot(f, displacements_function) * dss(6)
    pde_problem.add_scalar_output('compliance', compliance_form, 'displacements')

    # Add boundary conditions to the PDE problem:
    pde_problem.add_bc(df.DirichletBC(displacements_function_space, df.Constant((0.0, 0.0)), '(abs(x[0]-0.) < DOLFIN_EPS)'))
    pde_problem.add_bc(df.DirichletBC(displacements_function_space, df.Constant((0.0, 0.0)), '(abs(x[0]-2.) < DOLFIN_EPS)'))

    # num_dof_density = V_density.dim()

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

    # comp = CartesianDensityFilterComp(
    #     length_x=LENGTH_X,
    #     length_y=LENGTH_Y,
    #     num_nodes_x=NUM_ELEMENTS_X + 1,
    #     num_nodes_y=NUM_ELEMENTS_Y + 1,
    #     num_dvs=num_dof_density, 
    #     radius=2. * LENGTH_Y / NUM_ELEMENTS_Y,
    # )
    # prob.model.add_subsystem('density_filter_comp', comp, promotes=['*'])

    comp = GeneralFilterComp(density_function_space=density_function_space)
    prob.model.add_subsystem('general_filter_comp', comp, promotes=['*'])


    group = AtomicsGroup(pde_problem=pde_problem)
    prob.model.add_subsystem('atomics_group', group, promotes=['*'])

    prob.model.add_design_var('density_unfiltered',upper=1, lower=1e-4)
    prob.model.add_objective('compliance')
    prob.model.add_constraint('avg_density',upper=0.20)

    prob.driver = driver = om.pyOptSparseDriver()
    driver.options['optimizer'] = 'SNOPT'
    driver.opt_settings['Verify level'] = 0

    driver.opt_settings['Major iterations limit'] = 100000
    driver.opt_settings['Minor iterations limit'] = 100000
    driver.opt_settings['Iterations limit'] = 100000000
    driver.opt_settings['Major step limit'] = 2.0

    driver.opt_settings['Major feasibility tolerance'] = 1.0e-6
    driver.opt_settings['Major optimality tolerance'] =2.e-12

    prob.setup()
    prob.run_model()

    # prob.check_partials(compact_print=True)
    # print(prob['compliance']); exit()

    prob.run_driver()

    #save the solution vector
    df.File('solutions/displacement.pvd') << displacements_function
    df.File('solutions/stiffness_th_55.pvd') << density_function




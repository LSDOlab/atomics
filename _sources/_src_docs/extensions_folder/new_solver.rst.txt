Potentially implementing a new solver
---------------------------------------------

We implemented six options (two direct and two iterative solvers) for the solution of the total derivative
as well as three options for solving the finite element problem.
Please see subsection 6 of :ref:`step-label` for details.

If the user needs to change to a customized solver for solving the total derivative,
they need to modify the ``solve_linear()`` function in  ``atomics/atomics/states_comp.py``
The user can refer to OpenMDAO documentation on
`implicit component <https://openmdao.org/newdocs/versions/latest/features/core_features/working_with_components/implicit_component.html>`_ for details.
If mode is ``fwd`` (direct method), the right-hand side vector is ``d_residuals`` and the solution vector is ``d_outputs``.
If mode is ``rev`` (adjoint method), the right-hand side vector is ``d_outputs`` and the solution vector is ``d_residuals``.

If the user wants to change to a customized solver for solving the finite element problem,
they need to modify the ``solve_nonlinear()`` function in  ``atomics/atomics/states_comp.py``
The user can refer to OpenMDAO documentation on
`implicit component <https://openmdao.org/newdocs/versions/latest/features/core_features/working_with_components/implicit_component.html>`_ for details.
The options for the solver that we recommend are other FEniCS solvers or PETSc solvers.
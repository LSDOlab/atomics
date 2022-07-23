Adding additional design variables, constraints, or changing to a different objective
------------------------------------------------------------------------------------------
The design variables, constraints, and the objective for the optimization are all specified in the ``run_file``.

    1) Adding additional design variables
        The design variables are specified using OpenMDAO `independent variable component <https://openmdao.org/newdocs/versions/latest/_srcdocs/packages/core/indepvarcomp.html>`_.

        .. code-block:: python
                
            comp = om.IndepVarComp()
            comp.add_output(<design_variable_name>, shape=shape, val=np.ones((shape)))

        Then, the user can add it to the optimization problem.

        .. code-block:: python
                
            prob = om.Problem()
            prob.model.add_design_var(<design_variable_name>, upper=upper_bd, lower=lower_bd)

    2) Adding additional constraints or changing to a different objective
        The constraints and objective are the output for the optimization problem,
        which can be computed using one of the three methods described in Subsection 2 of this Section.
        Then, the user just need to specify

        .. code-block:: python

            prob.model.add_objective(<objective_name>)
            prob.model.add_constraint(<constraint_name>, upper=upper_bd, lower=lower_bd)
        
        in the  ``run_file``.
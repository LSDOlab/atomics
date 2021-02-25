API
===

StatesComp
-----------

.. autoclass:: atomics.states_comp.StatesComp

  .. automethod:: atomics.states_comp.StatesComp.initialize
  .. automethod:: atomics.states_comp.StatesComp.setup
  .. automethod:: atomics.states_comp.StatesComp.solve_nonlinear
  .. automethod:: atomics.states_comp.StatesComp.solve_linear


PDEProblem
-----------
.. autoclass:: atomics.api.PDEProblem

  .. automethod:: atomics.api.PDEProblem.__init__
  .. automethod:: atomics.api.PDEProblem.add_bc
  .. automethod:: atomics.api.PDEProblem.add_input
  .. automethod:: atomics.api.PDEProblem.add_scalar_output
  .. automethod:: atomics.api.PDEProblem.add_field_output

AtomicsGroup
-----------
.. autoclass:: atomics.api.AtomicsGroup

  .. automethod:: atomics.api.AtomicsGroup.initialize
  .. automethod:: atomics.api.AtomicsGroup.setup

GeneralFilterComp (linear direct filter)
-----------
.. autoclass:: atomics.general_filter_comp.GeneralFilterComp

  .. automethod:: atomics.general_filter_comp.GeneralFilterComp.initialize
  .. automethod:: atomics.general_filter_comp.GeneralFilterComp.compute

GeneralFilterComp (linear direct filter)
-----------
.. autoclass:: atomics.states_comp_filter.StatesFilterComp

  .. automethod:: atomics.states_comp_filter.StatesFilterComp.initialize
  .. automethod:: atomics.states_comp_filter.StatesFilterComp.setup
  .. automethod:: atomics.states_comp_filter.StatesFilterComp.solve_nonlinear
  .. automethod:: atomics.states_comp_filter.StatesFilterComp.solve_linear

.. toctree::
  :maxdepth: 1
  :titlesonly:



Implementing a customized computation for preprocessing, postprocessing or outputs
------------------------------------------------------------------------------------------------

This subsection explains how to implement a new customized computation.
There are three types of computations we defined in the ATOmiCS toolbox:

   1) The computation that can be directly implemented using OpenMDAO
        An example can be the interpolation component for case study 3 in the paper.
        In order to formulate this kind of component, the user needs to refer to OpenMDAO documentation for the `explicit component <https://openmdao.org/newdocs/versions/latest/features/core_features/working_with_components/explicit_component.html>`_.
        
        We briefly summarize this process in this documentation.
        Defining a customized explicit component requires writing
        a python class inherited from OpenMDAO ``ExplicitComponent`` 
        There are four main functions inside the class:

            ``setup()`` that takes the input and outputs of the component;

            ``setup_partials`` that defines the relationship of the outputs with respect to the inputs;
            
            ``compute()`` that contains the computation of the outputs given the inputs;
            
            and ``compute_partials()`` computes the partial derivative of the outputs with respect to the inputs.

   2) The computation that combines OpenMDAO and FEniCS and outputs a scalar on the entire domain
        This kind of output is wrapped as ``scalar_output`` in ATOmiCS. The user can define this type of output
        in the ``run_file``. 

        .. code-block:: python
    
            output_form = ...
            pde_problem.add_scalar_output(<output_name>, <output_form>, <argument_names>)

        The ``<output_name>`` is the name of the customized output;
        ``<output_form>`` is the form expressing the output using FEniCS UFL;
        ``<argument_names>`` are the name list of arguments that the user wants to take derivatives with respect to.

   3) The computation that combines OpenMDAO and FEniCS and outputs a scalar on each vertex

    .. code-block:: python
            
        output_form = ...
        pde_problem.add_field_output(<output_name>, <output_form>, <argument_names>)

    The ``<output_name>`` is the name of the customized output;
    ``<output_form>`` is the form expressing the output using FEniCS UFL;
    ``<argument_names>`` are the name list of arguments that the user wants to take derivatives with respect to.
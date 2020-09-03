class PDEProblem(object):

    def __init__(self, mesh):
        self.mesh = mesh

        self.inputs_dict = dict()
        self.states_dict = dict()
        self.scalar_outputs_dict = dict()
        self.field_outputs_dict = dict()

        self.bcs_list = list()

    def add_bc(self, bc):
        self.bcs_list.append(bc)

    def add_input(self, name, function):
        if name in self.inputs_dict:
            raise ValueError('name has already been used for an input')

        function.rename(name, name)
        self.inputs_dict[name] = dict(
            function=function,
        )

    def add_state(self, name, function, residual_form, *arguments):
        function.rename(name, name)
        self.states_dict[name] = dict(
            function=function,
            residual_form=residual_form,
            arguments=arguments,
        )

    def add_scalar_output(self, name, form, *arguments):
        self.scalar_outputs_dict[name] = dict(
            form=form,
            arguments=arguments,
        )

    def add_field_output(self, name, function, expression, *arguments):
        function.rename(name, name)
        self.field_outputs_dict[name] = dict(
            function=function,
            expression=expression,
            arguments=arguments,
        )
        raise NotImplemented()
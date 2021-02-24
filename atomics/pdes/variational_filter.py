import dolfin as df


def get_residual_form_variational_filter(rho_e_unfiltered, rho_e, C=df.Constant(7e-1)):
    v = df.TestFunction(rho_e_unfiltered.function_space())
    h = df.CellDiameter(rho_e_unfiltered.function_space().mesh())
    res_filter = (rho_e-rho_e_unfiltered)*v*df.dx + C*df.avg(h)*df.jump(rho_e)*df.jump(v)*df.dS
    return res_filter

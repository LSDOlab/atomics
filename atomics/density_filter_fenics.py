import dolfin as df


def get_residual_form(unfiltered_density, v, filtered_density, mesh):
    # v = TestFunction(self.VC)
    h = df.CellDiameter(mesh)
    C = df.Constant(0.43)
    res_filter = (filtered_density-unfiltered_density)*v*df.dx + \
        C*df.avg(h)*df.jump(filtered_density)*df.jump(v)*df.dS
    
    return res_filter
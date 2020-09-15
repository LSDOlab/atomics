import dolfin as df
import meshio
filename = 'test_gmsh_vtk'
mesh = meshio.read(
    filename,  # string, os.PathLike, or a buffer/open file
    file_format="vtk"  # optional if filename is a path; inferred from extension
)
points = mesh.points
cells = mesh.cells
meshio.write_points_cells(
    "fenics_mesh_l_bracket.xml",
    points,
    cells,
    # Optionally provide extra data on points, cells, etc.
    # point_data=point_data,
    # cell_data=cell_data,
    # field_data=field_data
    )
NUM_ELEMENTS_X = 80
NUM_ELEMENTS_Y = 40
LENGTH_X = 160.
LENGTH_Y = 80.
class TractionBoundary(df.SubDomain):
    def inside(self, x, on_boundary):
        return ((abs(x[1] - LENGTH_Y/2) < LENGTH_Y/NUM_ELEMENTS_Y * 2.) and (abs(x[0] - LENGTH_X ) < df.DOLFIN_EPS))

# Define the traction boundary
mesh = df.Mesh("fenics_mesh_l_bracket.xml")

sub_domains = df.MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
upper_edge = TractionBoundary()
upper_edge.mark(sub_domains, 6)
dss = df.Measure('ds')(subdomain_data=sub_domains)
f = df.Constant((0, -1. / 4 ))

# # PDE problem
# pde_problem = PDEProblem(mesh)
# Add input to the PDE problem:
# name = 'density', function = density_function (function is the solution vector here)
density_function_space = df.FunctionSpace(mesh, 'DG', 0)
density_function = df.Function(density_function_space)
# pde_problem.add_input('density', density_function)
df.plot(mesh)
import matplotlib.pyplot as ply
ply.show()
import dolfin as df
import meshio

# First, we need to export gmsh file to a vtk file (or stl maybe, not tested)
# Convert the mesh type from gmsh vtk to xml for fenics 
# (XDMF not working not sure why)
filename = 'test_gmsh_vtk'
mesh = meshio.read(
    filename,  
    file_format="vtk"  
)
points = mesh.points
cells = mesh.cells
meshio.write_points_cells(
    "fenics_mesh_l_bracket.xml",
    points,
    cells,
    )

# test if it work with fenics 
# by defining a traction boundary
NUM_ELEMENTS_X = 80
NUM_ELEMENTS_Y = 40
LENGTH_X = 160.
LENGTH_Y = 80.
class TractionBoundary(df.SubDomain):
    def inside(self, x, on_boundary):
        return ((abs(x[1] - LENGTH_Y/2) < LENGTH_Y/NUM_ELEMENTS_Y * 2.) and (abs(x[0] - LENGTH_X ) < df.DOLFIN_EPS))

# redefine the mesh
mesh = df.Mesh("fenics_mesh_l_bracket.xml")

# Mark the traction boundary ass dss(6) (line 37)
sub_domains = df.MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
upper_edge = TractionBoundary()
upper_edge.mark(sub_domains, 6)
dss = df.Measure('ds')(subdomain_data=sub_domains)

# Define the function space in fenics
density_function_space = df.FunctionSpace(mesh, 'DG', 0)
density_function = df.Function(density_function_space)


df.plot(mesh)
import matplotlib.pyplot as plt
plt.show()
import dolfin as df
from mshr import *
import matplotlib.pyplot as plt


mesh = df.UnitSquareMesh(10, 10)
V = df.FunctionSpace(mesh, "Lagrange", 1)
# df.plot(mesh, title="Unit square")
# plt.show()

'''Simplify the heating as a fixed temperature Dirichlet BC in the middle of the substrate'''
class HeatingBoundary(df.SubDomain):
    def inside(self, x, on_boundary):
        return (abs(x[0] - 0.) < df.DOLFIN_EPS)

# Define boundary condition
u0 = df.Constant(200.0)
bc = df.DirichletBC(V, u0, HeatingBoundary())

'''Assuming a convection BC on the right'''
class ConvectionBoundary(df.SubDomain):
    def inside(self, x, on_boundary):
        return (abs(x[0] - 1.) < df.DOLFIN_EPS and on_boundary)

# Define the traction boundary
sub_domains = df.MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
upper_edge = ConvectionBoundary()
upper_edge.mark(sub_domains, 6)
dss = df.Measure('ds')(subdomain_data=sub_domains)

conduction_coeff = 1.
convection_coeff = 1.
T_ambient = 25


f = df.Constant(0.)
T = df.Function(V)
T_test = df.TestFunction(V)



integrals_R_a = convection_coeff*T*T_test*dss(6)
integrals_R_L = convection_coeff*T_ambient*T_test*dss(6)
a = conduction_coeff*df.dot(df.grad(T), df.grad(T_test))*df.dx + integrals_R_a
L = f*T_test*df.dx + integrals_R_L

df.solve(a-L==0, T, bc)
file = df.File("test_convection_LCE.pvd")
file << T

'''Now do the elastic part'''
V = df.VectorFunctionSpace(mesh, "Lagrange", 1)

u = df.Function(V)
v = df.TestFunction(V)

E = 1.e6 

nu = 0.3 # Poisson's ratio

# # lame's parameter
lambda_ = E * nu/(1. + nu)/(1 - 2 * nu)
mu = E / 2 / (1 + nu) #lame's parameters
thermal_expansion = -1e-3


# # Th = df.Constant(17.)

w_ij = 0.5 * (df.grad(u) + df.grad(u).T)
v_ij = 0.5 * (df.grad(v) + df.grad(v).T)

d = len(u)

sigm = (lambda_ * df.tr(w_ij) - thermal_expansion * (3. * lambda_ + 2. * mu) * T) * df.Identity(d) + 2 * mu * w_ij

res = df.inner(sigm, v_ij) * df.dx

class ClampedBoundary(df.SubDomain):
    def inside(self, x, on_boundary):
        return (abs(x[0] - 0.) < df.DOLFIN_EPS and on_boundary)

# Define boundary condition
u0 = df.Constant((0.0, 0.0))
bc = df.DirichletBC(V, u0, ClampedBoundary())

df.solve(res==0, u, bc)
    
file = df.File("test_convection_LCE_deformation.pvd")
file << u



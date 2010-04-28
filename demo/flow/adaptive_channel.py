__author__ = "Harish Narayanan"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from cbc.flow import *

# Constants related to the geometry of the channel and the obstruction
channel_length  = 4.0
channel_height  = 1.0
structure_left  = 1.4
structure_right = 1.6
structure_top   = 0.5
nx = 80
ny = nx/4 

# Parameters
dt = 0.01
T = 2.0
tol = 1e-4

# Create the complete mesh
mesh = Rectangle(0.0, 0.0, channel_length, channel_height, nx, ny)

# Define dimension of mesh
D = mesh.topology().dim()

# Define structure subdomain
class Structure(SubDomain):
    def inside(self, x, on_boundary):
        return (x[0] >= structure_left) and (x[0] <= structure_right) \
            and (x[1] <= structure_top)

# Create structure subdomain
structure = Structure()

# Create subdomain markers (0=fluid,  1=structure)
sub_domains = MeshFunction("uint", mesh, D)
sub_domains.set_all(0)
structure.mark(sub_domains, 1)

# Create cell_domain markers (0=fluid,  1=structure)
cell_domains = MeshFunction("uint", mesh, D)
cell_domains.set_all(0)
structure.mark(cell_domains, 1)

# Extract submeshes for fluid and structure
Omega = mesh
Omega_F = SubMesh(mesh, sub_domains, 0)
Omega_S = SubMesh(mesh, sub_domains, 1)
omega_F = Mesh(Omega_F)

# Create facet marker for outflow
right = compile_subdomains("x[0] == channel_length")
exterior_boundary = MeshFunction("uint", Omega, D-1)
right.mark(exterior_boundary, 2)


# Define inflow boundary
def inflow(x):
    return x[0] < DOLFIN_EPS and x[1] > DOLFIN_EPS and x[1] < channel_height - DOLFIN_EPS

# Define outflow boundary
def outflow(x):
    return x[0] > channel_length - DOLFIN_EPS and x[1] > DOLFIN_EPS and x[1] < channel_height - DOLFIN_EPS

# Define noslip boundary
def noslip(x, on_boundary):
    return on_boundary and not inflow(x) and not outflow(x)

class Channel(NavierStokes):

    def mesh(self):
        return omega_F

    def viscosity(self):
        return 1e-2

    def density(self):
        return 1.0

    def boundary_conditions(self, V, Q):

        # Create no-slip boundary condition for velocity
        bcu = DirichletBC(V, Constant((0, 0)), noslip)

        # Create inflow and outflow boundary conditions for pressure
        bcp0 = DirichletBC(Q, Constant(2), inflow)
        bcp1 = DirichletBC(Q, Constant(0), outflow)

        return [bcu], [bcp0, bcp1]

   
    def time_step(self):
        return dt

    def end_time(self):
        return T

    def functional(self, u, p):
        return u((1.0, 0.5))[0]

    def reference(self, t):
        num_terms = 30
        u = 1.0
        c = 1.0
        for n in range(1, 2*num_terms, 2):
            a = 32.0 / (DOLFIN_PI**3*n**3)
            b = (1.0/8.0)*DOLFIN_PI**2*n**2
            c = -c
            u += a*exp(-b*t)*c
        return u

    def __str__(self):
        return "Pressure-driven channel (2D)"

class ChannelDual(NavierStokesDual):

    def mesh(self):
        return omega_F

    def viscosity(self):
        return 1e-2

    def boundary_conditions(self, V, Q):

        # Create no-slip boundary condition for velocity
        bcu = DirichletBC(V, Constant((0, 0)), noslip)

        # Create inflow and outflow boundary conditions for pressure
        bcp0 = DirichletBC(Q, Constant(1), inflow)
        bcp1 = DirichletBC(Q, Constant(0), outflow)

        return [bcu], [bcp0, bcp1]
# FIXME: base the following on zie goal funtional
#     def initial_conditions(self, V, Q):
#         u0 = Constant((0, 0))
#         p0 = Expression("1 - x[0]")
#         return u0, p0

    def time_step(self):
        return dt

    def end_time(self):
        return T

    def functional(self, u, p, V, Q, n):
#         right = RightBoundary(Q)
#         right = RightBoundary(Q)
#         sigma = nu*(grad(u) + grad(u).T) - p*Identity(u.cell().d)
#        n_F = FacetNormal(Omega_F)
        goal_F = inner(u, n)*ds(2)
        return goal_F

    def __str__(self):
        return "Pressure-driven channel (2D)"

# Solve problem
# problem = Channel()
# problem.parameters["solver_parameters"]["plot_solution"] = True
# problem.parameters["solver_parameters"]["store_solution"] = True
# u, p = problem.solve()
 
dual_problem = ChannelDual()
dual_problem.parameters["solver_parameters"]["plot_solution"] = True
dual_problem.parameters["solver_parameters"]["store_solution"] = False
w, r = dual_problem.solve()

#interactive()

# # Check error
# e = problem.functional(u, p) - problem.reference(0.5)
# print "Error is", e

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
T = 0.1
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

# Create subdomain markers
sub_domains = MeshFunction("uint", mesh, D)
sub_domains.set_all(0)
structure.mark(sub_domains, 1)

# Extract submesh for the fluid
Omega = mesh
Omega_F = SubMesh(mesh, sub_domains, 0)
omega_F = Mesh(Omega_F)

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
        bcp0 = DirichletBC(Q, Constant(1.0), inflow)
        bcp1 = DirichletBC(Q, Constant(0.0), outflow)

        return [bcu], [bcp0, bcp1]

    def time_step(self):
        return dt

    def end_time(self):
        return T

    def __str__(self):
        return "Pressure-driven channel (2D)"

class ChannelDual(NavierStokesDual):

    def mesh(self):
        return omega_F

    def boundary_markers(self):
        right = compile_subdomains("x[0] >= 4.0 - DOLFIN_EPS")
        boundary_marker = MeshFunction("uint", self.mesh(), self.mesh().topology().dim() - 1)
        right.mark(boundary_marker, 2)
        return boundary_marker

    def viscosity(self):
        return 1e-2

    def boundary_conditions(self, V, Q):

        # Create no-slip boundary condition for velocity
        bcu = DirichletBC(V, Constant((0.0, 0.0)), noslip)

        # Create inflow and outflow boundary conditions for pressure
        bcp0 = DirichletBC(Q, Constant(0.0), inflow)
        bcp1 = DirichletBC(Q, Constant(0.0), outflow)

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
        goal = inner(u, n)*ds(2)
        return goal

    def __str__(self):
        return "Pressure-driven channel (2D)"

# # Solve problem
# problem = Channel()
# problem.parameters["solver_parameters"]["plot_solution"] = True
# problem.parameters["solver_parameters"]["store_solution_data"] = True
# u, p = problem.solve()

dual_problem = ChannelDual()
dual_problem.parameters["solver_parameters"]["plot_solution"] = True
dual_problem.parameters["solver_parameters"]["save_solution"] = True
dual_problem.parameters["solver_parameters"]["store_solution_data"] = False
w, r = dual_problem.solve()

#interactive()

# # Check error
# e = problem.functional(u, p) - problem.reference(0.5)
# print "Error is", e

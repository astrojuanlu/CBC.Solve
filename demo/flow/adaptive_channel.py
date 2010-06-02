__author__ = "Harish Narayanan"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from cbc.flow import *

# Parameters
dt = 0.001
T = 0.1
tol = 1e-4

mesh = UnitSquare(10, 10)

# Define outflow boundary
def outflow(x):
    return x[0] > 1.0 - DOLFIN_EPS and x[1] > DOLFIN_EPS and x[1] < 1.0 - DOLFIN_EPS

# Define inflow boundary
def inflow(x):
    return x[0] < 0.0 + DOLFIN_EPS and x[1] > DOLFIN_EPS and x[1] < 1.0 - DOLFIN_EPS

# Define noslip boundary
def noslip(x, on_boundary):
    return on_boundary and not inflow(x) and not outflow(x)

class Channel(NavierStokes):

    def mesh(self):
        return mesh

    def viscosity(self):
        return 1.0/8.0

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
        return mesh

    def boundary_conditions(self, V, Q):

        # Create no-slip boundary condition for velocity
        bcu = DirichletBC(V, Constant((0.0, 0.0)), noslip)

        # Create inflow and outflow boundary conditions for pressure
        bcp0 = DirichletBC(Q, Constant(0.0), inflow)
        bcp1 = DirichletBC(Q, Constant(0.0), outflow)

        return [bcu], [bcp0, bcp1]

    def time_step(self):
        return dt

    def end_time(self):
        return T

    def functional(self, u, p, V, Q, n):
        goal = u[0]*ds(2)
        return goal

    def boundary_markers(self):
        # ("x == 0", 2)
        right = compile_subdomains("x[0] == 1.0")
        boundary_marker = MeshFunction("uint", self.mesh(), self.mesh().topology().dim() - 1)
        right.mark(boundary_marker, 2)
        return boundary_marker

    def __str__(self):
        return "Pressure-driven channel (2D)"

# Solve problem
# problem = Channel()
# problem.parameters["solver_parameters"]["plot_solution"] = True
# problem.parameters["solver_parameters"]["store_solution_data"] = True
# u, p = problem.solve()

dual_problem = ChannelDual()
dual_problem.parameters["solver_parameters"]["plot_solution"] = True
dual_problem.parameters["solver_parameters"]["save_solution"] = True
dual_problem.parameters["solver_parameters"]["store_solution_data"] = False
w, r = dual_problem.solve()

# # Check error
# e = problem.functional(u, p) - problem.reference(0.5)
# print "Error is", e

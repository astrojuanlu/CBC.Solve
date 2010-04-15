__author__ = "Harish Narayanan"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from cbc.flow import *

# Boundaries
def noslip_boundary(x):
    return x[1] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS

def inflow_boundary(x):
    return x[0] < DOLFIN_EPS

def outflow_boundary(x):
    return x[0] > 1 - DOLFIN_EPS

# Parameters
nu = 1.0/8.0
domain = UnitSquare(16, 16)
T = 0.5
dt = 0.01

class Channel(NavierStokes):

    def mesh(self):
        return domain

    def viscosity(self):
        return nu

    def boundary_conditions(self, V, Q):

        # Create no-slip boundary condition for velocity
        bcu = DirichletBC(V, Constant((0, 0)), noslip_boundary)

        # Create inflow and outflow boundary conditions for pressure
        bcp0 = DirichletBC(Q, Constant(1), inflow_boundary)
        bcp1 = DirichletBC(Q, Constant(0), outflow_boundary)

        return [bcu], [bcp0, bcp1]

    def initial_conditions(self, V, Q):
        u0 = Constant((0, 0))
        p0 = Expression("1 - x[0]")
        return u0, p0

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
        return domain

    def viscosity(self):
        return nu

    def boundary_conditions(self, V, Q):

        # Create no-slip boundary condition for velocity
        bcu = DirichletBC(V, Constant((0, 0)), noslip_boundary)

        # Create inflow and outflow boundary conditions for pressure
        bcp0 = DirichletBC(Q, Constant(1), inflow_boundary)
        bcp1 = DirichletBC(Q, Constant(0), outflow_boundary)

        return [bcu], [bcp0, bcp1]

    def initial_conditions(self, V, Q):
        u0 = Constant((0, 0))
        p0 = Expression("1 - x[0]")
        return u0, p0

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

# Solve problem
# problem = Channel()
# problem.parameters["solver_parameters"]["plot_solution"] = False
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

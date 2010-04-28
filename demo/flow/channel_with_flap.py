__author__ = "Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2010-04-28

from cbc.flow import *

def inflow_boundary(x):
    return x[0] < DOLFIN_EPS

def outflow_boundary(x):
    return x[0] > 4.0 - DOLFIN_EPS

def noslip_boundary(x, on_boundary):
    return \
        (x[1] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS) or \
        (on_boundary and abs(x[0] - 1.5) < 0.1 + DOLFIN_EPS)

class ChannelWithFlap(NavierStokes):

    def mesh(self):

        # Vertical resolution
        n = 30

        # Define geometry for channel
        channel = Rectangle(0.0, 0.0, 4.0, 1.0, 4*n, n)

        # Define geometry for flap
        class Flap(SubDomain):
            def inside(self, x, on_boundary):
                return x[0] >= 1.4 and x[0] <= 1.6 and x[1] <= 0.5
        flap = Flap()

        # Extract mesh for sub domain channel - flap
        sub_domains = MeshFunction("uint", channel, 2)
        sub_domains.set_all(0)
        flap.mark(sub_domains, 1)
        mesh = SubMesh(channel, sub_domains, 0)

        return mesh

    def viscosity(self):
        return 0.1

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

    def end_time(self):
        return 0.5

    def functional(self, u, p):
        return u((4.0, 0.5))[0]

    def __str__(self):
        return "Pressure-driven flow in channel with a flap (2D)"

# Solve problem
problem = ChannelWithFlap()
parameters["form_compiler"]["cpp_optimize"] = True
problem.parameters["solver_parameters"]["plot_solution"] = True
problem.parameters["solver_parameters"]["save_solution"] = True
u, p = problem.solve()

# Print value of functional
ux = problem.functional(u, p)
print "ux =", ux

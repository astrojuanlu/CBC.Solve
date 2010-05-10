__author__ = "Harish Narayanan"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from cbc.flow import *

def inflow_boundary(x):
    return x[0] == -10.0

def outflow_boundary(x):
    return x[0] == 10.0

# Dirichlet boundary
def noslip_boundary(x, on_boundary):
    return on_boundary and \
        not inflow_boundary(x) and \
        not outflow_boundary(x)

class Aneurysm(NavierStokes):

    def mesh(self):
        return Mesh("aneurysm.xml")

    def viscosity(self):
        return 1.0 / 8.0

    def boundary_conditions(self, V, Q):

        # Create no-slip boundary condition for velocity
        bcu = DirichletBC(V, Constant((0, 0)), noslip_boundary)

        # Create inflow and outflow boundary conditions for pressure
        bcp0 = DirichletBC(Q, Constant(1), inflow_boundary)
        bcp1 = DirichletBC(Q, Constant(0), outflow_boundary)

        return [bcu], [bcp0, bcp1]

    def end_time(self):
        return 5.0

    def __str__(self):
        return "Pressure-driven flow in an artery with an aneurysm"

# Solve problem
problem = Aneurysm()
problem.parameters["solver_parameters"]["store_solution_data"] = True
u, p = problem.solve()

__author__ = "Kent-Andre Mardal"
__copyright__ = "Copyright (C) 2011 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"


from cbc.flow import *

class BoundaryValue(Expression):

    def eval(self, values, x):
        if x[0] > DOLFIN_EPS and x[0] < 1.0 - DOLFIN_EPS and x[1] > 1.0 - DOLFIN_EPS:
            values[0] = 1.0
            values[1] = 0.0
        else:
            values[0] = 0.0
            values[1] = 0.0

    def value_shape(self):
        return (2,)

class Aneurysm(NavierStokes):

    def mesh(self):
        return Mesh("logg.xml.gz")

    def viscosity(self):
        return 0.00345 

    def velocity_dirichlet_values(self):
        self.g = BoundaryValue()
        return [self.g]

    def velocity_dirichlet_boundaries(self):
        return [3]

    def velocity_initial_condition(self):
        return (0, 0, 0)

    def pressure_dirichlet_values(self):
        return 0 

    def pressure_dirichlet_boundaries(self):
        return [2, 4]


    def pressure_initial_condition(self):
        return 0

    def end_time(self):
        return 1.0

    def functional(self, u, p):
        return u((0.75, 0.75))[0]

    def reference(self, t):
        return -0.0780739691918

    def __str__(self):
        return "Aneurysm (3D)"

# Solve problem
problem = Aneurysm()
problem.parameters["solver_parameters"]["plot_solution"] = True
problem.parameters["solver_parameters"]["save_solution"] = False
u, p = problem.solve()

# Check error
e = problem.functional(u, p) - problem.reference(0.5)
print "Error is", e

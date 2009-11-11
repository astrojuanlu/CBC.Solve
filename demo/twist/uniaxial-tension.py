__author__ = "Harish Narayanan"
__copyright__ = "Copyright (C) 2009 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from cbc.twist import *

class UniaxialTension(HyperelasticityProblem):

    def mesh(self):
        n = 8
        return UnitCube(n, n, n)

    def end_time(self):
        return 1.0

    def time_step(self):
        return 0.1

    def is_dynamic(self):
        return False

    def dirichlet_conditions(self, vector):
        clamp = Expression(("0.0", "0.0", "0.0"), V = vector)
        return [clamp]

    def dirichlet_boundaries(self):
        left = "x[0] == 0.0"
        return [left]

    def neumann_conditions(self, vector):
        T2 = Expression(("force*t", "0.0", "0.0"), V = vector)
        T2.force = 1.0
        return [T2]

    def neumann_boundaries(self):
        right = "x[0] == 1.0"
        return [right]

    def body_force(self, vector):
        B = Expression(("0.0", "0.0", "0.0"), V = vector)
        return B

    def material_model(self):
        mu    = 3.8461
        lmbda = 5.76
        material = StVenantKirchhoff([mu, lmbda])
        return material

    def __str__(self):
        return "A hyperelastic cube being pulled in one direction"

# Setup and solve problem
problem = UniaxialTension()
print problem
problem.solve()
interactive()

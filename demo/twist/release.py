__author__ = "Harish Narayanan"
__copyright__ = "Copyright (C) 2009 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from cbc.twist import *
from numpy import array, loadtxt

class Release(HyperelasticityProblem):

    def mesh(self):
        n = 8
        return UnitCube(n, n, n)

    def end_time(self):
        return 10.0

    def time_step(self):
        return 2.e-3

    def is_dynamic(self):
        return True

    def reference_density(self, scalar):
        return 1.0

    def initial_conditions(self, vector):
        """Return initial conditions for displacement field, u0, and
        velocity field, v0""" 
        u0 = Function(vector)
        u0.vector()[:] = loadtxt("twisty.txt")[:]
        v0 = Expression(("0.0", "0.0", "0.0"), V = vector)
        return u0, v0

    def dirichlet_conditions(self, vector):
        clamp = Expression(("0.0", "0.0", "0.0"), V = vector)
        return [clamp]

    def dirichlet_boundaries(self):
        left = "x[0] == 0.0"
        return [left]

    def material_model(self):
        mu    = 3.8461
        lmbda = 5.76
        material = StVenantKirchhoff([mu, lmbda])
        return material

    def __str__(self):
        return "A prestrained hyperelastic cube being let go"

# Setup and solve problem
problem = Release()
print problem
problem.solve()
interactive()

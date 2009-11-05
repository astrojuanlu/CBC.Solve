__author__ = "Harish Narayanan"
__copyright__ = "Copyright (C) 2009 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from cbc.twist import *
from numpy import array, loadtxt

class UniaxialTension(HyperelasticityProblem):

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

    def boundary_conditions(self, t, vector):
        clamp = Expression(("0.0", "0.0", "0.0"), V = vector)
        left = compile_subdomains(["(fabs(x[0]) < DOLFIN_EPS) && on_boundary"])
        bcu0 = DirichletBC(vector, clamp, left)
        return [bcu0]

    def body_force(self, t, vector):
        B = Expression(("0.0", "0.0", "0.0"), V = vector)
        B.t = t
        return B

    def surface_force(self, t, vector):
        # Need to specify Neumann boundary somewhere
        T = Expression(("0.0", "0.0", "0.0"), V = vector)
        T.t = t
        return T

    def material_model(self):
        mu    = 3.8461
        lmbda = 5.76
        material = LinearElastic([mu, lmbda])
        return material

    def info(self):
        return "A prestrained hyperelastic cube being let go"

# Setup and solve problem
problem = UniaxialTension()
print problem.info()
problem.solve()
interactive()

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

    def boundary_conditions(self, t, vector):
        clamp = Expression(("0.0", "0.0", "0.0"), V = vector)
        pull  = Expression(("ex*t", "0.0", "0.0"), V = vector)
        pull.ex = 0.5
        pull.t = t
        left, right = compile_subdomains(["(fabs(x[0]) < DOLFIN_EPS) && on_boundary",
                                          "(fabs(x[0] - 1.0) < DOLFIN_EPS) && on_boundary"]) 
        bcu0 = DirichletBC(vector, clamp, left)
        bcu1 = DirichletBC(vector, pull, right)
        return [bcu0, bcu1]

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
        c1 = 0.162
        c2 = 5.9e-3
        material = MooneyRivlin([c1, c2])
        return material

    def info(self):
        return "A hyperelastic cube being pulled in one direction"

# Setup and solve problem
problem = UniaxialTension()
problem.solve()
interactive()

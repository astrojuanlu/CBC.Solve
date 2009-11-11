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
        pull = Expression(("ex*t", "0.0", "0.0"), V = vector)
        pull.ex = 0.5

        return [clamp, pull]

    def dirichlet_boundary(self):
        left, right = ["x[0] < DOLFIN_EPS",
                       "x[0] > 1.0 - DOLFIN_EPS"]
        return [left, right]

#     def neumann_boundary_conditions(self, vector):
#         T = Expression(("force*t", "0.0", "0.0"), V = vector)
#         T.force = 1.0
#         return T

#     def neumann_boundary(self):
#         right = "x[0] > 1.0 - DOLFIN_EPS"
#         return [right]

#     def body_force(self, vector):
#         B = Expression(("0.0", "0.0", "0.0"), V = vector)
#         return B

#     def surface_force(self, vector):
#         # Need to specify Neumann boundary somewhere
        
# #         s = sigma(uf, pf)
# #         P = det(1 + Grad(us))*sigma(uf, pf)*inv((1+Grad(us)).T)
# #         P = inverse_piola_transform(s)
# #         N = facetnormal(self.mesh)
# #         f = P*N
# #         T.fx = f[0]
# #         T.fy = f[1]
# #         T.fz = f[2]

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

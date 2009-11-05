__author__ = "Harish Narayanan"
__copyright__ = "Copyright (C) 2009 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from cbc.twist import *

class Twist(StaticHyperelasticityProblem):

    def mesh(self):
        n = 8
        return UnitCube(n, n, n)

    def boundary_conditions(self, vector):
        clamp = Expression(("0.0", "0.0", "0.0"), V = vector)
        twist = Expression(("0.0",
                            "y0 + (x[1] - y0) * cos(theta) - (x[2] - z0) * sin(theta) - x[1]",
                            "z0 + (x[1] - y0) * sin(theta) + (x[2] - z0) * cos(theta) - x[2]"),
                           defaults = dict(y0 = 0.5, z0 = 0.5, theta = pi / 3), V = vector)
        left, right = compile_subdomains(["(fabs(x[0]) < DOLFIN_EPS) && on_boundary",
                                          "(fabs(x[0] - 1.0) < DOLFIN_EPS) && on_boundary"]) 
        bcu0 = DirichletBC(vector, clamp, left)
        bcu1 = DirichletBC(vector, twist, right)
        return [bcu0, bcu1]

    def body_force(self, vector):
        B = Expression(("0.0", "0.0", "0.0"), V = vector)
        return B

    def surface_force(self, vector):
        # Need to specify Neumann boundary somewhere
        T = Expression(("0.0", "0.0", "0.0"), V = vector)
        return T

    def material_model(self):
        # Material parameters can either be numbers or spatially
        # varying fields. For example,
        mu       = 3.8461
        lmbda    = Expression("x[0]*5.8 + (1 - x[0])*5.7", V = FunctionSpace(self.mesh(), "CG", 1))
        C10 = 0.171; C01 = 4.89e-3; C20 = -2.4e-4; C30 = 5.e-4

        # It is also easy to switch material models. Uncomment one of
        # the following lines to see a particular material's response.
        #material = MooneyRivlin([mu/2, mu/2])
        #material = StVenantKirchhoff([mu, lmbda])
        #material = neoHookean([mu])
        #material = Isihara([C10, C01, C20])
        material = Biderman([C10, C01, C20, C30])

        return material

    def info(self):
        return "A hyperelastic cube twisted by 60 degrees"

# Setup and solve problem
problem = Twist()
u = problem.solve()

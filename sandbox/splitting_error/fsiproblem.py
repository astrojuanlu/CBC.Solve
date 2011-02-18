__author__ = "Kristoffer Selim andAnders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2011-02-18

from dolfin import *
from numpy import array, append
from cbc.common import CBCProblem

from fsisolver import FSISolver

class FSI(CBCProblem):
    "Base class for all FSI problems"

    def __init__(self, mesh):
        "Create FSI problem"

        # Initialize base class
        CBCProblem.__init__(self)

        # Create solver
        self.solver = FSISolver(self)

        # Set up parameters
        self.parameters = Parameters("problem_parameters")
        self.parameters.add(self.solver.parameters)

        # Create submeshes and mappings
        self.init_meshes(mesh)

    def solve(self):
        "Solve and return computed solution (u_F, p_F)"

        # Update solver parameters
        self.solver.parameters.update(self.parameters["solver_parameters"])

        # Call solver
        return self.solver.solve()

    def init_meshes(self, Omega):

        # Set global mesh
        self.Omega = Omega


    # Return the fluid mesh as Omege
    def fluid_mesh(self):
        "Return mesh for fluid domain"
        return self.Omega


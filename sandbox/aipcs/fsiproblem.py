__author__ = "Kristoffer Selim andAnders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2011-02-28

from dolfin import *
from numpy import array, append
from cbc.common import CBCProblem

from fsisolver import FSISolver
from parameters import default_parameters, read_parameters

class FSI(CBCProblem):
    "Base class for all FSI problems"

    def __init__(self, mesh):
        "Create FSI problem"

        # Initialize base class
        CBCProblem.__init__(self)

        # Store original mesh
        self._original_mesh = mesh
        self.Omega = None

    def solve(self, parameters=default_parameters()):
        "Solve and return computed solution (u_F, p_F, U_S, P_S, U_M, P_M)"

        # Create submeshes and mappings (only first time)
        if self.Omega is None:

            # Refine original mesh
            mesh = self._original_mesh
            for i in range(parameters["num_initial_refinements"]):
                mesh = refine(mesh)

            # Initialize meshes
            self.init_meshes(mesh, parameters)

        # Create solver
        solver = FSISolver(self)

        # Solve
        return solver.solve(parameters)

    def init_meshes(self, Omega, parameters):
        "Create mappings between submeshes"

        info("Extracting fluid and structure submeshes")

        # Set global mesh
        self.Omega = Omega

    def mesh(self):
        "Return mesh for full domain"
        return self.Omega

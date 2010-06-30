__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2010-06-30

from dolfin import *
from cbc.common import CBCProblem

from fsisolver import FSISolver

class FSI(CBCProblem):
    "Base class for all FSI problems"

    def __init__(self, parameters=None):
        "Create FSI problem"

        # Initialize base class
        CBCProblem.__init__(self)

        # Create solver
        self.solver = FSISolver(self)

        # Set up parameters
        self.parameters = Parameters("problem_parameters")
        self.parameters.add(self.solver.parameters)

    def solve(self, tolerance):
        "Solve and return computed solution (u_F, p_F, U_S, P_S, U_M, P_M)"

        # Update solver parameters
        self.solver.parameters.update(self.parameters["solver_parameters"])

        # Call solver
        return self.solver.solve(tolerance)

    def time_step(self):
        "Return default time step (will be changed by adaptive algorithm)"
        return 1.0

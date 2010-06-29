__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2010-06-29

from fsiproblem import *

class ChannelWithFlap(FSI):

    # FIXME: Define problem data here

    def end_time(self):
        return 1.0

    def __str__(self):
        return "Channel with flap FSI problem"

# Solve problem
problem = ChannelWithFlap()
problem.parameters["solver_parameters"]["plot_solution"] = True
u_F, p_F, U_S, P_S, U_M, P_M = problem.solve(1e-3)

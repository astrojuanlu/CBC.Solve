"""This script runs a convergence test for the analytic test case
using uniform refinement in space and time, solving only the primal
problem."""

__author__ = "Anders Logg"
__copyright__ = "Copyright (C) 2012 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# First added:  2012-03-05
# Last changed: 2012-05-03

from cbc.swing import *

# Set up parameters
p = default_parameters()
p["solve_primal"] = True
p["solve_dual"] = False
p["estimate_error"] = False
p["plot_solution"] = False
p["uniform_timestep"] = True
p["uniform_mesh"] = True
p["initial_timestep"] = 0.02
p["tolerance"] = 1e-16
p["fixedpoint_tolerance"] = 1e-14
p["max_num_refinements"] = 100

# Run
run_local("analytic", p, "convergence-test-primal")

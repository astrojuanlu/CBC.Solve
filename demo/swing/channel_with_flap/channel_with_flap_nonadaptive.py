"""This script runs the channel_with_flap demo without adaptivity and
plots the solution for quick testing."""

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from cbc.swing import *

# Set parameters
p = default_parameters()
p["solve_dual"] = False
p["estimate_error"] = False
p["plot_solution"] = True
p["initial_timestep"] = 0.01

# Run problem
run_local("channel_with_flap", p)

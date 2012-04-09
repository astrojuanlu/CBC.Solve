"""This script runs a convergence test for the channel with flap test
problem using uniform refinement in space and time."""

__author__ = "Anders Logg"
__copyright__ = "Copyright (C) 2012 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# First added:  2012-04-08
# Last changed: 2012-04-08

from cbc.swing import *

# Set up parameters
p = default_parameters()
p["output_directory"] = "results_channel_with_flap_convergence_test"
p["uniform_timestep"] = True
p["uniform_mesh"] = True
p["tolerance"] = 1e-16

# Run
run_local("channel_with_flap", p, "convergence_test")

"""This script runs a convergence test for the analytic test case using uniform
refinement in space and time."""

__author__ = "Anders Logg"
__copyright__ = "Copyright (C) 2012 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# First added:  2012-03-05
# Last changed: 2012-03-09

from cbc.swing.fsirun import *
from cbc.swing.parameters import *
from numpy import pi

# Set up parameters
p = default_parameters()
p["solve_dual"] = False
p["estimate_error"] = False
p["plot_solution"] = False
p["uniform_timestep"] = True
p["uniform_mesh"] = False
p["fixedpoint_tolerance"] = 1e-6

# Reference value
C = 0.1
M_0 = C / (24.0*pi)

# Run convergence study
k = 0.1
for n in range(6):

    print "Running convergence test with n = %d and k = %g" % (n, k)

    p["num_initial_refinements"] = n
    p["initial_timestep"] = k

    status, output = run_local("analytic", p)

    if status == 0:
        M = float(output.split("Integrated goal functional at T: ")[1].split("\n")[0])
        e = M - M_0
        print "M = %g  M_0 = %g  e = %g" % (M, M_0, e)
    else:
        print "Did not converge"

    k = 0.5*k

    print

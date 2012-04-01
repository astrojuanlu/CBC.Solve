"""This script runs a convergence test for the primal problem of the
analytic test case using uniform refinement in space and time."""

__author__ = "Anders Logg"
__copyright__ = "Copyright (C) 2012 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# First added:  2012-03-05
# Last changed: 2012-04-01

from cbc.swing.fsirun import *
from cbc.swing.parameters import *
from numpy import pi
from time import time

# Set up parameters
p = default_parameters()
p["solve_primal"] = True
p["solve_dual"] = True
p["estimate_error"] = True
p["plot_solution"] = False
p["uniform_timestep"] = True
p["uniform_mesh"] = False
p["fixedpoint_tolerance"] = 1e-8
p["output_directory"] = "results_analytic_convergence_test"

# Number of refinements
#num_refinements = 6
num_refinements = 3 # use for quick testing

# Reference value
C = 0.1
M_0 = C / (24.0*pi)

# Function for extracting results
def get_value(output, prefix):

    print output

    return float(output.split(prefix)[1].split("\n")[0])

# Run convergence study
k = 0.01
for n in range(num_refinements):

    print "Running convergence test with n = %d and k = %g" % (n, k)

    # Set parameters
    p["num_initial_refinements"] = n
    p["initial_timestep"] = k

    # Run test
    cpu_time = time()
    status, output = run_local("analytic", p, n)
    cpu_time = time() - cpu_time

    # Check for convergence
    if status != 0:
        print "Did not converge"
        break

    # Extract results
    M = get_value(output, "Integrated goal functional at T: ")
    E = get_value(output, "E_tot = ")
    E_h = get_value(output, "E_h = ")
    E_k = get_value(output, "E_k = ")
    E_c = get_value(output, "E_c = ")
    E_0 = get_value(output, "E_0 = ")
    e = M - M_0
    I = E / e
    I_0 = E_0 / e

    # Report results
    print "Completed in %g seconds" % cpu_time
    print
    print "M   = %g" % M
    print "M_0 = %g" % M_0
    print "e   = %g" % e
    print "E_0 = %g" % E_0
    print "E   = %g" % E
    print "E_h = %g" % E_h
    print "E_k = %g" % E_k
    print "E_c = %g" % E_c
    print "I   = %g" % I
    print "I_0 = %g" % I
    print

    # Reduce size of time step
    k = 0.5*k

"Script for plotting functional values for all result directories"

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

import os, glob
from pylab import *

def read(filename):
    return zip(*[row.split(" ") for row in open(filename).read().split("\n") if " " in row])

# Extract values
functional_values = []
integrated_values = []
for directory in glob.glob("results-*"):
    print "Extracting values from %s" % directory
    filename = "%s/goal_functional_final.txt" % directory
    if os.path.isfile(filename):
        levels, functionals, integrated_functionals = read(filename)
        print "Found %d values" % len(levels)
        functional_values.append(levels)
        functional_values.append(functionals)
        integrated_values.append(levels)
        integrated_values.append(integrated_functionals)

# Plot functional values at t = T
figure(1)
plot(*functional_values)
xlabel("Refinement level")
ylabel("Functional value")
title("Functional values at end time")
grid(True)

# Plot integrated functional values
figure(2)
plot(*integrated_values)
xlabel("Refinement level")
ylabel("Functional value")
title("Integrated functional values")
grid(True)

show()

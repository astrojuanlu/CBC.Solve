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
legends = []
for directory in glob.glob("results-*"):
    print "Extracting values from %s" % directory

    # Get data
    filename = "%s/goal_functional_final.txt" % directory
    if os.path.isfile(filename):
        levels, functionals, integrated_functionals = read(filename)
        print "Found %d values" % len(levels)
        functional_values.append(levels)
        functional_values.append(functionals)
        integrated_values.append(levels)
        integrated_values.append(integrated_functionals)

        # Get legend
        filename = "%s/application_parameters.xml" % directory
        if os.path.isfile(filename):
            row = [row for row in open(filename).read().split("\n") if "description" in row][0]
            description = row.split('value="')[1].split('"')[0]
            legends.append(description)

# Plot functional values at t = T
figure(1)
plot(*functional_values, marker='o')
xlabel("Refinement level")
ylabel("Functional value")
title("Functional values at end time")
legend(legends)
grid(True)

# Plot integrated functional values
figure(2)
plot(*integrated_values, marker='o')
xlabel("Refinement level")
ylabel("Functional value")
title("Integrated functional values")
legend(legends)
grid(True)

show()

"Script for plotting functional values for all result directories"

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

import os, glob
from pylab import *

def read(filename):
    return zip(*[row.split(" ") for row in open(filename).read().split("\n") if " " in row])

# Check all results directories
functionals = []
legends = []
dofs = []
for directory in glob.glob("results-*"):
    print "Extracting values from %s" % directory

    # Check for data
    filename = "%s/goal_functional_final.txt" % directory
    if not os.path.isfile(filename): continue

    # Get functional values
    f = read(filename)[2]
    print "Found %d values" % len(f)
    functionals.append(f)

    # Get number of dofs
    filename = "%s/num_dofs.txt" % directory
    d = read(filename)[2]
    dofs.append(d)

    # Get legend
    filename = "%s/application_parameters.xml" % directory
    row = [row for row in open(filename).read().split("\n") if "description" in row][0]
    l = row.split('value="')[1].split('"')[0]
    legends.append(l)

# Group values
plotvals = []
for (x, y) in zip(dofs, functionals):
    if not len(x) == len(y):
        raise RuntimeError, ("Data size mismatch: %d %d" % (len(x), len(y)))
    plotvals.append(x)
    plotvals.append(y)

# Plot functional values
figure(1)
subplot(2, 1, 1)
semilogx(*plotvals, marker='o')
xlabel("Refinement level")
ylabel("Functional value")
title("Convergence of functional value")
legend(legends, bbox_to_anchor=(-0.1, -0.3), loc=2, borderaxespad=0)
grid(True)

show()

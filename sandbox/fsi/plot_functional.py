"Script for plotting functional values for all result directories"

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

import os, glob
from numpy import argsort, array
from pylab import *

# Set reference value
#reference_value = None
reference_value = 0.003656

def read(filename):
    return zip(*[[x for x in row.split(" ") if len(x) >0] for row in open(filename).read().split("\n") if " " in row and len(row) > 0])

def read_float(filename):
    return array([[float(x) for x in column] for column in read(filename)])

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
    f = read_float(filename)[2]
    print "Found %d values" % len(f)
    functionals.append(f)

    # Get number of dofs
    filename = "%s/num_dofs.txt" % directory
    d = read_float(filename)[2]
    dofs.append(d)

    # Get legend
    filename = "%s/application_parameters.xml" % directory
    row = [row for row in open(filename).read().split("\n") if "description" in row][0]
    l = row.split('value="')[1].split('"')[0]
    legends.append(l)

# Sort values by legend
indices = argsort(legends)
legends = [legends[i] for i in indices]

 # Group values
plotvals = []
for i in indices:
    x = dofs[i]
    if reference_value is None:
        y = functionals[i]
    else:
        y = [abs(value - reference_value) for value in functionals[i]]
    if not len(x) == len(y):
        raise RuntimeError, ("Data size mismatch: %d %d" % (len(x), len(y)))
    plotvals.append(x)
    plotvals.append(y)

# Set fontsize for legend
rcParams.update({'legend.fontsize': 11})

# Plot functional values
max_plots = 9
for i in range((len(legends) - 1) / max_plots + 1):
    xy = plotvals[2*i*max_plots:2*(i + 1)*max_plots]
    figure(i)
    subplot(2, 1, 1)
    if reference_value is None:
        ylabel("Functional value")
        semilogx(*xy, marker='o')
    else:
        ylabel("Error in functional value (absolute value)")
        loglog(*xy, marker='o')
    xlabel("Number of dofs (in space)")
    title("Convergence of functional value")
    legend(legends[i*max_plots:(i + 1)*max_plots], bbox_to_anchor=(-0.1, -0.3), loc=2, borderaxespad=0)
    grid(True)

show()

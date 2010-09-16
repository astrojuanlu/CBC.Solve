"This module plots data from the adaptive algorithm"

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2010-09-16

from pylab import *

def plot_time_steps():
    "Plot adaptive time steps"

    # Read file and get number of levels
    lines = open("timesteps.txt").read().split("\n")[:-1]
    num_levels = max(int(l.split(" ")[0]) for l in lines) + 1

    # Plot all adaptive time step sequences
    for level in range(num_levels):
        print "Plotting time steps for level %d" % level

        # Extract data
        level_lines = [l for l in lines if int(l.split(" ")[0]) == level]
        t = [float(l.split(" ")[1]) for l in level_lines]
        k = [float(l.split(" ")[2]) for l in level_lines]
        R = [float(l.split(" ")[3]) for l in level_lines]

        # Plot
        figure(level)
        subplot(2, 1, 1); grid(True); plot(t, k, '-o')
        ylabel("$k$"); title("Refinement level %d" % level)
        subplot(2, 1, 2); grid(True); plot(t, R, '-o')
        ylabel('$R_k$'); xlabel("$t$")

    show()

plot_time_steps()

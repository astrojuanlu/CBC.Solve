"This module plots the Galerkin stability factor as a function of time"

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2010-10-05

from pylab import *

def plot_stability_factor():

    lines = open("stability_factor.txt").read().split("\n")[:-1]
    level_lines = [l for l in lines]
    t = [float(l.split(" ")[0]) for l in level_lines]
    SGT = [float(l.split(" ")[1]) for l in level_lines]

    figure()
    plot()
    grid(True); plot(t, SGT, '-o')
    xlabel('$t$', fontsize=36);
    ylabel('$\mathcal{S}(T)$',fontsize=36 ); 
    title("Galerkin stability factor ",  fontsize=26)
    show()

plot_stability_factor()



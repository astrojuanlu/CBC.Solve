"This module provides various utility functions"

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2010-09-10

from dolfin import CellFunction

def array_to_meshfunction(x, mesh):
    "Convert array x to cell function on Omega"
    f = CellFunction("double", mesh)
    if not f.size() == x.size:
        raise RuntimeError, "Size of vector does not match number of cells."
    for i in range(x.size):
        f[i] = x[i]
    return f

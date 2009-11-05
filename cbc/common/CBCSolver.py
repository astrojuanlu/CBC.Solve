__author__ = "Anders Logg"
__copyright__ = "Copyright (C) 2009 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2009-11-05

from dolfin import error

class CBCSolver:
    "Base class for all solvers"

    def solve():
        error("solve() function not implemented by solver.")

    def __str__():
        error("__str__ not implemented by solver.")

__author__ = "Anders Logg"
__copyright__ = "Copyright (C) 2009 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2009-11-06

from dolfin import error

class CBCProblem:
    "Base class for all problems"

    def __str__():
        error("__str__ not implemented by problem.")

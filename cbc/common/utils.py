"This module provides a set of common utility functions."

__author__ = "Anders Logg"
__copyright__ = "Copyright (C) 2009 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from dolfin import PeriodicBC

def is_periodic(bcs):
    "Check if boundary conditions are periodic"
    return all(isinstance(bc, PeriodicBC) for bc in bcs)

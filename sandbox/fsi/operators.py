"This module defines special operators for the dual problem and residuals."

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2010-08-11

from dolfin import *

def F(v):
    "Return deformation gradient"
    I = Identity(v.geometric_dimension())
    return I + grad(v)

def J(v):
    "Return determinant of deformation gradient"
    return det(F(v))

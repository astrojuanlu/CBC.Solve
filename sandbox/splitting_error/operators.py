"This module defines special operators for the dual problem and residuals."

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2011-02-08

from dolfin import *

# Define identity matrix in 2D
I = Identity(2)

# Define the symetric gradient
def epsilon(v):
        return 0.5*(grad(v) + grad(v).T)

# Define the fluid stress tensor
def sigma(v,q,mu):
    return  2*mu*epsilon(v) - q*I
    

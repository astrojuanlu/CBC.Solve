"""My own Newton Solver, used in solving Nonlinear PDE problems"""

__author__ = "Gabriel Balaban"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from dolfin import *


class FSIMatrixMapper(object):
   def __init__(self,fluiddomain,structuredomain):
       

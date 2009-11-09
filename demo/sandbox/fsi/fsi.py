# A simple fsi solver

from cbc.flow import *
from cbc.twist import *
from fsi_problems import FluidProblem
from fsi_problems import StructureProblem

# Create fluid solver
fluid_problem = FluidProblem() 

# Solve fluid problem
fluid_problem.solve()
# Edit the Static NS solver!!!!

# Create structure solver
structure_problem = StructureProblem()

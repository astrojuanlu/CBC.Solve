# A simple fsi solver

from cbc.flow import *
from cbc.twist import *
from fsi_problems import *

# Create fluid problem (single time-step)
fluid_problem = FluidProblem() 

dt = 0.05
t = dt

# Solve fluid problem
for j in range(10):
    u1, p1 = fluid_problem.solve()
    plot(u1)
    plot(p1)
    t = t + dt

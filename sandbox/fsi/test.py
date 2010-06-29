from residual import *

# Construcut residual
R = Residual()

t = 0.5
dt = 0.02

R_k = R.compute_residuals(t, dt)
print "|| R_K || = ", R_k

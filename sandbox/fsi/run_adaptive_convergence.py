from fsirun import *

# Set parameters
p = default_parameters()
p["solve_primal"] = True
p["solve_dual"] = True
p["estimate_error"] = True
p["uniform_timestep"] = True
p["initial_timestep"] = 0.01
p["tolerance"] = 1e-6

# Run problem
run_local("modified_pressure_driven_cavity", p)

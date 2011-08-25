from fsirun import *

case = 1

# Set common parameters
p = default_parameters()
p["solve_primal"] = True
p["solve_dual"] = False
p["estimate_error"] = False
p["uniform_timestep"] = True
p["tolerance"] = 1e-7
p["initial_timestep"] = 0.01
p["goal_functional"] = case

# Run problem
run_local("lid_driven_cavity", parameters=p, case=case)

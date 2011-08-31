from fsirun import *

case = 2

# Set common parameters
p = default_parameters()
p["solve_primal"] = True
p["solve_dual"] = True
p["estimate_error"] = True
p["tolerance"] = 1e-7
p["initial_timestep"] = 0.01
p["goal_functional"] = case

# Run problem
run_local("channel_with_flap", parameters=p, case=1)

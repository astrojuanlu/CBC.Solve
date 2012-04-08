from fsirun import *

# Set common parameters
p = default_parameters()
p["solve_primal"] = True
p["solve_dual"] = True
p["estimate_error"] = True
p["initial_timestep"] = 0.025
p["uniform_timestep"] = True
p["max_num_refinements"] = 0

# Run problem
run_local("channel_with_flap", parameters=p, case="test")

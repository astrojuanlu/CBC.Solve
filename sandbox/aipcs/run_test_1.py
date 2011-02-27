from fsirun import *

# Set common parameters
p = default_parameters()
p["solve_primal"] = False
p["solve_dual"] = False
p["estimate_error"] = True
p["uniform_timestep"] = True
p["num_initial_refinements"] = 1

problem = "channel_with_flap"

run_local(problem, parameters=p, case="level_1")

from fsirun import *

# Set common parameters
p = default_parameters()
p["solve_primal"] = False
p["solve_dual"] = True
p["estimate_error"] = False
p["tolerance"] = 1e-7
p["initial_timestep"] = 0.01
#p["max_num_refinements"] = 0
p["goal_functional"] = 0
#p["plot_solution"] = True

# Run problem
run_local("channel_with_flap", parameters=p, case=0)

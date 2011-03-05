from fsirun import *

# Set common parameters
p = default_parameters()
p["estimate_error"] = False
p["tolerance"] = 1e-5
p["initial_timestep"] = 0.01
p["max_num_refinements"] = 0
p["plot_solution"] = True

# Run problem
run_local("flow_past_cylinder", parameters=p, case=0)

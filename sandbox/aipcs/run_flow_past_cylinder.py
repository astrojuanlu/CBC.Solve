from fsirun import *

# Set common parameters
p = default_parameters()
p["solve_dual"] = False
p["estimate_error"] = False
p["tolerance"] = 1e-5
p["initial_timestep"] = 0.1
p["max_num_refinements"] = 0
p["plot_solution"] = True

# Run problem
run_local("flow_past_cylinder", parameters=p, case=0)

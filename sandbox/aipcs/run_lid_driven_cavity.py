from fsirun import *

case = 2

# Set common parameters
p = default_parameters()
p["solve_primal"] = True
p["solve_dual"] = True
p["estimate_error"] = True
p["uniform_timestep"] = False
p["tolerance"] = 1e-7
p["initial_timestep"] = 0.05
p["marking_fraction"] =  0.3
#p["refinement_algorithm"] = "regular_cut"
p["refinement_algorithm"] = "recursive_bisection"

# Run problem
run_local("lid_driven_cavity", parameters=p, case=case)

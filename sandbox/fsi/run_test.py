from fsirun import *

# Set common parameters
p = default_parameters()
p["solve_primal"] = True
p["solve_dual"] = True
p["estimate_error"] = True
p["tolerance"] = 1e-5
p["initial_timestep"] = 0.05

# Run problem
run_local(problem, parameters=p, case="test")

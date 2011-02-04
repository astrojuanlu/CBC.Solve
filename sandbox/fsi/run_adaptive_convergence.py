from fsirun import *

# Set parameters
p = default_parameters()
p["solve_primal"] = True
p["solve_dual"] = False
p["estimate_error"] = False

# Run problem
run_local("modified_pressure_driven_cavity", p)

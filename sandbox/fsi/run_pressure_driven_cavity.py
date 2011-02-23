from fsirun import *

# Set common parameters
p = default_parameters()
p["solve_primal"] = True
p["solve_dual"] = True
p["estimate_error"] = True
p["uniform_timestep"] = True
p["tolerance"] = 1e-6
p["initial_timestep"] = 0.01
p["structure_element_degree"] = 1

# Choose test case
case = 1

# Test cases
if case is 1:
    p["description"] = "adaptive, k = 0.01 (regular)"
    p["refinement_algorithm"] = "regular_cut"
elif case is 2:
    p["description"] = "adaptive, k = 0.01 (regular, fixed fraction)"
    p["refinement_algorithm"] = "regular_cut"
    p["dorfler_marking"] = False
elif case is 3:
    p["description"] = "adaptive, k = 0.01 (bisection)"
    p["refinement_algorithm"] = "recursive_bisection"
elif case is 4:
    p["description"] = "adaptive, k = 0.01 (bisection, fixed fraction)"
    p["refinement_algorithm"] = "recursive_bisection"
    p["dorfler_marking"] = False
elif case is 5:
    p["description"] = "uniform, k = 0.01"
    p["uniform_mesh"] = True

# Run problem
run_local("pressure_driven_cavity", parameters=p, case=case)

# Script to run all test cases for the channel_with_flap problem on
# the bigblue server.

from fsirun import *

# Cases to investigate
#q_range = [1, 2]
q_range = [1]
k_range = [0.01, 0.005, 0.0025]
r_range = ["uniform", "recursive_bisection", "regular_cut"]
d_range = [None, True, False]
f_range = [None, 0.1, 0.2, 0.3, 0.4, 0.5]

# Iterate over cases
case = 0
for q in q_range:
    for k in k_range:
        for r in r_range:
            for d in d_range:
                for f in f_range:

                    # Handle d and f cases not relevant for uniform refinement
                    if r == "uniform" and not (d is None and f is None): continue
                    if r != "uniform" and (d is None or f is None): continue

                    # Set common parameters
                    p = default_parameters()
                    p["solve_primal"] = True
                    p["solve_dual"] = True
                    p["estimate_error"] = True
                    p["uniform_timestep"] = True
                    p["tolerance"] = 1e-8
                    p["fixedpoint_tolerance"] = 1e-12
                    p["max_num_refinements"] = 100

                    # Set parameters for current case
                    p["structure_element_degree"] = q
                    p["initial_timestep"] = k
                    if r == "uniform":
                        p["uniform_mesh"] = True
                        p["description"] = "q = %g k = %g (uniform)" % (q, k)
                    else:
                        p["refinement_algorithm"] = r
                        p["dorfler_marking"] = d
                        p["marking_fraction"] = f
                        if d:
                            p["description"] = "q = %g k = %g (%s, Dorfler %g)" % (q, k, r, f)
                        else:
                            p["description"] = "q = %g k = %g (%s, fixed fraction %g)" % (q, k, r, f)

                    print case, p["description"]

                    # Run problem
                    run_bb("channel_with_flap", parameters=p, case=case)

                    # Increase counter
                    case += 1

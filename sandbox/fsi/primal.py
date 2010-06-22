# A simple FSI problem involving a hyperelastic obstruction in a
# Navier-Stokes flow field. Lessons learnt from this exercise will be
# used to construct an FSI class in the future.

from pylab import *
from cbc.flow import *
from cbc.twist import *
from numpy import array, append
from problems import *
from common import *
from adaptivity import *

# Define the three problems
F = FluidProblem()
S = StructureProblem()
M = MeshProblem()

# Define problem parameters
plot_solution = False
store_vtu_files = False
store_bin_files = True
F.parameters["solver_parameters"]["plot_solution"] = False
F.parameters["solver_parameters"]["save_solution"] = False
F.parameters["solver_parameters"]["store_solution_data"] = False
S.parameters["solver_parameters"]["plot_solution"] = False
S.parameters["solver_parameters"]["save_solution"] = False
S.parameters["solver_parameters"]["store_solution_data"] = False

# Solve mesh equation (will give zero vector first time which corresponds to
# identity map between the current domain and the reference domain)
U_M = M.step(0.0)

# Create inital displacement vector
V0 = VectorFunctionSpace(Omega_S, "CG", 1)
v0 = Function(V0)
U_S_vector_old  = v0.vector()

# Create files for storing solution
file_u_F = File("u_F.pvd")
file_p_F = File("p_F.pvd")
file_U_S = File("U_S.pvd")
file_P_S = File("P_S.pvd")
file_U_M = File("U_M.pvd")
disp_vs_t = open("displacement_nx_dt_T_smooth"+ "_" + str(nx) + "_"  +  str(dt) + "_" + str(T) + "_"+ str(mesh_smooth), "w")
convergence_data = open("convergence_nx_dt_T_smooth" + "_" + str(nx)  +  "_"  +  str(dt) + "_" + str(T) +  "_" + str(mesh_smooth), "w")

# Storing of adaptive data
times = []
timesteps = []

# Time-stepping
t = dt
at_end = False
for t in t_range:
#while True:

    info("")
    begin("* Starting new time step")
    info_blue("  * t = %g (T = %g, dt = %g)" % (t, T, dt))

    # Fixed point iteration on FSI problem
    for iter in range(maxiter):

        info("")
        begin("* Starting nonlinear iteration")

        # Solve fluid equation
        begin("* Solving fluid sub problem (F)")
        u_F, p_F = F.step(dt)
        end()

        # Update fluid stress for structure problem
        begin("* Transferring fluid stresses to structure (F --> S)")
        Sigma_F = F.compute_fluid_stress(u_F, p_F, U_M)
        S.update_fluid_stress(Sigma_F)
        end()

        # Solve structure equation
        begin("* Solving structure sub problem (S)")
        structure_sol = S.step(dt)
        U_S, P_S = structure_sol.split(True)
        end()

        # Update structure displacement for mesh problem
        begin("* Transferring structure displacement to mesh (S --> M)")
        M.update_structure_displacement(U_S)
        end()

        # Solve mesh equation
        begin("* Solving mesh sub problem (M)")
        U_M = M.step(dt)
        end()

        # Update mesh displacement and mesh velocity
        begin("* Transferring mesh displacement to fluid (M --> S)")
        F.update_mesh_displacement(U_M)
        end()

        # Plot solutions
        if plot_solution:
           plot(u_F, title="Fluid velocity")
           plot(U_S, title="Structure displacement", mode="displacement")
           plot(U_M, title="Mesh displacement", mode="displacement")
           plot(F.w, title="Mesh velocity")

        # Compute residual
        U_S_vector_old.axpy(-1, U_S.vector())
        r = norm(U_S_vector_old)
        U_S_vector_old[:] = U_S.vector()[:]
        info("")

        # Check convergence
        if r < tol:
            info_green("    Residual = %g (tolerance = %g), converged after %d iterations" % (r, tol, iter + 1))
            end()
            break
        elif iter == maxiter - 1:
            raise RuntimeError, "FSI iteration failed to converge after %d iterations." % maxiter
        else:
            info_red("    Residual = %g (tolerance = %g), iteration %d" % (r, tol, iter + 1))
            end()

    # Compute displacement
    displacement = (1.0/structure_area)*assemble(U_S[0]*dx, mesh = U_S.function_space().mesh())

    # Move to next time step
    F.update(t)
    S.update()
    M.update(t)

    # FIXME: This should be done automatically by the solver
    F.update_extra()

    # Store solutions in .vtu format
    if store_vtu_files:
        file_u_F << u_F
        file_p_F << p_F
        file_U_S << U_S
        file_P_S << P_S
        file_U_M << U_M

    # Store raw data for displacement
    disp_vs_t.write(str(displacement) + " ,  " + str(t) + "\n")

    # Store primal vectors in .bin format
    if store_bin_files:
        primal_u_F.store(u_F.vector(), t)
        primal_p_F.store(p_F.vector(), t)
        primal_U_S.store(U_S.vector(), t)
        primal_P_S.store(P_S.vector(), t)
        primal_U_M.store(U_M.vector(), t)

    # Store time and time steps
    times.append(t)
    timesteps.append(dt)

    # Check if we have reached the end time
    if at_end:
        end()
        info("Finished time-stepping")
        break

    # FIXME: Compute these
    Rk = 1.0
    TOL = 0.1
    ST = 1.0

    # Compute new time step
    #(dt, at_end) = compute_timestep(Rk, ST, TOL, dt, t, T)

    end()

# Close file
disp_vs_t.close()

# Compute convergence indicators
end_displacement = (1.0/structure_area)*assemble(U_S[0]*dx, mesh = U_S.function_space().mesh())
end_velocity = u_F((4.0, 0.5))[0]
print "Functional 1 (displacement):", end_displacement
print "Functional 2 (velocity):    ", end_velocity

# Store info (some needs to be stored by hand... marked with ##)
convergence_data.write(str("==FLUID PARAMETERS==")+ "\n")
convergence_data.write(str("viscosity:  ") + str(F.viscosity()) + "\n")
convergence_data.write(str("density:    ") + str(F.density()) + "\n" + "\n")
convergence_data.write(str("==STRUCTURE PARAMETERS==")+ "\n")
convergence_data.write(str("density: ") + str(S.reference_density()) + "\n")
convergence_data.write(str("mu:      ") + str(0.15) + "\n") ##
convergence_data.write(str("lambda:  ") + str(0.25) + "\n" + "\n") ##
convergence_data.write(str("==MESH PARAMETERS==")+ "\n")
convergence_data.write(str("no. mesh smooth:    ") + str(mesh_smooth) + "\n")
convergence_data.write(str("alpha:              ") + str(M.alpha) + "\n")
convergence_data.write(str("mu:                 ") + str(M.mu) + "\n")
convergence_data.write(str("lambda:             ") + str(M.lmbda) + "\n" + "\n" + "\n" + "\n")
convergence_data.write(str("****MESH/TIME*****") +  "\n")
convergence_data.write(str("Mesh size (nx*ny):     ") + str(mesh.num_cells()) + "\n")
convergence_data.write(str("Min(hK):               ") + str(mesh.hmin()) + "\n")
convergence_data.write(str("End time T:            ") + str(T) + "\n")
convergence_data.write(str("Time step kn:          ") + str(dt) + "\n")
convergence_data.write(str("Tolerance (FSI f.p.):  ") + str(tol) + "\n" + "\n")
convergence_data.write(str("****INDICATORS****")+ "\n")
convergence_data.write(str("Functional 1 (displacement): ") + str(end_displacement) + "\n")
convergence_data.write(str("Functional 2 (velocity):     ") + str(end_velocity) + "\n")
convergence_data.close()

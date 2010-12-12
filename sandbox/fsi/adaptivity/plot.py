"This module plots data from the adaptive algorithm"

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2010-11-25

from pylab import *
from numpy import trapz, ones, abs
import sys 

print ""
print ""
print "*************************************"
print "Default: M=1, (Md, Mt, E, k, tol)=0"
print "*************************************"
print ""
print "M  = Goal Functional,                Md = Goal Functional vs # dof"
print "Mt = Goal Functional vs Time          E = Error Estimate"
print "k  = Time Steps & Residuals,        tol = FSI Tolerance & # iterations"
print ""
print ""

# Define default plot settings
plot_time_step       = 0
plot_FSI_tol         = 0
plot_goal_vs_level   = 1
plot_goal_vs_dofs    = 0
plot_goal_vs_time    = 0
plot_error_estimate  = 0


# Get command-line parameters
for arg in sys.argv[1:]:
    if not "=" in arg: continue
    key, val = arg.split("=")

    if key == "k":
         plot_time_step= int(val)
    elif key == "tol":
        plot_FSI_tol = int(val)
    elif key == "M":
        plot_goal_level = int(val)
    elif key == "Mt":
        plot_goal_vs_time = int(val)
    elif key == "E":
        plot_error_estimate = int(val)
    elif key == "Md":
        plot_goal_vs_dofs = int(val)

# Define plots
def plots():

    # Read files (these files are always created)
    lines_iter =  open("no_iterations.txt").read().split("\n")[:-1]
    lines_tol  =  open("fsi_tolerance.txt").read().split("\n")[:-1]
    lines_goal =  open("goal_functional.txt").read().split("\n")[:-1]
    lines_dofs =  open("num_dofs.txt").read().split("\n")[:-1]

    # Determine the number of refinement levels to plot
    num_levels = max(int(l_iter.split(" ")[0]) for l_iter in lines_iter) + 1

    # Plot time step sequences 
    if plot_time_step == True:
        
        # Read file (only created when an adaptive time step is used)
        lines_time = open("timesteps.txt").read().split("\n")[:-1]

        for level in range(num_levels):
            print "Plotting time steps (k=1) for level %d" % level

            # Extract data for time steps
            level_lines_time = [l_time for l_time in lines_time if int(l_time.split(" ")[0]) == level]
            t   =  [float(l_time.split(" ")[1]) for l_time in level_lines_time]
            k   =  [float(l_time.split(" ")[2]) for l_time in level_lines_time]
            R   =  [float(l_time.split(" ")[3]) for l_time in level_lines_time]

            # Plot time step and time residual
            figure(level)
            subplot(2, 1, 1); grid(True); plot(t, k, '-g',linewidth=4)
            ylabel("$k_n(t)$", fontsize=30); title("Time steps & residual,  level %d" %level, fontsize=30)
            subplot(2, 1, 2); grid(True); plot(t, R, '-r', linewidth=4)
            ylabel('$|r_k|$', fontsize=30)
            xlabel("$t$", fontsize=30)

    # Plot FSI tolerance and number of FSI iterations
    if plot_FSI_tol == True:

        for level in range(num_levels):
            print "Plotting FSI tolerance and no. of itrations (tol=1) for level %d" % level

            # Extract data for FSI tolerance 
            level_lines_tol = [l_tol for l_tol in lines_tol if int(l_tol.split(" ")[0]) == level]
            t_tol = [float(l_tol.split(" ")[1]) for l_tol in level_lines_tol]
            tol = [float(l_tol.split(" ")[2]) for l_tol in level_lines_tol]

            # Extract data for no. of iterations
            level_lines_iter = [l_iter for l_iter in lines_iter if int(l_iter.split(" ")[0]) == level]
            t_iter = [float(l_iter.split(" ")[1]) for l_iter in level_lines_iter]
            iter = [float(l_iter.split(" ")[2]) for l_iter in level_lines_iter]
            
            # Plot FSI tolerance and no. of FSI iterations
            figure((level + 100)) 
            subplot(2, 1, 1); grid(True); plot(t_tol, tol, '-k', linewidth=4)
            ylabel("$TOL_{fSM}$", fontsize=30); title("FSI tolerance & # iter., level %d" %level, fontsize=30)
            subplot(2, 1, 2); grid(False); 
            
            # FIXME: axhspan/vlines  do not work on BB
#             axhspan(0.0, max(iter), xmin=0, xmax=max(t_iter), color='w')
#             vlines(t_iter, iter, 1.0, color='k', linestyles='-',  linewidth=1.5)
            plot(t_iter, iter, '-or', linewidth=2); grid(True)
            ylabel("#iter.", fontsize=30)
            xlabel("$t$", fontsize=30)

    # Process goal functional data
    for level in range(num_levels):
           
        # Extract data for goal functional
        level_lines_goal = [l_goal for l_goal in lines_goal if int(l_goal.split(" ")[0]) == level]
        t_goal = [float(l_goal.split(" ")[1]) for l_goal in level_lines_goal]
        M = [float(l_goal.split(" ")[2]) for l_goal in level_lines_goal]

        # Extract number of dofs
        level_lines_dofs = [l_dofs for l_dofs in lines_dofs if int(l_dofs.split(" ")[0]) == level]
        dofs = [float(l_dofs.split(" ")[1]) for l_dofs in level_lines_dofs]
         
        # Compute integral goal functional
        M_ave = trapz(t_goal, M)

        # Write M_ave to file 
        f = open("M_ave.txt", "w")
        f.write("%d %g \n" % (level, M_ave))
        f.close()

        if plot_goal_vs_time == True:
            print "Plotting goal functional vs time (Mt=1) for level %d" % level

            # Plot goal functional as a function of time
            figure((level + 200))
            plot(t_goal, M, '-m', linewidth=4); grid(True)
            title("Goal Functional vs time, level %d" %level, fontsize=30)
            ylabel('$\mathcal{M}^t$',fontsize=36); 
            xlabel("$t$", fontsize=30)

        # Extract data for integrated goal functional
        lines_M = open("M_ave.txt").read().split("\n")[:-1]
        level_lines_M = [l_M for l_M in lines_M]
        ref_level = [float(l_M.split(" ")[0]) for l_M in level_lines_M]
        M_int = [float(l_M.split(" ")[1]) for l_M in level_lines_M]
            
    # Plot integrated goal functional vs refinement level
    if plot_goal_vs_level == True:
        print "Plotting goal functional vs refinenment level (M=1)"

        # FIXME: Add reference values
        figure((level + 300))
        title("Goal Functional", fontsize=30)
        plot(ref_level, M_int, '-dk'); grid(True)
        ylabel('$\int_0^T \mathcal{M}^t dt$',fontsize=36); 
        xlabel("Refinment level", fontsize=30)

    # Plot integrated goal functional vs number of dofs
    if plot_goal_vs_dofs == True:
        print "Plotting goal functional vs #dofs (Md=1)"
        
        # FIXME: Add reference values
        figure((level + 900))
        title("Convergence of Goal Functional", fontsize=30)
        semilogx(dofs, abs(M_int), '-dk'); grid(True)
        ylabel('$\int_0^T \mathcal{M}^t dt$',fontsize=36); 
        xlabel("# Dofs", fontsize=30)
        legend(["Adaptive"], loc='best')

    # Plot error estimate
    if plot_error_estimate == True:
        print "Plotting error estimates (E=1)"

        # Extract data
        lines = open("error_estimates.txt").read().split("\n")[:-1]
        level_lines = [l for l in lines]
        ref = [float(l.split(" ")[0]) for l in level_lines]
        E   = [float(l.split(" ")[1]) for l in level_lines]
        E_h = [float(l.split(" ")[2]) for l in level_lines]
        E_k = [float(l.split(" ")[3]) for l in level_lines]
        E_c = [float(l.split(" ")[4]) for l in level_lines]

        # Plot error estimates 
        figure(666)
        subplot(4, 1, 1); plot(ref, E, '-or');grid(True)
        title("Error estimate ",  fontsize=30)	
        legend(["E"], loc='best');
        subplot(4, 1, 2); plot(ref, E_h, 'dg-');grid(True)
        legend(["E_h"], loc='best');
        subplot(4, 1, 3); plot(ref, E_k, 'p-');grid(True)
        legend(["E_h"], loc='best');
        subplot(4, 1, 4); plot(ref, E_c,'-sk');grid(True)
        legend(["E_c"], loc='best');
        xlabel('Refinement level', fontsize=30);

    show()
plots()


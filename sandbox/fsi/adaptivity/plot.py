"This module plots data from the adaptive algorithm"

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2010-11-25

from pylab import *
from numpy import trapz
import sys 

print ""
print ""
print "================================"
print "Default: (M,E)=1, (k,tol,Mt)=0"
print "================================"
print ""
print "M  = Goal Functional ,                E = Error Estimate"
print "k  = Time Steps & Residuals,        tol = FSI Tolerance & # iterations"
print "Mt = Goal Functional vs Time"
print ""
print ""


# Define default plot settings
plot_time_step       = 0
plot_FSI_tol         = 0
plot_goal_functional = 1
plot_goal_vs_time    = 0
plot_error_estimate  = 1

# Get command-line parameters
for arg in sys.argv[1:]:
    if not "=" in arg: continue
    key, val = arg.split("=")

    if key == "k":
         plot_time_step= int(val)
    elif key == "tol":
        plot_FSI_tol = int(val)
    elif key == "M":
        plot_goal_functional = int(val)
    elif key == "Mt":
        plot_goal_vs_time = int(val)
    elif key == "E":
        plot_error_estimate = int(val)

# Define plots
def plots():
    "Plot adaptive time steps"

    # Read files and get number of levels
    lines      =  open("timesteps.txt").read().split("\n")[:-1]
    lines_tol  =  open("fsi_tolerance.txt").read().split("\n")[:-1]
    lines_iter =  open("no_iterations.txt").read().split("\n")[:-1]
    lines_goal =  open("goal_functional.txt").read().split("\n")[:-1]

    # Determine the number of refinement leves to plot
    num_levels = max(int(l.split(" ")[0]) for l in lines) + 1

    # Plot all adaptive time step sequences 
    if plot_time_step == True:

        for level in range(num_levels):
            print "Plotting time steps (k=1) for level %d" % level

            # Extract data for time steps
            level_lines = [l for l in lines if int(l.split(" ")[0]) == level]
            t   =  [float(l.split(" ")[1]) for l in level_lines]
            k   =  [float(l.split(" ")[2]) for l in level_lines]
            R   =  [float(l.split(" ")[3]) for l in level_lines]

            # Plot time step and time residual
            figure(level)
            subplot(2, 1, 1); grid(True); plot(t, k, '-g',linewidth=4)
            ylabel("$k_n(t)$", fontsize=30); title("Time steps & residual,  level %d" %level, fontsize=30)
            subplot(2, 1, 2); grid(True); plot(t, R, '-r', linewidth=4)
            ylabel('$|r_k|$', fontsize=30)
            xlabel("$k_n(t)$", fontsize=30)

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
            subplot(2, 1, 2); grid(True); plot(t_iter, iter,'-r', linewidth=4)
            ylabel("#iter.", fontsize=30)
            xlabel("$k_n(t)$", fontsize=30)

    # Plot goal functional as a function of time 
    for level in range(num_levels):
           
        # Extract data for goal functional
        level_lines_goal = [l_goal for l_goal in lines_goal if int(l_goal.split(" ")[0]) == level]
        t_goal = [float(l_goal.split(" ")[1]) for l_goal in level_lines_goal]
        M = [float(l_goal.split(" ")[2]) for l_goal in level_lines_goal]
            
        # Compute integral goal functional
        M_ave = trapz(t_goal, M)
            
        # Write to file (does not matter if the data is duplicated!!!)
        f = open("M_ave.txt", "a")
        f.write("%d %g \n" % (level, M_ave))
        f.close()

        if plot_goal_vs_time == True:
            print "Plotting goal functional vs time (Mt=1) for level %d" % level

            # Plot goal functional as a function of time
            figure((level + 200))
            plot(t_goal, M, '-m', linewidth=4); grid(True)
            title("Goal Functional vs time, level %d" %level, fontsize=30)
            ylabel('$\mathcal{M}(u^h)$',fontsize=36); 
            xlabel("$k_n(t)$", fontsize=30)

        # Extract data for integrated goal functional
        lines_M = open("M_ave.txt").read().split("\n")[:-1]
        level_lines_M = [l for l in lines_M]
        ref_level = [float(l.split(" ")[0]) for l in level_lines_M]
        M_int = [float(l.split(" ")[1]) for l in level_lines_M]
            
    # Plot integrated goal functional
    if plot_goal_functional == True:
        print "Plotting goal functional (M=1)"
        figure((level + 300))
        title("Goal Functional", fontsize=30)
        plot(ref_level, M_int, '-dk'); grid(True)
        ylabel('$\int_0^T \mathcal{M}^t dt$',fontsize=36); 
        xlabel("Refinment level", fontsize=30)

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
        legend('E');
        subplot(4, 1, 2); plot(ref, E_h, 'dg-');grid(True)
        legend('h');
        subplot(4, 1, 3); plot(ref, E_k, 'p-');grid(True)
        legend('k');
        subplot(4, 1, 4); plot(ref, E_c,'-sk');grid(True)
        legend('c');
        xlabel('Refinement level', fontsize=30);
        
    show()
plots()


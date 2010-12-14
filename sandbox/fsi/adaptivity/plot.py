"This module plots data from the adaptive algorithm"

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2010-12-14

from pylab import *
from numpy import trapz, ones, abs
import sys 

print ""
print ""
print "***************************************"
print "Default: Mt=1, (M, Md, E, k, tol, I)=0"
print "***************************************"
print ""
print "M  = Goal Functional vs refinement,   Md = Goal Functional vs #dof"
print "Mt = Goal Functional vs Time,         E = Error Estimate"
print "k  = Time Steps & Residuals,        tol = FSI Tolerance & #iterations"
print "I  = Efficiency Index,               DT = Number of dofs and time steps "
print ""
print ""

# Define default plot settings
plot_time_step        = 0
plot_FSI_tol          = 0
plot_goal_vs_level    = 0
plot_goal_vs_dofs     = 0
plot_goal_vs_time     = 1
plot_error_estimate   = 0
plot_efficiency_index = 0
plot_dofs_vs_level    = 0

# Get command-line parameters
for arg in sys.argv[1:]:
    if not "=" in arg: continue
    key, val = arg.split("=")

    if key == "E":
        plot_error_estimate = int(val)
    elif key == "k":
         plot_time_step = int(val)
    elif key == "tol":
        plot_FSI_tol = int(val)
    elif key == "I":
         plot_efficiency_index = int(val)
    elif key == "M":
        plot_goal_vs_level = int(val)
    elif key == "Mt":
        plot_goal_vs_time = int(val)
    elif key == "Md":
        plot_goal_vs_dofs = int(val)
    elif key == "DT":
        plot_dofs_vs_level = int(val)

# Define plots
def plots():

    # Read files ("on the run data")
    lines_iter =  open("no_iterations.txt").read().split("\n")[:-1]
    lines_tol  =  open("fsi_tolerance.txt").read().split("\n")[:-1]
    lines_goal =  open("goal_functional.txt").read().split("\n")[:-1]
    
    # Determine the number of refinement levels for on the run data
    num_levels = max(int(l_iter.split(" ")[0]) for l_iter in lines_iter) + 1

    # Empty old file
    f = open("M_ave.txt", "w")
    f.close()

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
            ylabel("$TOL_{fSM}$", fontsize=30)
            title("FSI tolerance & # iter., level %d" %level, fontsize=30)
            subplot(2, 1, 2); grid(False); 
            
            # FIXME: axhspan/vlines  do not work on BB
#            axhspan(0.0, max(iter), xmin=0, xmax=max(t_iter), color='w')
#            vlines(t_iter, iter, 1.0, color='k', linestyles='-',  linewidth=1.5)
            plot(t_iter, iter, '-or', linewidth=2); grid(True)
            ylabel("#iter.", fontsize=30)
            xlabel("$t$", fontsize=30)

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
        legend(["$\sum$ E "], loc='best');
        subplot(4, 1, 2); plot(ref, E_h, 'dg-');grid(True)
        legend(["E_h"], loc='best');
        subplot(4, 1, 3); plot(ref, E_k, 'p-');grid(True)
        legend(["E_h"], loc='best');
        subplot(4, 1, 4); plot(ref, E_c,'-sk');grid(True)
        legend(["E_c"], loc='best');
        xlabel('Refinement level', fontsize=30);


    # Process goal functional data 
    for level in range(num_levels):
           
        # Extract data for goal functional
        level_lines_goal = [l_goal for l_goal in lines_goal if int(l_goal.split(" ")[0]) == level]
        t_goal = [float(l_goal.split(" ")[1]) for l_goal in level_lines_goal]
        M = [float(l_goal.split(" ")[2]) for l_goal in level_lines_goal]

        # Compute integral goal functional
        M_ave = trapz(t_goal, M)

        # Write M_ave to file 
        f = open("M_ave.txt", "a")
        f.write("%d %g \n" % (level, M_ave))
        f.close()

        # Plot goal functional vs time 
        if plot_goal_vs_time == True:
            print "Plotting goal functional vs time (Mt=1) for level %d" % level

            # Plot goal functional as a function of time
            figure((level + 200))
            plot(t_goal, M, '-m', linewidth=4); grid(True)
            title("Goal Functional vs time, level %d" %level, fontsize=30)
            ylabel('$\mathcal{M}(u^h)$', fontsize=36); 
            xlabel("$t$", fontsize=30)
            
    # Process and plot goal functional as a function of end time T

    # Extract data for dofs
    lines_dofs =  open("num_dofs.txt").read().split("\n")[:-1]
    level_lines_dofs = [l_dofs for l_dofs in lines_dofs]
    dofs  = [float(l_dofs.split(" ")[1]) for l_dofs in level_lines_dofs]
    space_dofs = [float(l_dofs.split(" ")[2]) for l_dofs in level_lines_dofs]
    time_dofs = [float(l_dofs.split(" ")[3]) for l_dofs in level_lines_dofs]

    # Extract data for goal functional 
    lines_MT   =  open("M_ave.txt").read().split("\n")[:-1]
    level_lines_MT = [l_MT for l_MT in lines_MT]
    ref_level_temp = [float(l_MT.split(" ")[0]) for l_MT in level_lines_MT]
    MT_temp = [float(l_MT.split(" ")[1]) for l_MT in level_lines_MT]
    
    # Create empty sets 
    MT = []
    ref_level = []

    # Determine the number of complete computed cycles 
    cycles = max(int(l.split(" ")[0]) for l in lines_dofs) + 1

    # Extract goal functionals at end time T
    for j in range(cycles):
        MT.append(MT_temp[j])
        ref_level.append(ref_level_temp[j])
           
    # Plot integrated goal functional vs refinement level
    if plot_goal_vs_level == True:
        print "Plotting integrated goal functional vs refinenment level (M=1)"
          
        # FIXME: Add reference values
        figure((level + 300))
        title("Goal Functional", fontsize=30)
        plot(ref_level, MT, '-dk'); grid(True)
        ylabel('$\mathcal{M}^T$', fontsize=36); 
        xlabel("Refinment level", fontsize=30)

    # Plot integrated goal functional vs number of dofs
    if plot_goal_vs_dofs == True:
        print "Plotting goal functional vs #dofs (Md=1)"

        # FIXME: Add reference values
        figure((level + 900))
        title("Convergence of Goal Functional", fontsize=30)
        semilogx(dofs, abs(MT), '-dk'); grid(True)
        ylabel('$\mathcal{M}^T$', fontsize=36); 
        xlabel("#dofs", fontsize=30)
        legend(["Adaptive"], loc='best')

    # Plot number of (space) dofs and number of time steps
    if plot_dofs_vs_level == True:
        print "Plotting  #dofs and #time steps (DT=1)"

        figure(level + 1200)        
        title("#dofs vs refinement level", fontsize=30)
        semilogy(ref_level, space_dofs, '--dg', linewidth=3); grid(True)
        semilogy(ref_level, time_dofs, '--or', linewidth=3); grid(True)   
        legend(["dofs", "Time steps"], loc='best');
        xlabel('Refinement level', fontsize=30);
 

    # Plot efficiency index
    if plot_efficiency_index == True:
        print " Implement!!!  "


    show()
plots()





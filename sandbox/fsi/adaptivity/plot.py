"This module plots data from the adaptive algorithm"

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2010-12-14

from pylab import *
from numpy import trapz, ones, abs, array
import sys 

print ""
print ""
print "*********************************************"
print "Default: Mt=1, (M, Md, E, k, tol, I, UA)=0"
print "*********************************************"
print ""
print "M  = Goal Functional vs refinement,   Md = Goal Functional vs #dof"
print "Mt = Goal Functional vs Time,         E = Error Estimate"
print "k  = Time Steps & Residuals,        tol = FSI Tolerance & #iterations"
print "I  = Efficiency Index,               DT = Number of dofs and time steps "
print "UA = Uniform/Adaptive error vs #dofs"
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
plot_error_adaptive_vs_uniform = 0

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
    elif key == "UA":
        plot_error_adaptive_vs_uniform = int(val)

# Define plots
def plots():

    # Read files ("on the run data")
    lines_iter =  open("no_iterations.txt").read().split("\n")[:-1]
    lines_tol  =  open("fsi_tolerance.txt").read().split("\n")[:-1]
    lines_goal =  open("goal_functional.txt").read().split("\n")[:-1]
  
    # FIXME: Add reference values here!!!

    # Determine the number of refinement levels for on the run data
    num_levels = max(int(l.split(" ")[0]) for l in lines_iter) + 1

    # Empty old file
    f = open("M_ave.txt", "w")
    f.close()
  
    # Plot time step sequences for each adaptive loop
    if plot_time_step == True:
        
        # Read file (only created when an adaptive time step is used)
        lines_time = open("timesteps.txt").read().split("\n")[:-1]

        for level in range(num_levels):
            print "Plotting time steps (k=1) for level %d" % level

            # Extract data for time steps
            level_lines_time = [l for l in lines_time if int(l.split(" ")[0]) == level]
            t   =  [float(l.split(" ")[1]) for l in level_lines_time]
            k   =  [float(l.split(" ")[2]) for l in level_lines_time]
            R   =  [float(l.split(" ")[3]) for l in level_lines_time]

            # Plot time step and time residual
            figure(level)
            subplot(2, 1, 1); grid(True); plot(t, k, '-g',linewidth=4)
            ylabel("$k_n(t)$", fontsize=30); title("Time steps & residual,  level %d" %level, fontsize=30)
            subplot(2, 1, 2); grid(True); plot(t, R, '-r', linewidth=4)
            ylabel('$|r_k|$', fontsize=30)
            xlabel("$t$", fontsize=30)

    # Plot FSI tolerance and number of FSI iterations for each adaptive loop
    if plot_FSI_tol == True:

        for level in range(num_levels):
            print "Plotting FSI tolerance and no. of itrations (tol=1) for level %d" % level

            # Extract data for FSI tolerance 
            level_lines_tol = [l for l in lines_tol if int(l.split(" ")[0]) == level]
            t_tol = [float(l.split(" ")[1]) for l in level_lines_tol]
            tol   = [float(l.split(" ")[2]) for l in level_lines_tol]

            # Extract data for no. of iterations
            level_lines_iter = [l for l in lines_iter if int(l.split(" ")[0]) == level]
            t_iter = [float(l.split(" ")[1]) for l in level_lines_iter]
            iter   = [float(l.split(" ")[2]) for l in level_lines_iter]
            
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

    # --Process goal functional---------------------------------------------- 
    for level in range(num_levels):
           
        # Extract data for goal functional
        level_lines_goal = [l for l in lines_goal if int(l.split(" ")[0]) == level]
        t_goal = [float(l.split(" ")[1]) for l in level_lines_goal]
        M      = [float(l.split(" ")[2]) for l in level_lines_goal]

        # Compute time integrated goal functional
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
   
    # Extract data for dofs
    lines_dofs =  open("num_dofs.txt").read().split("\n")[:-1]
    level_lines_dofs = [l for l in lines_dofs]
    dofs       = [float(l.split(" ")[1]) for l in level_lines_dofs]
    space_dofs = [float(l.split(" ")[2]) for l in level_lines_dofs]
    time_dofs  = [float(l.split(" ")[3]) for l in level_lines_dofs]

    # Extract data for goal functional 
    lines_MT   =  open("M_ave.txt").read().split("\n")[:-1]
    level_lines_MT = [l for l in lines_MT]
    ref_level_temp = [float(l.split(" ")[0]) for l in level_lines_MT]
    MT_temp        = [float(l.split(" ")[1]) for l in level_lines_MT]
    
    # Create empty sets 
    MT = []
    ref_level_T = []

    # Determine the number of complete computed cycles 
    cycles = max(int(l.split(" ")[0]) for l in lines_dofs) + 1

    # Extract goal functionals at end time T
    for j in range(cycles):
        MT.append(MT_temp[j])
        ref_level_T.append(ref_level_temp[j])
         
    # Create an array for MT (used in efficiency index)
    MT_array = array(MT)


    # --Process reference value--------------------------------------- 

    # Extract data
    lines_reference = open("reference_paper1.txt").read().split("\n")[:-1]
    level_lines_reference  = [l for l in lines_reference]
    ny_reference      = [float(l.split(" ")[0]) for l in level_lines_reference]
    MT_reference      = [float(l.split(" ")[1]) for l in level_lines_reference]
    dofs_reference    = [float(l.split(" ")[2]) for l in level_lines_reference]
    h_dofs_reference  = [float(l.split(" ")[3]) for l in level_lines_reference]
    dt_dofs_reference = [float(l.split(" ")[4]) for l in level_lines_reference]
    
    # Extract reference value
    last = len(MT_reference) - 1
    goal_reference = MT_reference[last]

    # Compute uniform error 
    MT_reference_array = array(MT_reference)
    E_uniform = abs(MT_reference_array - goal_reference)

    # Compute error in goal functional (|M(u^h) - M(u)|)
    Me = abs(MT_array - goal_reference)

    # --Process error estimate--------------------------------------- 

    # Extract data
    lines_error = open("error_estimates.txt").read().split("\n")[:-1]
    level_lines_error = [l for l in lines_error]
    refinment_level = [float(l.split(" ")[0]) for l in level_lines_error]
    E   = [float(l.split(" ")[1]) for l in level_lines_error]
    E_h = [float(l.split(" ")[2]) for l in level_lines_error]
    E_k = [float(l.split(" ")[3]) for l in level_lines_error]
    E_c = [float(l.split(" ")[4]) for l in level_lines_error]

    # Create an array for E (used in efficiency index)
    E_array = array(E)

    # Compute efficiency index
    E_index = E_array / Me

    # --Plot error estimates and goal functional----------------------

    # Plot error estimate
    if plot_error_estimate == True:
        print "Plotting error estimates (E=1)"

        # Plot error estimates 
        figure(88888)
        subplot(4, 1, 1); plot(refinment_level, E, '-or');grid(True)
        title("Error estimate ",  fontsize=30)	
        legend(["$\sum$ E "], loc='best');
        subplot(4, 1, 2); plot(refinment_level, E_h, 'dg-');grid(True)
        legend(["E_h"], loc='best');
        subplot(4, 1, 3); plot(refinment_level, E_k, 'p-');grid(True)
        legend(["E_h"], loc='best');
        subplot(4, 1, 4); plot(refinment_level, E_c,'-sk');grid(True)
        legend(["E_c"], loc='best');
        xlabel('Refinement level', fontsize=30);

    # Plot uniform error vs adaptive error 
    if plot_error_adaptive_vs_uniform == True:
        figure(600)        
        title("Error ", fontsize=30)
        semilogy(dofs_reference, E_uniform, '-dg', linewidth=3); grid(True)
        semilogy(dofs, Me, '-or', linewidth=3); grid(True)   
        legend(["Uniform", "Adaptive"], loc='best');
        xlabel('#dofs ', fontsize=30);

    # Plot time integrated goal functional vs refinement level
    if plot_goal_vs_level == True:
        print "Plotting integrated goal functional vs refinenment level (M=1)"
          
        # FIXME: Add reference values
        figure(300)
        title("Goal Functional", fontsize=30)
        plot(ref_level_T, MT, '-dk'); grid(True)
        ylabel('$\mathcal{M}^T$', fontsize=36); 
        xlabel("Refinment level", fontsize=30)

    # Plot integrated goal functional vs number of dofs
    if plot_goal_vs_dofs == True:
        print "Plotting goal functional vs #dofs (Md=1)"

        figure(400)
        title("Convergence of Goal Functional", fontsize=30)
        semilogx(dofs, MT, '-dr'); grid(True)
        semilogx(dofs_reference, MT_reference, '-ok'); grid(True)
        ylabel('$\mathcal{M}^T$', fontsize=36); 
        xlabel("#dofs", fontsize=30)
        legend(["Adaptive", "Uniform"], loc='best')

    # Plot efficiency index
    if plot_efficiency_index == True:
        print "Plotting efficiency index (I=1)"

        figure(500)        
        title("Efficieny Index", fontsize=30)
        semilogx(dofs, E_index, '-dg', linewidth=3); grid(True)
        legend(["E / |M(e)|"], loc='best');
        ettor = ones(len(dofs))
        semilogx(dofs, ettor, 'b', linewidth=8); grid(True)   
        xlabel('#dofs', fontsize=30);


    # Plot number of (space) dofs and number of time steps
    if plot_dofs_vs_level == True:
        print "Plotting #dofs and #time steps (DT=1)"
        
        figure(700)        
        title("#dofs vs refinement level", fontsize=30)
        semilogy(ref_level_T, space_dofs, '--dg', linewidth=3); grid(True)
        semilogy(ref_level_T, time_dofs, '--or', linewidth=3); grid(True)   
        legend(["dofs", "Time steps"], loc='best');
        xlabel('Refinement level', fontsize=30);

    show()
plots()




#     figure()
#     plot(dofs_ref, MT_ref, '-ok', linewidth=2); grid(True)
#     title("Goal Functional; uniform ref.",fontsize=30)
#     ylabel("M(u^h)", fontsize=30)
#     xlabel("#dofs (h+k)", fontsize=30)


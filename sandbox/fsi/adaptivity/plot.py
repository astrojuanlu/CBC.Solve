"This module plots data from the adaptive algorithm"

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2011-01-27

from pylab import *
from numpy import trapz, ones, abs, array
import sys 
from dolfin import info_red, info_blue

print ""
info_red("A full adaptive loop has to be completed in order to plot") 
info_blue("A key has to be assigned a value, for example M=1") 

print ""
print ""
info_blue("M    =  Goal Functional vs refinement")
info_blue("Md   =  Goal Functional vs #dofs")
info_blue("Mdh  =  Goal Functional vs #space dofs")
info_blue("Mt   =  Goal Functional vs Time")
info_blue("E    =  Error Estimate")
info_blue("UA   =  Uniform/Adaptive error vs #dofs")
info_blue("UAh  =  Uniform/Adaptive error vs #space dofs")
info_blue("I    =  Efficiency Index")
info_blue("DT   =  #dofs and time steps vs refinemnet level ")
info_blue("k    =  Time Steps & Residuals")
info_blue("tol, at_level = FSI Tolerance & #iterations ")
print ""
info_blue("BIG_KAHUNA =  plot all!")
print ""
print ""

# Define default plot settings
plot_time_step        = 0
plot_FSI_tol          = 0
at_level              = 0
plot_goal_vs_level    = 0
plot_goal_vs_dofs     = 0
plot_goal_vs_time     = 0
plot_error_estimate   = 0
plot_efficiency_index = 0
plot_dofs_vs_level    = 0
plot_goal_vs_space_dofs  = 0
plot_error_adaptive_vs_uniform = 0
plot_error_adaptive_vs_uniform_space_dofs = 0


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
    elif key == "Mdh":
        plot_goal_vs_space_dofs = int(val)
    elif key == "DT":
        plot_dofs_vs_level = int(val)
    elif key == "UA":
        plot_error_adaptive_vs_uniform = int(val)
    elif key == "UAh":
        plot_error_adaptive_vs_uniform_space_dofs= int(val)
    elif key == "at_level":
        at_level = int(val)
    elif key == "BIG_KAHUNA":
        plot_time_step        = 1
        plot_FSI_tol          = 1
        at_level              = 0
        plot_goal_vs_level    = 1
        plot_goal_vs_dofs     = 1
        plot_goal_vs_time     = 1
        plot_error_estimate   = 1
        plot_efficiency_index = 1
        plot_dofs_vs_level    = 1
        plot_goal_vs_space_dofs  = 1
        plot_error_adaptive_vs_uniform = 1
        plot_error_adaptive_vs_uniform_space_dofs = 1
        

# Define plots
def plots():

    # Read files ("on the run data")
    lines_iter =  open("no_iterations.txt").read().split("\n")[:-1]
    lines_tol  =  open("fsi_tolerance.txt").read().split("\n")[:-1]
    lines_goal =  open("goal_functional.txt").read().split("\n")[:-1]
  
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
            figure(1)
            subplot(2, 1, 1); grid(True); plot(t, k, '-k',linewidth=0.75)
            ylabel("$k_n(t)$", fontsize=30); title("Time steps & residual, up to level %d" %level, fontsize=25)
            subplot(2, 1, 2); grid(True); plot(t, R, 'r', linewidth=0.75)
            ylabel('$|r_k|$', fontsize=30)
            xlabel("$t$", fontsize=30)


    # Plot FSI tolerance and number of FSI iterations for each adaptive loop
    if plot_FSI_tol == True:

            # Get the level
            level = at_level

            if at_level > num_levels - 1:
                info_red("Level do not exist! Highest available level: %d", num_levels - 1) 
                exit(True)
                
            print "Plotting FSI tolerance and no. of itrations (tol=1) at level %d" % level

            # Extract data for FSI tolerance 
            level_lines_tol = [l for l in lines_tol if int(l.split(" ")[0]) == level]
            t_tol = [float(l.split(" ")[1]) for l in level_lines_tol]
            tol   = [float(l.split(" ")[2]) for l in level_lines_tol]

            # Extract data for no. of iterations
            level_lines_iter = [l for l in lines_iter if int(l.split(" ")[0]) == level]
            t_iter = [float(l.split(" ")[1]) for l in level_lines_iter]
            iter   = [float(l.split(" ")[2]) for l in level_lines_iter]
            
            # Plot FSI tolerance and no. of FSI iterations
            figure((level + 50)) 
            subplot(2, 1, 1); grid(True); plot(t_tol, tol, '-k', linewidth=4.0)
            ylabel("$TOL_{fSM}$", fontsize=30)
            title("FSI tolerance & # iter., level %d" %level, fontsize=30)
            subplot(2, 1, 2); grid(False); 
            
            # FIXME: axhspan/vlines do not work on BB
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
            figure(200)
            plot(t_goal, M, '-k', linewidth=1); grid(True)
            title("Goal Functional vs time, up to level %d" %level, fontsize=25)
            ylabel('$\mathcal{M}(u^h)$', fontsize=36); 
            xlabel("$t$", fontsize=30)
   
    # Extract data for dofs
    lines_dofs =  open("num_dofs.txt").read().split("\n")[:-1]
    level_lines_dofs = [l for l in lines_dofs]
    dofs    = [float(l.split(" ")[1]) for l in level_lines_dofs]
    h_dofs  = [float(l.split(" ")[2]) for l in level_lines_dofs]
    dt_dofs = [float(l.split(" ")[3]) for l in level_lines_dofs]

    # Create extra set of dofs
    # In some cases this is needed for efficiency index (I)
    # and uniform/adaptive error (UA)
    dofs_index    = [float(l.split(" ")[1]) for l in level_lines_dofs]
    h_dofs_index  = [float(l.split(" ")[2]) for l in level_lines_dofs]

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

    # Extract uniform refinement data (reference data)
    lines_reference = open("pressure_driven_cavity_ref.txt").read().split("\n")[:-1]
    level_lines_reference  = [l for l in lines_reference]
    ny_reference      = [float(l.split(" ")[0]) for l in level_lines_reference]
    MT_reference      = [float(l.split(" ")[1]) for l in level_lines_reference]
    dofs_reference    = [float(l.split(" ")[2]) for l in level_lines_reference]
    h_dofs_reference  = [float(l.split(" ")[3]) for l in level_lines_reference]
    dt_dofs_reference = [float(l.split(" ")[4]) for l in level_lines_reference]

#     figure(11)
#     plot(dofs_reference , MT_reference, '-ok', linewidth=2); grid(True)
#     title("Goal Functional; uniform ref.",fontsize=30)
#     ylabel("M(u^h)", fontsize=30)
#     xlabel("#dofs (h + dt)", fontsize=30)
    
    # Extract reference value (last one with largest amount of dofs)
    last = len(MT_reference) - 1
    goal_reference = MT_reference[last]

    # Compute uniform error (|M(u_uni.) - M(u_reference_value)|)
    MT_reference_array = array(MT_reference)
    E_uniform = abs(MT_reference_array - goal_reference) 
    
    # Compute error in goal functional (|M(u^h) - M(u_reference_value)|)
    Me = abs(MT_array - goal_reference)



    # --Process error estimate--------------------------------------- 

    # Extract data
    lines_error = open("error_estimates.txt").read().split("\n")[:-1]
    level_lines_error = [l for l in lines_error]
    refinement_level = [float(l.split(" ")[0]) for l in level_lines_error]
    E     = [float(l.split(" ")[1]) for l in level_lines_error]
    E_h   = [float(l.split(" ")[2]) for l in level_lines_error]
    E_k   = [float(l.split(" ")[3]) for l in level_lines_error]
    E_c   = [float(l.split(" ")[4]) for l in level_lines_error]
    E_c_F = [float(l.split(" ")[5]) for l in level_lines_error]
    E_c_S = [float(l.split(" ")[6]) for l in level_lines_error]
    E_c_M = [float(l.split(" ")[7]) for l in level_lines_error]

    # Create an array for E (used in efficiency index)
    E_array = array(E)

    # Check if the error estimate is obtained at the current level
    # Often, the dual/residual is still computing at the 
    # current level while the goal functional is already obtained
    if len(E_array) < len(Me):
        print " "
        print "Waiting for error estimate on level %d" %(len(Me) - 1)
        print " => can only plot efficiency index (I) up to level %d" %(len(Me) - 2)
        print " => can only plot uniform/adaptive error (UA) up to level %d" %(len(Me) - 2)
        print " "
     
        # Create empty sets
        Me_temp =  []
        dofs_index_temp = []
        h_dofs_index_temp = []
        
        # Remove the last value
        for j in range(len(Me) - 1):
            Me_temp.append(Me[j])
            dofs_index_temp.append(dofs_index[j])
            h_dofs_index_temp.append(h_dofs_index[j])

        # Update Me and dofs_index
        Me = array(Me_temp)
        dofs_index = array(dofs_index_temp)
        h_dofs_index = array(h_dofs_index_temp)


    # Compute efficiency index
    E_index = E_array / Me


    # --Plot error estimates and goal functional----------------------

    # Plot error estimate
    if plot_error_estimate == True:
        print "Plotting error estimates (E=1)"

        figure(1) 
        subplot(4, 1, 1); plot(refinement_level, E, '-ok');grid(True)
        title("Error estimate ",  fontsize=30)	
        legend(["E"], loc='best')
        subplot(4, 1, 2); plot(refinement_level, E_h, 'db-');grid(True)
        legend(["E_h"], loc='best')
        subplot(4, 1, 3); plot(refinement_level, E_k, '-pr');grid(True)
        legend(["E_k"], loc='best')
        subplot(4, 1, 4); plot(refinement_level, E_c,'-sm');grid(True)
        legend(["E_c"], loc='best')
        xlabel('Refinement level', fontsize=30)
        
        figure(2) 
        subplot(3, 1, 1); plot(refinement_level, E_c_F, 'db-');grid(True)
        title("Computational errors ",  fontsize=30)	
        legend(["Ec_F"], loc='best')
        subplot(3, 1, 2); plot(refinement_level, E_c_S, '-pr');grid(True)
        legend(["Ec_S"], loc='best')
        subplot(3, 1, 3); plot(refinement_level, E_c_M,'-sm');grid(True)
        legend(["Ec_M"], loc='best')
        xlabel('Refinement level', fontsize=30)

    # Plot uniform error vs adaptive error (dofs) 
    if plot_error_adaptive_vs_uniform == True:
        print "Plotting uniform errors and adaptive error vs #dofs (UA=1)"
  
        figure(3)        
        title("Error ", fontsize=30)
        semilogy(dofs_reference, E_uniform, '-dg', linewidth=3); grid(True)
        semilogy(dofs_index, Me, '-ok', linewidth=3); grid(True)   
        legend(["Uniform", "Adaptive"], loc='best')
        ylabel("log(E)", fontsize=25)
        xlabel('#dofs ', fontsize=30)

    # Plot uniform error vs adaptive error (dofs) 
    if plot_error_adaptive_vs_uniform_space_dofs == True:
        print "Plotting uniform errors and adaptive error vs #space dofs (UAh=1)"
  
        figure(4)        
        title("Error ", fontsize=30)
        semilogy(h_dofs_reference, E_uniform, '-dg', linewidth=3); grid(True)
        semilogy(h_dofs_index, Me, '-ok', linewidth=3); grid(True)   
        legend(["Uniform", "Adaptive"], loc='best')
        ylabel("log(E)", fontsize=25)
        xlabel('#space dofs ', fontsize=30)

    # Plot time integrated goal functional vs refinement level
    if plot_goal_vs_level == True:
        print "Plotting integrated goal functional vs refinenment level (M=1)"

        # Create list for reference
        goal_reference_list = []
        for j in range(len(ref_level_T)):
            goal_reference_list.append(goal_reference)


        figure(5)
        title("Goal Functional", fontsize=30)
        plot(ref_level_T, MT, '-dk'); grid(True)
        plot(ref_level_T, goal_reference_list, '-g', linewidth=6)
        legend(['Adaptive', 'Reference Value'])
        ylabel('$\int_0^T \mathcal{M}^t dt$', fontsize=36)
        xlabel("Refinment level", fontsize=30)

    # Plot integrated goal functional vs number of dofs
    if plot_goal_vs_dofs == True:
        print "Plotting goal functional vs #dofs (Md=1)"

        figure(6)
        title("Convergence of Goal Functional", fontsize=30)
        semilogx(dofs, MT, '-dk'); grid(True)
        semilogx(dofs_reference, MT_reference, '-og'); grid(True)
        ylabel('$\int_0^T \mathcal{M}^t dt$', fontsize=36)
        xlabel("#dofs", fontsize=30)
        legend(["Adaptive", "Uniform"], loc='best')

    # Plot integrated goal functional vs number of space dofs
    if plot_goal_vs_space_dofs == True:
        print "Plotting goal functional vs #space dofs (Mdh=1)"

        figure(7)
        title("Convergence of Goal Functional", fontsize=30)
        semilogx(h_dofs, MT, '-dk'); grid(True)
        semilogx(h_dofs_reference, MT_reference, '-og'); grid(True)
        ylabel('$\int_0^T \mathcal{M}^t dt$', fontsize=36)
        xlabel("#space dofs", fontsize=30)
        legend(["Adaptive", "Uniform"], loc='best')

    # Plot efficiency index
    if plot_efficiency_index == True:
        print "Plotting efficiency index (I=1)"

        figure(8)        
        title("Efficiency Index", fontsize=30)
        semilogx(dofs_index, E_index, '-dk', linewidth=3); grid(True)
        legend(["E / |M(e)|"], loc='best')
        ettor = ones(len(dofs_index))
        semilogx(dofs_index, ettor, 'g', linewidth=8); grid(True)   
        xlabel('#dofs', fontsize=30)

    # Plot number of (space) dofs and number of time steps
    if plot_dofs_vs_level == True:
        print "Plotting # (space) dofs and #time steps (DT=1)"
        
        figure(9)        
        title("#dofs vs refinement level", fontsize=30)
        semilogy(ref_level_T, h_dofs, '-db', linewidth=3); grid(True)
        semilogy(ref_level_T, dt_dofs, '-or', linewidth=3); grid(True)   
        legend(["Space dofs", "Time steps"], loc='best')
        xlabel('Refinement level', fontsize=30)

    show()
plots()





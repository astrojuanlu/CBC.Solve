"This module plots data from the adaptive algorithm"

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2010-10-05

from pylab import *



def plot_time_steps():
    "Plot adaptive time steps"

    # Read file and get number of levels
    lines = open("timesteps.txt").read().split("\n")[:-1]
    num_levels = max(int(l.split(" ")[0]) for l in lines) + 1

    # Plot all adaptive time step sequences
    for level in range(num_levels):
        print "Plotting time steps for level %d" % level

        # Extract data
        level_lines = [l for l in lines if int(l.split(" ")[0]) == level]
        t = [float(l.split(" ")[1]) for l in level_lines]
        k = [float(l.split(" ")[2]) for l in level_lines]
        R = [float(l.split(" ")[3]) for l in level_lines]

        # Plot
        figure(level)
        subplot(2, 1, 1); grid(True); plot(t, k, '-o')
        ylabel("$k_n(t)$", fontsize=30); title("Refinement level, %d" %level, fontsize=30)
        subplot(2, 1, 2); grid(True); plot(t, R, '-o')
        ylabel('$R_k$', fontsize=30); xlabel("$t$", fontsize=30)
   
    def plot_error_estimates():
        "Plot error estimates"
        
        # Extract data
        lines = open("error_estimates.txt").read().split("\n")[:-1]
        level_lines = [l for l in lines]
        ref = [float(l.split(" ")[0]) for l in level_lines]
        E   = [float(l.split(" ")[1]) for l in level_lines]
        E_h = [float(l.split(" ")[2]) for l in level_lines]
        E_k = [float(l.split(" ")[3]) for l in level_lines]
        E_c = [float(l.split(" ")[4]) for l in level_lines]

        # Plot
        figure()
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

        def plot_goal_functional():
            "Plot goal functional"
            
            # Extract data
            lines = open("goal_functional.txt").read().split("\n")[:-1]
            level_lines = [l for l in lines]
            M = [float(l.split(" ")[0]) for l in level_lines]

            # Plot
            figure()
            grid(True)
            plot(M, 'og-')
            title('Goal functional', fontsize=30)
            xlabel('Refinement level', fontsize=30);
            ylabel('$\mathcal{M}(u^h)$',fontsize=36); 
               
        plot_goal_functional()
    plot_error_estimates()
plot_time_steps()

show()


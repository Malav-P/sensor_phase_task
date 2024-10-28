import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.animation as animation
import numpy as np
from copy import copy
from typing import Optional
import warnings

from SensorTasking.ssa_problem import SSA_Problem
from SensorTasking.compute_coefficients import compute_coefficients

def render(p: SSA_Problem,
           x: np.ndarray[float],
           fig: int,
           control: Optional[np.ndarray[int]]= None,
           save: Optional[bool] = False):
    """
    Renders the simulation animation for the given SSA problem and agent phases.

    Parameters:
        p (SSA_Problem): The SSA problem instance.
        x (np.ndarray[float]): The phases of each observer.
        fig (int): The figure number to use for the animation.
        control (Optional[np.ndarray[int]], optional): The control sequence. Defaults to None.
        save : Saves gif to file if True
    Returns:
        animation.FuncAnimation: The animation object.

    Notes:
        - If control is None, the optimal control is used.
        - Otherwise, it uses the given control.
        - The animation showcases the orbits of observers and targets and the connections between them.

    """

    plt.close(fig)
    fig1 = plt.figure(fig)
    ax = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
    # ax.set_title(f"Opt: {p.opt}")
    # ax.set_xlabel("x (DU)")
    # ax.set_ylabel("y (DU)")

    if control is None:
        p._gen_env(x)
        information = compute_coefficients(p.env)
        control, _ = p.solve_func(information)
    else:
        p._gen_env(x)

    if control.dtype != int:
        warnings.warn("`control` does not contain elements of type `int`, program may raise Exceptions")


    p.env.reset()

    truth_posns = np.zeros(shape=(p.env.truths.size, 2)) # x, y coordinates only, z is ignored
    obs_posns = np.zeros(shape=(p.env.observers.size, 2))
    
    # plot observer orbits
    for i, observer in enumerate(p.env.observers):
        t = np.arange(0, observer.period, observer.tstep)
        state_hist = observer.spl(t)
        ax.plot(state_hist[:,0], state_hist[:, 1], linewidth='0.5', color = 'black')
        obs_posns[i] = state_hist[0, :2]

    
    # plot target orbits 
    for i, truth in enumerate(p.env.truths):
        t = np.arange(0, truth.period, truth.tstep)
        state_hist = truth.spl(t)
        ax.plot(state_hist[:,0], state_hist[:, 1], linewidth='0.5', color = 'black')

        truth_posns[i] = state_hist[0, :2]


    # place dots at initial target positions
    truth_scat = ax.scatter(truth_posns[:,0], truth_posns[:,1], c="red")
    # place dots at initial observer positions
    obs_scat = ax.scatter(obs_posns[:,0], obs_posns[:,1], c="blue")

    # connect the observer and target with a line
    ctrls = np.where(control[0] == 1)[1]
    lines = [None]*p.env.observers.size

    
    for i, ctrl in enumerate(ctrls):
        lines[i] = ax.plot([obs_posns[i,0], truth_posns[ctrl,0]], [obs_posns[i,1], truth_posns[ctrl,1]], linewidth = 0.2, linestyle = "--")

    def update(frame):
        # for each frame, update the data stored on each artist.
        for i, truth in enumerate(p.env.truths):
            state_ = truth.spl(frame*truth.tstep)
            truth_posns[i] = state_[:2]

        # connect the observer and target with a line
        ctrls = np.where(control[frame] == 1.)[1]

        for i, observer in enumerate(p.env.observers):
            state = observer.spl(frame*observer.tstep)
            obs_posns[i] = state[:2]

            lines[i][0].set_xdata( [obs_posns[i,0], truth_posns[ctrls[i],0]] )
            lines[i][0].set_ydata( [obs_posns[i,1], truth_posns[ctrls[i],1]] )


        # update the scatter plots
        truth_scat.set_offsets(truth_posns)
        obs_scat.set_offsets(obs_posns)

        return (truth_scat, obs_scat, *lines)
    
    
    ani = animation.FuncAnimation(fig=fig1, func=update, frames=p.env.maxsteps, interval=30)

    if save:

        writer = animation.PillowWriter(fps=15,
                                        metadata=dict(artist='Me'),
                                        bitrate=1800)
        ani.save('animation.gif', writer=writer)

    return ani

def visualize_info_vs_phase(p: SSA_Problem,
                            phases: np.ndarray[float],
                            observer:int,
                            fig: int,
                            fixed_phases: Optional[list] = None,
                            show_myopic: Optional[bool] = False,
                            show_ROG: Optional[bool] = True,
                            marker: Optional[str] = "."):
    """
    Visualizes the information gain versus phase for a given observer. Keeps phasing for all other observers fixed at 0.

    Parameters:
        p (SSA_Problem): The SSA problem instance.
        phases (np.ndarray[float]): An array of phase values.
        observer (int): The index of the observer.
        fig (int): The figure number for plotting.
        fixed_phases (list): A list of len num_observers - 1 containing the phases to fix other observers at
        show_myopic (bool): Whether to show the myopic policy as well.
        show_ROG (bool): Whether to show the ROG. Defaults to True. Is ignored if show_myopic is False

    Returns:
        None

    Notes:
        - Computes fitness values for the given phases and observer.
        - Plots the phase against the fitness values on a log scale
        - The plot represents the information gain for the observer.
    """
    sols = np.zeros_like(phases)
    opt_controls = []
    if fixed_phases is None:
        x = np.zeros(p.num_agents)
    else:
        x = copy(fixed_phases)
        x.insert(observer-1, 0.0)

    
    for i, phase in enumerate(phases):
        x[observer-1] = phase
        opt_control, sols[i] = p.get_control_obj(x)
        opt_controls.append(copy(opt_control))

    plt.close(fig)
    plt.figure(fig)
    ax = plt.gca()

    plt.rcParams.update({'font.size': 16})

    plt.xlabel("Phase")
    plt.ylabel("f")
    plt.yscale("log")
    plt.title(f"Objective vs. Observer {observer} Phase", wrap=True)
    plt.scatter(phases, sols, color = "black", marker=marker, label=f"optimal")
    

    
    if show_myopic:
        myopic_sols = np.zeros_like(sols)
        for i, phase in enumerate(phases):
            x[observer-1] = phase
            obj_in_list, myop_control = p.myopic_fitness(x)
            myopic_sols[i] = -obj_in_list[0]

            num_diff_control = np.sum(np.abs(myop_control - opt_controls[i])) / 2

        rel_obj_gap = np.mean((sols - myopic_sols)/sols)
        # abs_obj_gap = np.mean(sols - myopic_sols)
        print("Rel Obj Gap", rel_obj_gap)
        # print("Abs Obj Gap", abs_obj_gap)

        
        plt.scatter(phases, myopic_sols, color = "red", marker="x", label=f"myopic")
        if show_ROG:
            # Add a text box to show the value of rel_obj_gap
            plt.text(0.05, 0.6, f'ROG: {rel_obj_gap*100:.1f} %', transform=plt.gca().transAxes,
                fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
        plt.legend()



    # plt.scatter(phases, np.log(sols - myopic_sols), color = "black", marker='.', label="difference")
    plt.draw_if_interactive()
    
    return

def plot_orbits(p: SSA_Problem, fig: int, projection: Optional[str] = "xy"):
    """
    Visualize the orbits of agents and observers. Usually this is used to determine if orbits are intersecting.

    Parameters:
        p (SSA_Problem): The instance of the SSA problem.
        fig (int): The fig number to plot on.
        projection (str): The type of projection to plot. Can be one of "3d", "xy", "xz" or "yz"

    """

    plt.close(fig)
    plt.figure(fig)

    if projection == "3d":
        # ax = plt.gcf().add_axes((0, 0, 1, 1), projection = "3d")
        ax = plt.gcf().add_subplot(projection="3d")

        ax.set_xlabel("x (DU)")
        ax.set_ylabel("y (DU)")
        ax.set_zlabel("z (DU)")
        ax.set_title("Agent/Target Orbits")


        for catalog_ID in range(p.tg.num_options):
            state_hist, _ = p.tg.gen_state_history(catalog_ID=catalog_ID, n_points=500)
            x = state_hist[:,1]
            y = state_hist[:,2]
            z = state_hist[:,3]
            ax.plot3D(x, y, z, color="black", linewidth=0.5)

        for catalog_ID in range(p.ag.num_options):
            state_hist, _ = p.ag.gen_state_history(catalog_ID=catalog_ID, n_points=500)
            x = state_hist[:,1]
            y = state_hist[:,2]
            z = state_hist[:,3]
            ax.plot3D(x, y, z, color="red", linewidth=0.5)

        # Creating dummy plots for legend
        red_line = plt.Line2D((0,1),(0,0), color='r', linewidth=2)
        black_line = plt.Line2D((0,1),(0,0), color='k', linewidth=2)

        # Adding the legend
        ax.legend([red_line, black_line], ['Agent', 'Target'])

    else:

        match projection:
            case "xy":
                xlabel = "x"
                ylabel = "y"
                idx1 = 1
                idx2 = 2
            case "xz":
                xlabel = "x"
                ylabel = "z"
                idx1 = 1
                idx2 = 3
            case "yz":
                xlabel = "y"
                ylabel = "z"
                idx1 = 2
                idx2 = 3

        for catalog_ID in range(p.tg.num_options):
            state_hist, _ = p.tg.gen_state_history(catalog_ID=catalog_ID, n_points=500)
            x = state_hist[:,idx1]
            y = state_hist[:,idx2]
            plt.plot(x, y, color="black", linewidth=0.5)

        for catalog_ID in range(p.ag.num_options):
            state_hist, _ = p.ag.gen_state_history(catalog_ID=catalog_ID, n_points=500)
            x = state_hist[:,idx1]
            y = state_hist[:,idx2]
            plt.plot(x, y, color="red", linewidth=0.5)

        plt.rcParams.update({'font.size': 16})
        plt.xlabel(f"{xlabel} (DU)")
        plt.ylabel(f"{ylabel} (DU)")
        plt.title("Agent/Target Orbits")

        # Creating dummy plots for legend
        red_line = plt.Line2D((0,1),(0,0), color='r', linewidth=2)
        black_line = plt.Line2D((0,1),(0,0), color='k', linewidth=2)

        # Adding the legend
        plt.legend([red_line, black_line], ['Agent', 'Target'], bbox_to_anchor = [0.295, 0.655])


    plt.draw_if_interactive()

    return


def _visualize_info_other_observers_fixed(prob: SSA_Problem, observer: int, fig_num: int):
    """
    Create a grid of plots showing the objective as the phase of an observer is varied, keeping all other observers 
    at fixed phase. Each axes will have a different fixed phase.

    Args:
        prob : an instance of an SSA_Problem
        observer: the id of the observer whose phase we'd like to vary
        fig_num: the integer identifier for the figure to plot on

    """

    plt.close(fig_num)
    plt.figure(fig_num, figsize=(10,6))

    n_observers = prob.num_agents
    fig, axs = plt.subplots(2, 3, num=fig_num)

    fig.suptitle(f"Objective Vs. Observer {observer} phase")


    phases = np.linspace(0.0, 1.0, 40)

    fixed_phases = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]

    for i, ax in enumerate(axs.flatten()):
        sols = np.zeros_like(phases)

        x = [fixed_phases[i]]*n_observers
        
        for j, phase in enumerate(phases):
            x[observer-1] = phase
            sols[j] = -prob.fitness(x)[0]
        
        ax.set_xlabel("Phase")
        ax.set_ylabel("f")
        ax.set_yscale("log")
        ax.set_title(f"Fixed Obs Phase = {fixed_phases[i]}")
        ax.scatter(phases, sols, color = "black", marker='.', s=2)

    
    plt.tight_layout()
    plt.show()

    return
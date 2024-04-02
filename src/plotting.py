import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from typing import Optional

from SensorTasking.ssa_problem import SSA_Problem
from SensorTasking.compute_coefficients import compute_coefficients, solve_model

def render(p: SSA_Problem, x: np.ndarray[float], fig: int, control: Optional[np.ndarray[int]]= None):
    """
    Renders the simulation animation for the given SSA problem and agent phases.

    Parameters:
        p (SSA_Problem): The SSA problem instance.
        x (np.ndarray[float]): The phases of each observer.
        fig (int): The figure number to use for the animation.
        control (Optional[np.ndarray[int]], optional): The control sequence. Defaults to None.

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

    if control is None:
        p._gen_env(x)
        information = compute_coefficients(p.env)
        control, _ = solve_model(information)
    else:
        p._gen_env(x)

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
            state = truth.spl(frame*truth.tstep)
            truth_posns[i] = state[:2]

        # connect the observer and target with a line
        ctrls = np.where(control[frame] == 1)[1]

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

    return ani

def visualize_info_vs_phase(p: SSA_Problem, phases: np.ndarray[float], observer:int, fig: int):
    """
    Visualizes the information gain versus phase for a given observer. Keeps phasing for all other observers fixed at 0.

    Parameters:
        p (SSA_Problem): The SSA problem instance.
        phases (np.ndarray[float]): An array of phase values.
        observer (int): The index of the observer.
        fig (int): The figure number for plotting.

    Returns:
        None

    Notes:
        - Computes fitness values for the given phases and observer.
        - Plots the phase against the logarithm of the fitness values.
        - The plot represents the information gain for the observer.
    """
    sols = np.zeros_like(phases)

    for i, phase in enumerate(phases):
        x = np.zeros(p.num_agents)
        x[observer-1] = phase
        sols[i] = p.fitness(x)[0]

    plt.close(fig)
    plt.figure(fig)

    plt.xlabel("Phase")
    plt.ylabel("log(f)")
    plt.title(f"Observer {observer} Information Gain")
    plt.scatter(phases, np.log(-sols), color = "blue", marker='.', label="optimal")

    plt.draw_if_interactive()
    
    return
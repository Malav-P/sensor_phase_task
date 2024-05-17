import pygmo as pg
import numpy as np
from typing import Optional, Union
import time

from .ssa_problem import Greedy_SSA_Problem, SSA_Problem


def greedy_search(Y: np.ndarray[float],
                  Y_periods: np.ndarray[float],
                  X: np.ndarray[float],
                  X_periods: np.ndarray[float],
                  init_phase_guess: Optional[Union[float, list]] = 0.5,
                  opt: Optional[str] = "max") -> np.ndarray[float]:
    """
    Perform greedy search optimization for the phases of all observers.

    This function optimizes the phase of each observer in a greedy fashion.
    The phase of the n-th observer is optimized by placing it as the sole observer in the environment finding the optimal control
    for a chosen phase. The phase is varied and optimized with an L-BFGS algorithm.

    Parameters:
        Y (np.ndarray[float]): Initial conditions of targets. Each row is an initial condition.
        Y_periods (np.ndarray[float]): Periods of targets.
        X (np.ndarray[float]): Initial conditions of agents. Each row is an initial condition.
        X_periods (np.ndarray[float]): Periods of agents.
        init_phase_guess (Optional[float], optional): Initial phase guess. Defaults to 0.5.
        opt (str): the type of inner loop optimization to run. One of either "max" or "maxmin"

    Returns:
        np.ndarray[float]: Array containing optimized phases.
        np.ndarray[int]: Control for observers in constellation

    Notes:
        - Optimization is performed using the L-BFGS-B method.
        - The function returns an array of optimized phase for each agent.
    """
    n_agents = X_periods.size

    if isinstance(init_phase_guess, (float, int)):
        init_phase_guess = [init_phase_guess] * n_agents

    # Initialize instance of problem with first agent
    p = Greedy_SSA_Problem(target_ics=Y,
                           target_periods=Y_periods,
                           agent_ics=[X[0]],
                           agent_periods=[X_periods[0]],
                           opt=opt)
    print("Beginning Optimization...\n")
    start_time = time.time()
    
    pg_problem = pg.problem(p)
    pop = pg.population(prob=pg_problem)
    pop.push_back(np.array([init_phase_guess[0]]))
    scp = pg.algorithm(pg.scipy_optimize(method="L-BFGS-B"))

    # Optimize phase of first observer
    p.opt_phases.append(scp.evolve(pop).champion_x[0])
    control, _ = p.get_control_obj([p.opt_phases[0]])
    p.opt_controls.append(control)

    # Iteratively add agents to problem instance and optimize
    for i in range(1, n_agents):

        p.remove_agent(index=0)
        p.add_agent(X[i], X_periods[i])
        pg_problem = pg.problem(p)
        pop = pg.population(prob=pg_problem)
        pop.push_back(np.array([init_phase_guess[i]]))
        scp = pg.algorithm(pg.scipy_optimize(method="L-BFGS-B"))

        p.opt_phases.append(scp.evolve(pop).champion_x[0])
        control, _ = p.get_control_obj([p.opt_phases[i]])
        p.opt_controls.append(control)
            

    end_time = time.time()
    print(f"Finished in {end_time-start_time} sec.")

    control = np.zeros(shape=(p.env.maxsteps, n_agents, p.env.truths.size), dtype=int)

    for i, elem in enumerate(p.opt_controls):
        control[:, i, :] = np.squeeze(elem, axis=1)

    p_ = SSA_Problem(target_ics=Y,
                    target_periods=Y_periods,
                    agent_ics=X,
                    agent_periods=X_periods,
                    opt=opt)
    
    objective = p_.get_obj(x=p.opt_phases, u=control)

    return p.opt_phases, control, objective

def search(Y: np.ndarray[float],
           Y_periods: np.ndarray[float],
           X: np.ndarray[float],
           X_periods: np.ndarray[float],
           init_phase_guess: Optional[float] = 0.5,
           opt: Optional[str] = "max") -> np.ndarray[float]:
    """
    Perform search optimization for the phases of all observers.

    This function optimizes the phases of observers using L-BFGS-B. 

    Parameters:
        Y (np.ndarray[float]): Initial conditions of targets. Each row is an initial condition.
        Y_periods (np.ndarray[float]): Periods of targets.
        X (np.ndarray[float]): Initial conditions of agents. Each row is an initial condition.
        X_periods (np.ndarray[float]): Periods of agents.
        init_phase_guess (Optional[float], optional): Initial phase guess. Defaults to 0.5.
        opt (str): the type of optimization to run. One of either "max" or "maxmin"

    Returns:
        np.ndarray[float]: Array containing optimized phases for alignment.
        np.ndarray[int]: Array containing control for all observers.
        float: Objective value

    Notes:
        - Optimization is performed using the L-BFGS-B method.
        - The function returns an array of optimized phases for each agent.
    """

    n_agents = X_periods.size

    if isinstance(init_phase_guess, (float, int)):
        init_phase_guess = [init_phase_guess] * n_agents

    # Initialize instance of problem with first agent
    p = SSA_Problem(target_ics=Y,
                    target_periods=Y_periods,
                    agent_ics=X,
                    agent_periods=X_periods,
                    opt=opt)
    print("Beginning Optimization...\n")

    start_time = time.time()

    pg_problem = pg.problem(p)
    pop = pg.population(prob=pg_problem)
    pop.push_back(np.array(init_phase_guess))
    scp = pg.algorithm(pg.scipy_optimize(method="L-BFGS-B"))

    # Optimize phases
    opt_phases = scp.evolve(pop).champion_x

    end_time = time.time()
    print(f"Finished in {end_time-start_time} sec.")

    control, obj = p.get_control_obj(opt_phases)

    return opt_phases, control, obj
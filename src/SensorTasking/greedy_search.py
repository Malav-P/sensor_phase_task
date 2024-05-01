import pygmo as pg
import numpy as np
from typing import Optional
import time

from .ssa_problem import Greedy_SSA_Problem, SSA_Problem


def greedy_search(Y: np.ndarray[float], Y_periods: np.ndarray[float], X: np.ndarray[float], X_periods: np.ndarray[float], init_phase_guess: Optional[float] = 0.5):
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

    Returns:
        np.ndarray[float]: Array containing optimized phases.

    Notes:
        - Optimization is performed using the L-BFGS-B method.
        - The function returns an array of optimized phase for each agent.
    """

    # Initialize instance of problem with first agent
    print("Initializing Problem...\n")
    p = Greedy_SSA_Problem(target_ics=Y, target_periods=Y_periods, agent_ics=[X[0]], agent_periods=[X_periods[0]])
    print("Beginning Optimization...\n")
    start_time = time.time()
    
    pg_problem = pg.problem(p)
    pop = pg.population(prob=pg_problem)
    pop.push_back(np.array([init_phase_guess]))
    scp = pg.algorithm(pg.scipy_optimize(method="L-BFGS-B"))

    # Optimize phase of first observer
    p.opt_phases = np.append(p.opt_phases, scp.evolve(pop).champion_x[0])
    

    # Iteratively add agents to problem instance and optimize
    n_agents = X_periods.size
    for i in range(1, n_agents):

        p.remove_agent(index=0)
        p.add_agent(X[i], X_periods[i])
        pg_problem = pg.problem(p)
        pop = pg.population(prob=pg_problem)
        pop.push_back(np.array([init_phase_guess]))
        scp = pg.algorithm(pg.scipy_optimize(method="L-BFGS-B"))

        p.opt_phases = np.append(p.opt_phases, scp.evolve(pop).champion_x[0])

    end_time = time.time()
    print(f"Finished in {end_time-start_time} sec.")


    return p.opt_phases

def search(Y: np.ndarray[float], Y_periods: np.ndarray[float], X: np.ndarray[float], X_periods: np.ndarray[float], init_phase_guess: Optional[float] = 0.5):
    """
    Perform search optimization for the phases of all observers.

    This function optimizes the phases of observers using L-BFGS-B. 

    Parameters:
        Y (np.ndarray[float]): Initial conditions of targets. Each row is an initial condition.
        Y_periods (np.ndarray[float]): Periods of targets.
        X (np.ndarray[float]): Initial conditions of agents. Each row is an initial condition.
        X_periods (np.ndarray[float]): Periods of agents.
        init_phase_guess (Optional[float], optional): Initial phase guess. Defaults to 0.5.

    Returns:
        np.ndarray[float]: Array containing optimized phases for alignment.

    Notes:
        - Optimization is performed using the L-BFGS-B method.
        - The function returns an array of optimized phases for each agent.
    """

    n_agents = X_periods.size

    # Initialize instance of problem with first agent
    print("Initializing Problem...\n")
    p = SSA_Problem(target_ics=Y, target_periods=Y_periods, agent_ics=X, agent_periods=X_periods)
    print("Beginning Optimization...\n")

    start_time = time.time()

    pg_problem = pg.problem(p)
    pop = pg.population(prob=pg_problem)
    pop.push_back(np.array([init_phase_guess]*n_agents))
    scp = pg.algorithm(pg.scipy_optimize(method="L-BFGS-B"))

    # Optimize phases
    opt_phases = scp.evolve(pop).champion_x

    end_time = time.time()

    print(f"Finished in {end_time-start_time} sec.")
    
    return opt_phases
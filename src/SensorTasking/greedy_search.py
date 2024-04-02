import pygmo as pg
import numpy as np
from typing import Optional

from .ssa_problem import Greedy_SSA_Problem


def greedy_search(Y: np.ndarray[float], Y_periods: np.ndarray[float], X: np.ndarray[float], X_periods: np.ndarray[float], init_phase_guess: Optional[float] = 0.5):
    """
    Perform greedy search optimization for the phases of all observers.

    This function optimizes the phase of each observer in a greedy fashion.
    In general, the (n+1)-th entry of the design vector is optimized alone, holding the previous n entries 
    constant at their already optimized values.

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

    # Initialize instance of problem with first agent
    print("Initializing Problem...\n")
    p = Greedy_SSA_Problem(target_ics=Y, target_periods=Y_periods, agent_ics=[X[0]], agent_periods=[X_periods[0]])
    print("Beginning Optimization...\n")
    pg_problem = pg.problem(p)
    pop = pg.population(prob=pg_problem)
    pop.push_back(np.array([init_phase_guess]))
    scp = pg.algorithm(pg.scipy_optimize(method="L-BFGS-B"))

    # Optimize phase of first observer
    p.opt_phases = np.append(p.opt_phases, scp.evolve(pop).champion_x[0])
    

    # Iteratively add agents to problem instance and optimize
    n_agents = X_periods.size
    for i in range(1, n_agents):

        p.add_agent(X[i], X_periods[i])
        pg_problem = pg.problem(p)
        pop = pg.population(prob=pg_problem)
        pop.push_back(np.array([init_phase_guess]))
        scp = pg.algorithm(pg.scipy_optimize(method="L-BFGS-B"))

        p.opt_phases = np.append(p.opt_phases, scp.evolve(pop).champion_x[0])

    return p.opt_phases
import pygmo as pg
import numpy as np
from typing import Optional, Union, List
import time

from .ssa_problem import Greedy_SSA_Problem, SSA_Problem


def greedy_search(targets: np.ndarray[float],
                  target_periods: np.ndarray[float],
                  agents: np.ndarray[float],
                  agent_periods: np.ndarray[float],
                  init_phase_guess: Optional[np.ndarray[float]] = None,
                  opt: Optional[str] = "max") -> np.ndarray[float]:
    """
    Perform greedy search optimization for the phases of all observers.

    This function optimizes the phase of each observer in a greedy fashion.
    The phase of the n-th observer is optimized by placing it as the sole observer in the environment finding the optimal control
    for a chosen phase. The phase is varied and optimized with an L-BFGS algorithm.

    Parameters:
        targets (np.ndarray[float]): Initial conditions of targets. Each row is an initial condition.
        target_periods (np.ndarray[float]): Periods of targets.
        agents (np.ndarray[float]): Initial conditions of agents. Each row is an initial condition.
        agent_periods (np.ndarray[float]): Periods of agents.
        init_phase_guess (np.ndarray[float]) : Initial guesses as a 2D numpy array. Each column contains a set of initial conditions for an observer
        opt (str): the type of inner loop optimization to run. One of either "max" or "maxmin"

    Returns:
        np.ndarray[float]: Array containing optimized phases for alignment.
        np.ndarray[int]: Array containing control for all observers.
        float: Objective value

    Notes:
        - Optimization is performed using the L-BFGS-B method.
        - The function returns an array of optimized phase for each agent.
    """
    n_agents = agent_periods.size

    if init_phase_guess is None:
        ics = np.linspace(0., 1, 10).reshape(-1, 1)
        init_phase_guess = np.tile(ics, (1, n_agents))

    if isinstance(init_phase_guess, (float, int)):
        ics = np.array([init_phase_guess])
        init_phase_guess = np.tile(ics, (1, n_agents))

    if not (init_phase_guess.ndim == 2 and init_phase_guess.shape[1] == n_agents):
        raise ValueError("init_phase_guess must be a 2d numpy array with number of columns equal to number of agents")


    # Initialize instance of problem with first agent
    p = Greedy_SSA_Problem(targets=targets,
                           target_periods=target_periods,
                           agents=[agents[0]],
                           agent_periods=[agent_periods[0]],
                           opt=opt)
    print("Beginning Optimization...\n")
    start_time = time.time()
    
    pg_problem = pg.problem(p)
    algo = pg.algorithm(pg.scipy_optimize(method="L-BFGS-B"))

    initial_conditions = init_phase_guess[:, 0]

    champion = _run_multiple_ics(initial_conditions=initial_conditions,
                                 pg_problem=pg_problem,
                                 algo=algo)

    # Optimize phase of first observer
    p.opt_phases.append(champion[0])
    control, _ = p.get_control_obj([p.opt_phases[0]])
    p.opt_controls.append(control)

    # Iteratively add agents to problem instance and optimize
    for i in range(1, n_agents):

        p.remove_agent(index=0)
        p.add_agent(agents[i], agent_periods[i])

        pg_problem = pg.problem(p)
        algo = pg.algorithm(pg.scipy_optimize(method="L-BFGS-B"))

        initial_conditions = init_phase_guess[:, i]
        champion = _run_multiple_ics(initial_conditions=initial_conditions,
                                     pg_problem=pg_problem,
                                     algo=algo)


        p.opt_phases.append(champion[0])
        control, _ = p.get_control_obj([p.opt_phases[i]])
        p.opt_controls.append(control)
            

    end_time = time.time()
    print(f"Finished in {end_time-start_time} sec.")

    control = np.zeros(shape=(p.env.maxsteps, n_agents, p.env.truths.size), dtype=int)

    for i, elem in enumerate(p.opt_controls):
        control[:, i, :] = np.squeeze(elem, axis=1)

    p_ = SSA_Problem(targets=targets,
                    target_periods=target_periods,
                    agents=agents,
                    agent_periods=agent_periods,
                    opt=opt)
    
    objective = p_.get_obj(x=p.opt_phases, u=control)

    return p.opt_phases, control, objective

def search(targets: np.ndarray[float],
           target_periods: np.ndarray[float],
           agents: np.ndarray[float],
           agent_periods: np.ndarray[float],
           init_phase_guess: Optional[np.ndarray[float]] = None,
           opt: Optional[str] = "max") -> np.ndarray[float]:
    """
    Perform search optimization for the phases of all observers.

    This function optimizes the phases of observers using L-BFGS-B. 

    Parameters:
        targets (np.ndarray[float]): Initial conditions of targets. Each row is an initial condition.
        target_periods (np.ndarray[float]): Periods of targets.
        agents (np.ndarray[float]): Initial conditions of agents. Each row is an initial condition.
        agent_periods (np.ndarray[float]): Periods of agents.
        init_phase_guess (np.ndarray[float]): Initial phase guess as a 2d numpy array. Each row is an initial condition.
        opt (str): the type of optimization to run. One of either "max" or "maxmin"

    Returns:
        np.ndarray[float]: Array containing optimized phases for alignment.
        np.ndarray[int]: Array containing control for all observers.
        float: Objective value

    Notes:
        - Optimization is performed by running L-BFGS-B on each initial condition in parallel.
        - The function returns an array of optimized phases for each agent.
        - each list in init_phase_guess must have length equal to the number of observers
    """

    n_agents = agent_periods.size

    if init_phase_guess is None:
        ics = np.linspace(0., 1, 10).reshape(-1, 1)
        init_phase_guess = np.tile(ics, (1, n_agents))

    if isinstance(init_phase_guess, (float, int)):
        ics = np.array([init_phase_guess])
        init_phase_guess = np.tile(ics, (1, n_agents))

    if not (init_phase_guess.ndim == 2 and init_phase_guess.shape[1] == n_agents):
        raise ValueError("init_phase_guess must be a 2d numpy array with number of columns equal to number of agents")


    # Initialize instance of problem with first agent
    p = SSA_Problem(targets=targets,
                    target_periods=target_periods,
                    agents=agents,
                    agent_periods=agent_periods,
                    opt=opt)
    print("Beginning Optimization...\n")

    start_time = time.time()

    pg_problem = pg.problem(p)
    algo = pg.algorithm(pg.scipy_optimize(method="L-BFGS-B"))

    opt_phases = _run_multiple_ics(initial_conditions=init_phase_guess,
                                   pg_problem=pg_problem,
                                   algo=algo)

    end_time = time.time()
    print(f"Finished in {end_time-start_time} sec.")

    control, obj = p.get_control_obj(opt_phases)

    return opt_phases, control, obj

def sga_search(targets: np.ndarray[float],
           target_periods: np.ndarray[float],
           agents: np.ndarray[float],
           agent_periods: np.ndarray[float],
           init_phase_guess: Optional[np.ndarray[float]] = None,
           opt: Optional[str] = "max") -> np.ndarray[float]:
    """
    Perform search optimization for the phases of all observers using a simple genetic algorithm

    This function optimizes the phases of observers using a simple genetic algorithm from pygmo. 

    Parameters:
        targets (np.ndarray[float]): Initial conditions of targets. Each row is an initial condition.
        target_periods (np.ndarray[float]): Periods of targets.
        agents (np.ndarray[float]): Initial conditions of agents. Each row is an initial condition.
        agent_periods (np.ndarray[float]): Periods of agents.
        init_phase_guess (np.ndarray[float]): Initial phase guess as a list of lists. Each list is an initial condition.
        opt (str): the type of optimization to run. One of either "max" or "maxmin"

    Returns:
        np.ndarray[float]: Array containing optimized phases for alignment.
        np.ndarray[int]: Array containing control for all observers.
        float: Objective value

    """

    n_agents = agent_periods.size

    if init_phase_guess is None:
        ics = np.linspace(0., 1, 10).reshape(-1, 1)
        init_phase_guess = np.tile(ics, (1, n_agents))

    if isinstance(init_phase_guess, (float, int)):
        ics = np.array([init_phase_guess])
        init_phase_guess = np.tile(ics, (1, n_agents))

    if not (init_phase_guess.ndim == 2 and init_phase_guess.shape[1] == n_agents):
        raise ValueError("init_phase_guess must be a 2d numpy array with number of columns equal to number of agents")


    # Initialize instance of problem with first agent
    p = SSA_Problem(targets=targets,
                    target_periods=target_periods,
                    agents=agents,
                    agent_periods=agent_periods,
                    opt=opt)
    print("Beginning Optimization...\n")

    start_time = time.time()

    pg_problem = pg.problem(p)
    algo = pg.algorithm(pg.sga(gen=100))

    pop = pg.population(pg_problem)
    for ic in init_phase_guess:
        if isinstance(ic, float):
            ic = [ic]
        pop.push_back(ic)

    print("population size :", pop.get_ID().size)

    pop = algo.evolve(pop)

    opt_phases = pop.champion_x

    end_time = time.time()
    print(f"Finished in {end_time-start_time} sec.")

    control, obj = p.get_control_obj(opt_phases)

    return opt_phases, control, obj

def _run_multiple_ics(initial_conditions: np.ndarray[float],
                     pg_problem: pg.problem,
                     algo: pg.algorithm) -> List:
    """
    Run separate optimization problems in parallel for each initial conditions in a a given set. Return the best solution

    Args:
        initial_conditions (np.ndarray): a set of initial conditions to try. Can be 1d array (for greedy optimization) or 2d (for regular search)
        pg_problem (pygmo.problem): an Pygmo problem instance
        algo (pg.algorithm): a Pygmo algorithm instance

    Returns:
        champion (np.ndarray): the best candidate out of all the optimization runs
    
    """
    # Create the archipelago with n_islands islands
    archi = pg.archipelago()

    # Create and add each island with its corresponding initial condition
    for ic in initial_conditions:
        # Create a population with the current initial condition
        if isinstance(ic, float):
            ic = [ic]
        pop = pg.population(pg_problem)
        pop.push_back(ic)

        # Add the island to the archipelago
        archi.push_back(pg.island(algo=algo, pop=pop))

    archi.evolve()
    archi.wait()

    champions_x = archi.get_champions_x()
    champions_f = archi.get_champions_f()
    champions_f = [-item[0] for item in champions_f]
    champ_idx = np.argmax(champions_f)

    champion = champions_x[champ_idx]

    return champion
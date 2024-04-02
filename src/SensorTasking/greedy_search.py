import pygmo as pg
import numpy as np

from typing import Optional

from .compute_coefficients import Greedy_SSA_Problem


def greedy_search(Y, Y_periods, X, X_periods, init_phase_guess: Optional[float] = None):

    if init_phase_guess is None:
        init_phase_guess = 0.5

    p = Greedy_SSA_Problem(target_ics=Y, target_periods=Y_periods, agent_ics=[X[0]], agent_periods=[X_periods[0]])
    pg_problem = pg.problem(p)
    pop = pg.population(prob=pg_problem)
    pop.push_back(np.array([init_phase_guess]))
    scp = pg.algorithm(pg.scipy_optimize(method="L-BFGS-B"))
    p.opt_phases = np.append(p.opt_phases, scp.evolve(pop).champion_x[0])
    

    n_agents = X_periods.size
    for i in range(1, n_agents):

        p.add_agent(X[i], X_periods[i])

        pg_problem = pg.problem(p)
        pop = pg.population(prob=pg_problem)
        pop.push_back(np.array([init_phase_guess]))
        scp = pg.algorithm(pg.scipy_optimize(method="L-BFGS-B"))
        p.opt_phases = np.append(p.opt_phases, scp.evolve(pop).champion_x[0])



    return p.opt_phases







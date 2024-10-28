from .search_methods import greedy_search, search, sga_search
from numpy.typing import ArrayLike
from typing import Optional
import numpy as np

def run_experiment(obj: str,
                   method: str,
                   targets: ArrayLike,
                   target_periods: ArrayLike,
                   agents: ArrayLike,
                   agent_periods: ArrayLike,
                   init_phase_guess: Optional[np.ndarray[float]] = None):
    """
    Runs the experiment with given parameters

    Parameters:
        obj (str): The objective, either "maxmin" or "max"
        method (str) : The solver method, either "greedy", "exhaustive", or "ga"
        targets (ArrayLike) : target initial conditions
        target_periods (ArrayLike) : target periods
        agents (ArrayLike): agent initial conditions
        agent_periods (ArrayLike) : agent periods
        init_phase_guess (Optional[np.ndarray[float]]) : initial guesses for optimizer

    Returns:
        np.ndarray[float]: Array containing optimized phases for alignment.
        np.ndarray[int]: Array containing control for all observers.
        float: Objective value

    """

    match method.lower():
        case "greedy":
            search_method = greedy_search
        case "exhaustive":
            search_method = search
        case "ga":
            search_method = sga_search
        case _ :
            raise ValueError("method must be one of 'greedy', 'exhaustive', or 'ga'")

    obj = obj.lower()
    if obj not in ["maxmin", "max"]:
        raise ValueError("method must be one of 'maxmin' or 'max'")

    phases, control, objective = search_method(targets,
                                              target_periods,
                                              agents,
                                              agent_periods,
                                              init_phase_guess = init_phase_guess,
                                              opt = obj)

    print("search method: ", method)
    print(f"obj type: ", obj)
    print("log10 obj ", np.log10(objective))
    print("phases ", phases)


    return phases, control, objective
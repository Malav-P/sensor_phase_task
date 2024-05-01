import numpy as np
import gurobipy as gp
from gurobipy import GRB

from .spacenv import SpaceEnv


def compute_coefficients(env: SpaceEnv):
    """
    Computes the information coefficients for the linear program.

    Parameters:
        env (SpaceEnv): The environment containing observers and targets.

    Returns:
        np.ndarray[float]: Information coefficients for each observer and truth at each time step.

    Notes:
        - This function computes the information coefficients, which measure the information content 
          provided by observers about the true state of each target in the environment.
        - The information coefficients are computed using the Kalman filter equations.
        - The function iterates over time steps in the environment and computes the information coefficients
          for each observer-truth pair.
    """

    env.reset()

    dt = 0.1*env.tstep

    sigma =  3 * np.pi / 180
    R_inv = 1 / sigma**2 * np.block([[np.eye(3), np.zeros(shape=(3, 3))], [np.zeros(shape=(3,3)), (0.5 * dt**2)*np.eye(3)]])

    information = np.zeros(shape=(env.maxsteps, env.observers.size, env.truths.size), dtype=float)

    for k in range(env.maxsteps):

        out = env.step()
        Hs = out[-1]

        for j,truth in enumerate(env.truths):
            stm = truth.eval_stm_spl( truth.t - truth.tstep / 2)
            stm_end = truth.eval_stm_spl(truth.period)

            # Phi(t2, t1) Phi(t1, t0)  = Phi(t2, t0)      ===>       Phi(t2, t1) = Phi(t2, t0) * Phi(t1, t0)^-1    ===== > Phi(t1, t2) = Phi(t1, t0) * Phi(t2, t0)^-1
            phi_j = stm.reshape(6, 6) @ np.linalg.inv(stm_end.reshape(6, 6))

            for i in range(env.observers.size):
                information[k, i ,j] = np.trace(phi_j.T @ Hs[i, j].T @ R_inv @ Hs[i,j] @ phi_j)

    return information

def solve_model(information: np.ndarray[float]):
    """
    Solves the optimization model to assign observers to targets based on information coefficients.

    Parameters:
        information (np.ndarray[float]): Information coefficients for each observer and truth at each time step.

    Returns:
        Tuple[np.ndarray[float], float]: A tuple containing the binary assignment matrix and the objective value.

    Notes:
        - This function formulates and solves an optimization model to assign observers to targets
          based on the computed information coefficients.
        - The optimization model maximizes the total information obtained by assigning observers to targets.
        - Each observer is constrained to look at only one target at each time step.
    """
    m = gp.Model("sensortask")

    # Silence model output
    m.Params.LogToConsole = 0

    # Create variables
    u = m.addMVar(shape=information.shape, vtype=GRB.BINARY, name="u")

    # Set objective
    obj = information.reshape(-1)
    m.setObjective(obj @ u.reshape(-1), GRB.MAXIMIZE)

    # observer i can only look at one target at each timestep
    m.addConstr(u.sum(axis=2) <= 1, name="row")

    m.optimize()

    return u.X, m.getObjective().getValue()
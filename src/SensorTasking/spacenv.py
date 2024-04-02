import numpy as np
from .state import Spline

class SpaceEnv:
    """
    Represents the space environment with agents and targets.

    Attributes:
        M (int): Number of targets.
        N (int): Number of agents.
        maxsteps (int): Maximum number of steps.
        tstep (float): Time step.
        elapsed_steps (int): Number of steps elapsed.
        observers (np.ndarray): Array of observers.
        truths (np.ndarray): Array of truths/targets.

    Methods:
        __init__(agents, targets, maxsteps, tstep): Initializes the SpaceEnv with agents, targets, maximum steps, and time step.
        reset(): Resets the environment to its initial state.
        step(): Advances the environment by one step and returns termination status and observation Jacobians.
        _get_obs_jacobian(truth, observer): Computes the observation Jacobian between a truth and an observer.
        reset_new_agents(agents): Adds agents to environment and resets the environment.

    """
    def __init__(self, agents:np.ndarray[dict], targets: np.ndarray[dict], maxsteps: int, tstep: float):
        """
        Initializes the SpaceEnv with agents, targets, maximum steps, and time step.

        Parameters:
            agents (np.ndarray): Array containing information about agents.
            targets (np.ndarray): Array containing information about targets.
            maxsteps (int): Maximum number of steps.
            tstep (float): Time step.

        """
        self.M  = targets.size
        self.N = agents.size

        self.maxsteps = maxsteps
        self.tstep = tstep
        self.elapsed_steps = 0

        self.observers = np.array( [Spline( tstep=tstep, spl=agents[i]["spline"], stm_spl = agents[i]["stm_spline"], period=agents[i]["period"]) for i in range(self.N)] )
        self.truths = np.array( [Spline( tstep=tstep, spl=targets[i]["spline"], stm_spl = targets[i]["stm_spline"], period=targets[i]["period"]) for i in range(self.M)] )

    def reset(self):
        """
        Resets the environment to its initial state.

        Returns:
            None

        """
        self.elapsed_steps = 0

        for observer in self.observers:
            observer.reset()
        
        for truth in self.truths:
            truth.reset()

        return
    
    def step(self):
        """
        Advances the environment by one step and returns termination status and observation Jacobians.

        Returns:
            Tuple[bool, np.ndarray]: A tuple containing termination status and observation Jacobians.

        """

        self.elapsed_steps += 1

        for observer in self.observers:
            observer.propagate(steps=1)

        for truth in self.truths:
            truth.propagate(steps=1)

        H = np.ndarray(shape = (self.observers.size, self.truths.size), dtype=object)

        for i in range(self.observers.size):
            for j in range(self.truths.size):
                H[i, j] = self._get_obs_jacobian(self.truths[j], self.observers[i])

        
        terminated = ((self.elapsed_steps == self.maxsteps))

        return  terminated, H
    
    def _get_obs_jacobian(self, truth: Spline, observer: Spline):
        """
        Computes the observation Jacobian between a truth and an observer.

        Parameters:
            truth (Spline): Spline object representing the truth.
            observer (Spline): Spline object representing the observer.

        Returns:
            np.ndarray: Observation Jacobian matrix.

        """
        truthx = truth.spl(truth.t - truth.tstep/2)
        observerx = observer.spl(observer.t - observer.tstep/2)

        rOT = truthx[:3] - observerx[:3]
        vOT = truthx[3:] - observerx[3:]

        norm_rOT = np.linalg.norm(rOT)

        H11 = 1 / norm_rOT * np.eye(3) - np.outer(rOT, rOT) / norm_rOT**3
        H22 = H11
        H21 = - 1/norm_rOT**3 * np.outer(vOT, rOT) - 1/norm_rOT**3 * (np.outer(rOT, vOT) + np.dot(rOT, vOT)*np.eye(3)) + 3/ norm_rOT**5 * (np.dot(rOT, vOT)*np.outer(rOT, rOT))

        return np.block([[H11, np.zeros(shape=(3,3))], [H21, H22]])
    
    def reset_new_agents(self, agents_info: np.ndarray[dict]):
        """
        Resets the environment with new agents.

        Parameters:
            agents_info (np.ndarray): Array containing information about new agents.

        Returns:
            None

        """
        self.N = agents_info.size
        self.observers = np.array( [Spline( tstep=self.tstep, spl=agents_info[i]["spline"], stm_spl = agents_info[i]["stm_spline"], period=agents_info[i]["period"]) for i in range(self.N)] )
        self.reset()

        return




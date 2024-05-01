import numpy as np
from numpy.typing import ArrayLike

from .compute_coefficients import compute_coefficients, solve_model
from .spacenv import SpaceEnv
from .state import Spline
from data_util.target_generation import TargetGenerator


class SSA_Problem():
    """
    This class defines the Space Situational Awareness (SSA) Problem.
    
    Parameters:
        target_ics (list): A list of target initial conditions.
        target_periods (list): A list of target periods.
        agent_ics (list): A list of agent initial conditions.
        agent_periods (list): A list of agent periods.
    
    Attributes:
        ag (TargetGenerator): A generator/propagator for agent initial conditions.
        tg (TargetGenerator): A generator/propagator for target intial conditions.
        num_agents (int): Number of agents.
        tstep (float): Timestep for numerical propagation.
        period (float): Shortest period of amongst all targets and observers. This is the simulation time.
        maxsteps (int) : Maximum number of timesteps for simulation.
        env (SpaceEnv): Space environment object where targets and observers are propagated.
        min_target_period (ndarray): minimum period amongst all target orbits. Excludes agent orbits!
    
    Methods:
        fitness(x): This method evaluates the fitness of a decision vector 'x'.
        myopic_fitness(x): Evaluates fitness of decision vector assuming closest-target observation policy. 
        get_bounds(self): This method returns the bounds of the optimization problem. The bounds are [0, 1].
        _closest_target(observer): Returns the index of the closest target to given observer.
        get_bounds(): Returns the bounds of the decision vector.
        _gen_env(x): Updates the environment given the decision vector and resets environment to initial state.
    """
    def __init__(self, target_ics: list, target_periods:list,  agent_ics:list, agent_periods:list) -> None:
        self.tg = TargetGenerator(target_ics, periods=target_periods)
        targets = np.array([self.tg.gen_phased_ics(catalog_ID=i, num_targets=1, gen_P=False)[0] for i in range(self.tg.num_options)])

        self.ag = TargetGenerator(agent_ics, periods = agent_periods)
        self.num_agents = len(agent_periods)


        tmp_agents = self.ag.gen_phased_ics_from([0.0] * self.num_agents)
        
        self.tstep = 0.015
        self.period = np.min([np.min(agent_periods), np.min(target_periods)])
        self.maxsteps = int(np.floor(self.period/self.tstep))

        self.env = SpaceEnv(tmp_agents, targets, self.maxsteps, self.tstep)
        self.min_target_period = np.min(target_periods)

    def remove_agent(self, index:int = 0):
        """
        Removes the agent at the specified index in the list of agents and resets the Space Environment to initial state.

        Parameters:
            index (int): index of agent to remove
        """

        self.ag.remove_from_catalog(index)
        self.num_agents = self.ag.num_options

        if self.ag.periods.size == 0:
            self.period = self.min_target_period
        else:
            self.period = np.min([np.min(self.ag.periods), self.min_target_period])

        self.maxsteps = int(np.floor(self.period/self.tstep))
        self._gen_env(x=[0.0]*self.num_agents)

    def add_agent(self, agent_ic: np.ndarray[float], agent_period: float):
        """
        Adds an agent to the problem and resets the Space Environment to initial state.

        Parameters:
            agent_ic (np.ndarray[float]): Initial condition of the agent.
            agent_period (float): Period of the agent.
        """
        self.ag.add_to_catalog(agent_ic, agent_period)
        self.num_agents = self.ag.num_options

        self.period = np.min([agent_period, self.period])
        self.maxsteps = int(np.floor(self.period/self.tstep))

        self._gen_env(x=[0.0]*self.num_agents)

    def fitness(self, x: ArrayLike):
        """
        Computes the fitness of the current decision vector and returns value appropriate for pygmo.

        Parameters:
            x (ArrayLike): Decision vector.

        Returns:
            list: Negative objective value.
        """

        self._gen_env(x)
        information = compute_coefficients(self.env)
        control, objective = solve_model(information)

        return [-objective]
    
    def myopic_fitness(self, x):
        """
        Computes the fitness of the decision vector assuming a greedy closest-target based observation policy is used.

        Parameters:
            x (array-like): Decision vector.

        Returns:
            tuple: A tuple containing negative objective value, control matrix.
        """
        self._gen_env(x)
        information = compute_coefficients(self.env)

        self.env.reset()

        u = np.zeros_like(information, dtype=int)

        dists = np.zeros_like(information)

        for k in range(u.shape[0]):
            
            for i, observer in enumerate(self.env.observers):
                j = self._closest_target(observer)
                u[k, i, j] = 1
                dists[k, i, :] = np.array([np.linalg.norm(observer.x[:3] - target.x[:3]) for target in self.env.truths])
                # print(self.env.observers[0].x[:3], self.env.truths[0].x[:3])  Used for debugging, can delete

            out = self.env.step()
        
    
        return [-u.reshape(-1) @ information.reshape(-1)], u

    def _closest_target(self, observer: Spline):
        """
        Finds the index of the closest target to the observer.

        Parameters:
            observer (Spline): The observer.

        Returns:
            int: Index of the closest target.
        """

        dists = np.array([np.linalg.norm(observer.x[:3] - target.x[:3]) for target in self.env.truths])
        return np.argmin(dists)

    
    def get_bounds(self):
        """
        Gets the bounds for the optimization.

        Returns:
            tuple: A tuple containing lower and upper bounds for the agents.
        """
        return ([0] * self.num_agents, [1] * self.num_agents)
    
    def _gen_env(self, x: ArrayLike):
        """
        Updates the environment based on the decision vector. Resets the environment to initial state.

        Parameters:
            x (ArrayLike): Current state.

        """
        agents_info = self.ag.gen_phased_ics_from(x)
        self.env.reset_new_agents(agents_info=agents_info)




class Greedy_SSA_Problem(SSA_Problem):
    """
    This class represents the SSA_Problem suited for greedy optimization using pygmo.
    
    Parameters:
        target_ics (list): A list of target initial conditions.
        target_periods (list): A list of target periods.
        agent_ics (list): A list of agent initial conditions.
        agent_periods (list): A list of agent periods.
    
    Attributes:
        opt_phases (numpy.ndarray): An array containing the optimal phases found so far.
    
    Methods:
        fitness(self, x): This method evaluates the fitness of a given solution 'x'.
        get_bounds(self): This method returns the bounds of the optimization problem. The bounds are [0, 1].
    """
    def __init__(self, target_ics, target_periods,  agent_ics , agent_periods ) -> None:
        super().__init__(target_ics=target_ics, target_periods=target_periods, agent_ics=agent_ics, agent_periods=agent_periods)

        self.opt_phases = np.array([])

    def fitness(self, x):
        """
        Evaluates the fitness of a given design variable 'x'. Example, x = [0.5]
        
        Parameters:
            x (list): A list representing a design variable to the optimization problem.
        
        Returns:
            list: The fitness of the design variable 'x'.
        """

        # phase = np.hstack((self.opt_phases, x))

        return super().fitness(x)

    def get_bounds(self):
        """
        Returns the bounds of the optimization problem.
        
        Returns:
            tuple: A tuple containing the lower and upper bounds of the optimization problem. The bounds are [0, 1].
        """
        return ([0], [1])
        
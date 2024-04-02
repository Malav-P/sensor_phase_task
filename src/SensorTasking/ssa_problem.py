import numpy as np

from .compute_coefficients import compute_coefficients, solve_model
from .spacenv import SpaceEnv
from data_util.target_generation import TargetGenerator


class SSA_Problem():
    def __init__(self, target_ics, target_periods,  agent_ics, agent_periods) -> None:
        tg = TargetGenerator(target_ics, periods=target_periods)
        targets = np.array([tg.gen_phased_ics(catalog_ID=i, num_targets=1, gen_P=False)[0] for i in range(tg.num_options)])

        self.ag = TargetGenerator(agent_ics, periods = agent_periods)
        self.num_agents = len(agent_periods)
        tmp_agents = self.ag.gen_phased_ics_from([0.0] * self.num_agents)
        
        self.tstep = 0.015
        self.period = np.min([np.min(agent_periods), np.min(target_periods)])
        self.maxsteps = int(np.floor(self.period/self.tstep))

        self.env = SpaceEnv(tmp_agents, targets, self.maxsteps, self.tstep)

    def add_agent(self, agent_ic, agent_period):
        self.ag.add_to_catalog(agent_ic, agent_period)
        self.num_agents = self.ag.num_options

        self.period = np.min([self.period, agent_period])
        self.maxsteps = int(np.floor(self.period/self.tstep))

        self._gen_env(x=[0.0]*self.num_agents)

    def fitness(self, x):

        self._gen_env(x)
        information = compute_coefficients(self.env)
        control, objective = solve_model(information)

        return [-objective]
    
    def myopic_fitness(self, x):
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
        
    
        return [-u.reshape(-1) @ information.reshape(-1)], u, dists

    def _closest_target(self, observer):

        dists = np.array([np.linalg.norm(observer.x[:3] - target.x[:3]) for target in self.env.truths])
        return np.argmin(dists)

    
    def get_bounds(self):
        return ([0] * self.num_agents, [1] * self.num_agents)
    
    def _gen_env(self, x):
        agents_info = self.ag.gen_phased_ics_from(x)
        self.env.reset_new_agents(agents_info=agents_info)

        return
    
 
    
    ## TODO : (1) generate pygmo problem using this class and (2) solve the problem via GA


class Greedy_SSA_Problem(SSA_Problem):
    def __init__(self, target_ics, target_periods,  agent_ics , agent_periods ) -> None:
        super().__init__(target_ics=target_ics, target_periods=target_periods, agent_ics=agent_ics, agent_periods=agent_periods)

        self.opt_phases = np.array([])

    def fitness(self, x):

        phase = np.hstack((self.opt_phases, x))

        return super().fitness(phase)

    def get_bounds(self):
        return ([0], [1])
        
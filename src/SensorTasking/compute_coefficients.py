import numpy as np
import sys
import gurobipy as gp
from gurobipy import GRB

import pygmo
from typing import Optional

sys.path.append("../")

from .spacenv import SpaceEnv
from data_util.target_generation import TargetGenerator

def compute_coefficients(env):


    obs, info = env.reset()
    terminated = False
    
    dt = 0.1*env.observers[0].tstep

    sigma =  3 * np.pi / 180
    R_inv = 1 / sigma**2 * np.block([[np.eye(3), np.zeros(shape=(3, 3))], [np.zeros(shape=(3,3)), (0.5 * dt**2)*np.eye(3)]])


    information = np.zeros(shape=(env.maxsteps, env.observers.size, env.truths.size), dtype=float)

    for k in range(env.maxsteps):

        out = env.step()

        Hs = out[-1]


        for j,truth in enumerate(env.truths):
            stm = truth.eval_stm_spl( truth.t - truth.tstep / 2)
            stm_end = truth.eval_stm_spl(truth.period)

            # Phi(t2, t1) Phi(t1, t0)  = Phi(t2, t0)      ===>       Phi(t2, t1) = Phi(t2, t0) * Phi(t1, t0)^-1    ===== > Phi(t1, t2) = Phi(t1, t0)^-1 * Phi(t2, t0)^-1
            phi_j = stm.reshape(6, 6) @ np.linalg.inv(stm_end.reshape(6, 6))


            for i in range(env.observers.size):
                information[k, i ,j] = np.trace(phi_j.T @ Hs[i, j].T @ R_inv @ Hs[i,j] @ phi_j)

    

    return information

def solve_model(information):
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






class SSA_Problem():
    def __init__(self, target_ics, target_periods,  agent_ics, agent_periods) -> None:
        tg = TargetGenerator(target_ics, periods=target_periods)
        targets = tg.gen_phased_ics([1, 1], gen_P=True)

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
        agents = self.ag.gen_phased_ics_from(x)
        self.env.reset_new_agents(agents=agents)

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
        
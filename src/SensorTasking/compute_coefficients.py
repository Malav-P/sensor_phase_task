import numpy as np
import sys
import gurobipy as gp
from gurobipy import GRB
import pygmo

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

            phi_j = np.linalg.inv(stm_end.reshape(6, 6)) @ stm.reshape(6, 6)


            for i in range(env.observers.size):
                information[k, i ,j] = np.trace(phi_j.T @ Hs[i, j].T @ R_inv @ Hs[i,j] @ phi_j)

    

    return information

def solve_model(information):
    m = gp.Model("sensortask")

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
        tmp_agents = self.ag.gen_phased_ics_from([0.0])
        self.num_agents = len(agent_periods)

        self.maxsteps = 215
        self.tstep = 0.015

        self.env = SpaceEnv(tmp_agents, targets, self.maxsteps, self.tstep)

    def fitness(self, x):

        self._gen_env(x)
        information = compute_coefficients(self.env)
        control, objective = solve_model(information)

        return objective
    
    def get_bounds(self):
        return ([0] * self.num_agents, [1] * self.num_agents)
    
    def _gen_env(self, x):
        agents = self.ag.gen_phased_ics_from(x)
        self.env.reset_new_agents(agents=agents)

        return


import numpy as np
import sys
import gurobipy as gp
from gurobipy import GRB
import pygmo

sys.path.append("../")

from .spacenv import SpaceEnv
from .observation_model import ApparentMag
from .observation_spaces import Type1
from data_util.target_generation import TargetGenerator

def compute_coefficients(env):


    obs, info = env.reset()
    terminated = False

    dt = 0.1*env.observers[0].tstep

    sigma =  3 * np.pi / 180
    R_inv = 1 / sigma**2 * np.block([[np.eye(3), np.zeros(shape=(3, 3))], [np.zeros(shape=(3,3)), (0.5 * dt**2)*np.eye(3)]])


    information = np.zeros(shape=(env.maxsteps, env.observers.size, env.truths.size), dtype=float)

    for k in range(env.maxsteps):

        out = env.bare_step()

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
    def __init__(self, target_ics, target_periods,  agent_ics) -> None:
        tg = TargetGenerator(target_ics, periods=target_periods)
        self.targets = tg.gen_phased_ics([1, 1], gen_P=True)

        self.ag = TargetGenerator(agent_ics, periods = [6.45])
        self.num_agents = 1

        self.maxsteps = 215
        self.tstep = 0.015
        self.obs_class = Type1()

        # mass parameter of earth-moon system
        mu = 1.215058560962404e-02
        # apparent magnitude of sun
        ms = -26.4
        # specular reflection coefficient
        aspec = 0
        # diffuse reflection coefficient
        adiff = 0.2
        # diameter of target (LU)
        d = 0.001 / 384400
        # earth radius (LU)
        rearth = 6371 / 384400
        # moon radius (LU)
        rmoon = 1737.4 / 384400

        self.params = {
            "mu":mu,
            "ms":ms,
            "aspec":aspec,
            "adiff":adiff,
            "d":d,
            "rearth":rearth,
            "rmoon":rmoon
        }

        self.obs_model = ApparentMag(self.params, self.tstep)



    def fitness(self, x):

        env = self._gen_env(x)
        information = compute_coefficients(env)
        control, objective = solve_model(information)

        return objective
    
    def get_bounds(self):
        return ([0] * self.num_agents, [1] * self.num_agents)
    
    def _gen_env(self, x):
        agents = self.ag.gen_phased_ics_from(x)
        env = SpaceEnv(agents, self.targets, self.obs_model, self.maxsteps, self.tstep, obs_class=self.obs_class)


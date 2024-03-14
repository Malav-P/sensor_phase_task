import numpy as np
from .state import Spline

class SpaceEnv:
    def __init__(self, agents, targets, maxsteps, tstep):
        self.M  = targets.size
        self.N = agents.size

        self.maxsteps = maxsteps
        self.tstep = tstep
        self.elapsed_steps = 0

        self.observers = np.array( [Spline( tstep=tstep, spl=agents[i]["spline"], stm_spl = agents[i]["stm_spline"], period=agents[i]["period"]) for i in range(self.N)] )
        self.truths = np.array( [Spline( tstep=tstep, spl=targets[i]["spline"], stm_spl = targets[i]["stm_spline"], period=targets[i]["period"]) for i in range(self.M)] )



    
    def reset(self, seed=None, options=None):
        self.elapsed_steps = 0

        for observer in self.observers:
            observer.reset()
        
        for truth in self.truths:
            truth.reset()

        obs = self.get_observation()
        info = self.get_info()

        return obs, info
    
    def get_observation(self):
        return None
    
    def get_info(self):
        return None
    

    def step(self):

        self.elapsed_steps += 1

        for observer in self.observers:
            observer.propagate(steps=1)

        for truth in self.truths:
            truth.propagate(steps=1)

        H = np.ndarray(shape = (self.observers.size, self.truths.size), dtype=object)

        for i in range(self.observers.size):
            for j in range(self.truths.size):
                H[i, j] = self._get_obs_jacobian(self.truths[j], self.observers[i])

        
        reward = 0
        obs = self.get_observation()
        info = self.get_info()
        terminated = ((self.elapsed_steps == self.maxsteps))

        return obs, reward, terminated, False, info, H
    
    def _get_obs_jacobian(self, truth, observer):
        truthx = truth.spl(truth.t - truth.tstep/2)
        observerx = observer.spl(observer.t - observer.tstep/2)

        rOT = truthx[:3] - observerx[:3]
        vOT = truthx[3:] - observerx[3:]

        norm_rOT = np.linalg.norm(rOT)

        H11 = 1 / norm_rOT * np.eye(3) - np.outer(rOT, rOT) / norm_rOT**3
        H22 = H11
        H21 = - 1/norm_rOT**3 * np.outer(vOT, rOT) - 1/norm_rOT**3 * (np.outer(rOT, vOT) + np.dot(rOT, vOT)*np.eye(3)) + 3/ norm_rOT**5 * (np.dot(rOT, vOT)*np.outer(rOT, rOT))

        return np.block([[H11, np.zeros(shape=(3,3))], [H21, H22]])
    
    def reset_new_agents(self, agents):
        self.N = agents.size
        self.observers = np.array( [Spline( tstep=self.tstep, spl=agents[i]["spline"], stm_spl = agents[i]["stm_spline"], period=agents[i]["period"]) for i in range(self.N)] )
        obs, info = self.reset()




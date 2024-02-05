import numpy as np
from gymnasium import Env, spaces

from .filter import KalmanFilter
from .state import Dynamics
from .observation_model import some_function

class SpaceEnv(Env):
    def __init__(self, agents, targets, metric, maxsteps, tstep):
        self.M  = targets.size
        self.N = agents.size

        self.kalman_objects = np.array( [KalmanFilter(timestep=tstep, xof=targets[i]["state"], Pof=targets[i]["covariance"], func=targets[i]["f"], jac=targets[i]["jac"], f_params=targets[i]["f_params"], jac_params=targets[i]["jac_params"]) for i in range(self.M)] ) 
        self.observers = np.array( [Dynamics(x0=agents[i]["state"], tstep=tstep, f=agents[i]["f"], jac=agents[i]["jac"], f_params=agents[i]["f_params"], jac_params=agents[i]["jac_params"]) for i in range(self.N)] )
        self.truths = np.array( [Dynamics(x0=targets[i]["state"], tstep=tstep, f=targets[i]["f"], jac=targets[i]["jac"], f_params=targets[i]["f_params"], jac_params=targets[i]["jac_params"]) for i in range(self.M)] )

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.M,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([self.M+1 for i in range(self.N)])

        self.elapsed_steps = 0
        self.maxsteps = maxsteps

    def step(self, action):
        self.elapsed_steps += 1

        for observer in self.observers:
            observer.propagate(steps=1)

        for truth in self.truths:
            truth.propagate(steps=1)

        uniques, cs = np.unique(action, return_counts=True)
        counts = np.zeros(self.M+1)

        for tar, c in zip(uniques, cs):
            counts[tar] = c

        counts = counts[1:]

        for i, count in enumerate(counts):
            kalman_object = self.kalman_objects[i]
            
            if count > 0:
                truth = self.truths[i]
                observers = self.observers[np.where(action==i+1)[0]]
                Z, R_invs = some_function(truth, observers)

            else:
                Z, R_invs = None, None
            
            kalman_object.propagate(Z=Z, R_invs=R_invs)
        
        reward = self.get_reward()
        obs = self.get_observation()
        info = self.get_info()
        terminated = self.elapsed_steps == self.maxsteps
        return obs, reward, terminated, False, info
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.elapsed_steps = 0
        for kalman_object in self.kalman_objects:
            kalman_object.reset()

        for observer in self.observers:
            observer.reset()
        
        for truth in self.truths:
            truth.reset()

        obs = self.get_observation()
        info = self.get_info()

        return obs, info
    
    def get_reward(self, threshold = 0.0005):
        reward = 0

        for kalman_object in self.kalman_objects:
            tr_cov = np.trace(kalman_object.Pa)

            if tr_cov <= threshold:
                reward += 1

        reward *= 1 / self.M

        return reward
    
    def get_observation(self):
        tr_cov = np.zeros(self.kalman_objects.size, dtype=np.float32)

        for i, kalman_object in enumerate(self.kalman_objects):
            tr_cov[i] = np.trace(kalman_object.Pa)

        return tr_cov
    
    def get_info(self):
        return {"info" : None}

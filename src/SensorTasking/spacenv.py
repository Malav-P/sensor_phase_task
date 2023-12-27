import numpy as np
from gymnasium import Env, spaces

from .filter import KalmanFilter
from .state import Dynamics
from .observation_model import some_function

class SpaceEnv(Env):
    def __init__(self, agent_ICs, target_ICs, target_covs, metric, maxsteps, tstep, f, jac):
        self.M  = target_ICs.shape[0]
        self.N = agent_ICs.shape[0]

        self.kalman_objects = np.array( [KalmanFilter(timestep=tstep, xof=target_ICs[i, :-1], Pof=target_covs[:, :, i], func=f, jac=jac) for i in range(self.M)] ) 
        self.observers = np.array( [Dynamics(x0=agent_ICs[i,:-1], tstep=tstep, f=f, jac=jac) for i in range(self.N)] )
        self.truths = np.array( [Dynamics(x0=target_ICs[i, :-1], tstep=tstep, f=f, jac=jac) for i in range(self.M)] )

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
        super().init(seed=seed)
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
    
    def get_reward(self, threshold = 1):
        reward = 0

        for kalman_object in self.kalman_objects:
            tr_cov = np.trace(kalman_object.Pa)

            if tr_cov <= threshold:
                reward += 1

        reward *= 1 / self.M

        return reward
    
    def get_observation(self):
        tr_cov = np.zeros(self.kalman_objects.size)

        for i, kalman_object in enumerate(self.kalman_objects):
            tr_cov[i] = np.trace(kalman_object.Pa)

        return tr_cov
    
    def get_info(self):
        return None

import numpy as np
from gymnasium import Env, spaces

from .filter import KalmanFilter
from .state import Dynamics
from .rewards import reward1, reward2

class SpaceEnv(Env):
    def __init__(self, agents, targets, obs_model, maxsteps, tstep, obs_class = None, budget = 150):
        self.M  = targets.size
        self.N = agents.size

        self.kalman_objects = np.array( [KalmanFilter(timestep=tstep, xof=targets[i]["state"], Pof=targets[i]["covariance"], func=targets[i]["f"], jac=targets[i]["jac"], f_params=targets[i]["f_params"], jac_params=targets[i]["jac_params"]) for i in range(self.M)] ) 
        self.observers = np.array( [Dynamics(x0=agents[i]["state"], tstep=tstep, f=agents[i]["f"], jac=agents[i]["jac"], f_params=agents[i]["f_params"], jac_params=agents[i]["jac_params"]) for i in range(self.N)] )
        self.truths = np.array( [Dynamics(x0=targets[i]["state"], tstep=tstep, f=targets[i]["f"], jac=targets[i]["jac"], f_params=targets[i]["f_params"], jac_params=targets[i]["jac_params"]) for i in range(self.M)] )

        self.obs_class = obs_class
        self.obs_model = obs_model

        self.observation_space = obs_class.gen_observation_space(self.M)
        self.action_space = spaces.MultiDiscrete([self.M+1 for i in range(self.N)])

        self.elapsed_steps = 0
        self.maxsteps = maxsteps

        self.prev_action = None
        self.prev_available_actions = np.array([0, 1, 2])

        self.budget = budget
        self.n_obs = 0

    def step(self, action):
        self.prev_available_actions = self.get_available_actions()
        self.prev_action = action

        self.elapsed_steps += 1

        for state in self.obs_model.states:
            state.propagate(steps=1)

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
                self.n_obs += count
                truth = self.truths[i]
                observers = self.observers[np.where(action==i+1)[0]]
                Z, R_invs = self.obs_model.make_measurement(truth, observers, verbose=False)

            else:
                Z, R_invs = None, None
            
            kalman_object.propagate(Z=Z, R_invs=R_invs)
        
        reward = self.get_reward()
        obs = self.get_observation()
        info = self.get_info()
        terminated = ((self.elapsed_steps == self.maxsteps))
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

        for state in self.obs_model.states:
            state.reset()

        self.prev_action = None
        self.prev_available_actions = np.array([0,1, 2])
        self.n_obs = 0

        obs = self.get_observation()
        info = self.get_info()


        return obs, info
    
    def get_reward(self):
        return reward1(self)
    
    def get_observation(self):
        obs =  self.obs_class.get_observation(self)
        return obs
    
    def get_info(self):
        info = {}
        
        for i, kalman_object in enumerate(self.kalman_objects):
            info[f"target{i+1}_cov"] = np.trace(kalman_object.P)
        
        for i, observer in enumerate(self.observers):
            for j, truth in enumerate(self.truths):
                info[f"target{j+1}_obs{i+1}_apmag"] = self.obs_model._compute(truth, observer)

        info["prev_action"] = self.prev_action


        return info
    
    def get_available_actions(self):
        return self.obs_model.get_available_actions(self.truths, self.observers, self)
    
    def valid_action_mask(self):
        mask = [True, True, True]

        available_action = self.get_available_actions()

        for i,action in enumerate([0, 1, 2]):
            if action not in available_action:
                mask[i] = False

        return mask


